/**
 * The eight demo-tour steps. Each `run` performs the real action against the
 * backend and returns the numbers it observed; `render` only formats what
 * `run` returned (no invented values). `ctx` carries results between steps.
 */
import useStore from '../../store/useStore';
import Kv from './Kv';
import {
  gdFetch, sleep, pairKey, alertPair, fmtPc, fmtNum, fmtMiss, fmtDuration,
} from './guideApi';

const SCENARIO = 'cascade_demo';

// ── DOM helpers (all optional: the tour still works if a panel is missing) ──
function clickAllTab() {
  const tab = Array.from(document.querySelectorAll('.tq-tab')).find((t) => t.textContent.trim() === 'ALL');
  if (tab && tab.getAttribute('aria-selected') !== 'true') tab.click();
}

function rowFor(alertId) {
  return document.getElementById(`threat-card-${alertId}`);
}

function scrollListTo(el, block = 'start') {
  if (!el) return;
  try { el.scrollIntoView({ block, behavior: 'smooth' }); } catch { el.scrollIntoView(); }
}

function sectionIn(row, label) {
  if (!row) return null;
  const lab = Array.from(row.querySelectorAll('.tq-section-label'))
    .find((l) => l.textContent.trim().toLowerCase() === label.toLowerCase());
  return lab?.parentElement || null;
}

async function expandRow(alert, freshAlerts, signal) {
  clickAllTab();
  let row = null;
  for (let i = 0; i < 16 && !row; i += 1) {
    row = rowFor(alert.id);
    if (!row) {
      // The queue refreshes every 30 s; hand it the alerts we just fetched.
      if (i === 4 && freshAlerts) useStore.getState().setAlerts(freshAlerts);
      await sleep(500, signal);
    }
  }
  if (!row) return false;
  if (!row.classList.contains('is-expanded')) row.querySelector('.tq-row-head')?.click();
  await sleep(150, signal);
  scrollListTo(rowFor(alert.id));
  return true;
}

async function fetchAlerts(signal) {
  return gdFetch('/api/alerts', { timeoutMs: 90000, signal });
}

/** Polling variant: a slow/busy backend is reported as progress, not as a failure. */
async function pollAlerts(signal, progress, label) {
  try {
    return await fetchAlerts(signal);
  } catch (e) {
    if (signal?.aborted) throw e;
    progress(`${label} (backend busy: ${e.message}; retrying)`);
    return null;
  }
}

function findPairAlert(data, pair) {
  return (data?.alerts || []).find((a) => a.source !== 'debris' && alertPair(a) === pair) || null;
}

// ─────────────────────────────────────────────────────────────────────────────
export const TOUR_STEPS = [
  {
    key: 'catalogue',
    title: 'The catalogue we watch',
    targets: ['.sh-readouts', 'section[aria-label="Live catalog globe"]'],
    async run({ signal }) {
      const [sats, alerts, clock] = await Promise.all([
        gdFetch('/api/satellites', { timeoutMs: 30000, signal }),
        fetchAlerts(signal),
        gdFetch('/api/simulation/time', { signal }).catch(() => null),
      ]);
      const list = sats?.satellites || [];
      const byAgency = {};
      for (const s of list) {
        const a = s.agency || 'Unknown';
        byAgency[a] = (byAgency[a] || 0) + 1;
      }
      const agencies = Object.entries(byAgency).sort((a, b) => b[1] - a[1]);
      const sev = { CRITICAL: 0, WARNING: 0, WATCH: 0 };
      let debris = 0;
      for (const a of alerts?.alerts || []) {
        if (a.severity in sev) sev[a.severity] += 1;
        if (a.source === 'debris') debris += 1;
      }
      return {
        objects: list.length,
        alerts: (alerts?.alerts || []).length,
        sev,
        debris,
        agencies,
        offset: clock?.offset_hours ?? null,
        simTime: clock?.simulation_time ?? null,
      };
    },
    render(d) {
      const named = d.agencies.filter(([a]) => a !== 'Unknown');
      return (
        <>
          <p className="gd-lead">
            Right now the backend propagates <b>{d.objects}</b> objects with SGP4 and screens every pair over the next 24 h.
            It found <b>{d.alerts}</b> predicted close approaches
            ({d.sev.CRITICAL} critical, {d.sev.WARNING} warning, {d.sev.WATCH} watch{d.debris ? `, ${d.debris} from debris` : ''}).
          </p>
          <p>
            The objects belong to <b>{named.length}</b> operators/owners (from the CelesTrak SATCAT), largest:{' '}
            {named.slice(0, 5).map(([a, n]) => `${a} ${n}`).join(' · ')}
            {d.agencies.some(([a]) => a === 'Unknown') ? ` · unknown owner ${d.agencies.find(([a]) => a === 'Unknown')[1]}` : ''}.
          </p>
          {d.offset ? (
            <p className="gd-warn">The simulation clock is shifted by {fmtNum(d.offset, 2)} h. Use “Reset demo” at the end to return to real time.</p>
          ) : (
            <p className="gd-dim">Simulation clock = real time ({String(d.simTime || '').substring(11, 19)} UTC).</p>
          )}
          <p className="gd-dim">Next: we inject a satellite on a crossing orbit and see whether the screening finds it.</p>
        </>
      );
    },
  },

  {
    key: 'scenario',
    title: 'Load a crossing scenario',
    targets: ['section[aria-label="Live catalog globe"]'],
    async run({ signal, ctx, progress }) {
      progress('POST /api/simulate {"name":"cascade_demo"} …');
      const res = await gdFetch('/api/simulate', { method: 'POST', body: { name: SCENARIO }, timeoutMs: 90000, signal });
      const a = res?.object_a;
      const b = res?.object_b;
      if (!a?.norad_id || !b?.norad_id) throw new Error('Scenario loaded but the response has no object pair');
      ctx.scenario = res;
      ctx.pair = pairKey(a.norad_id, b.norad_id);
      ctx.satA = Number(a.norad_id);
      ctx.satB = Number(b.norad_id);
      useStore.getState().setSelectedSatelliteId(Number(a.norad_id));
      return res;
    },
    render(r) {
      const c = r.construction || {};
      const a = r.object_a || {};
      const b = r.object_b || {};
      return (
        <>
          <p className="gd-lead">
            Scenario loaded. We picked a real catalogue object, <b>{a.name}</b> (NORAD {a.norad_id}, {fmtNum(a.mean_alt_km, 0)} km,
            {' '}{fmtNum(a.inclination_deg, 1)}°), in the busiest LEO shell
            {r.host_shell ? ` (${r.host_shell.population} objects at ${r.host_shell.alt_range_km?.join('–')} km)` : ''}, and fitted
            a synthetic object <b>{b.name}</b> ({fmtNum(b.inclination_deg, 1)}°) whose orbit passes through its predicted position.
          </p>
          <Kv rows={[
            ['Encounter in', `${fmtNum(c.encounter_minutes, 0)} min (${String(c.encounter_utc || '').substring(11, 19)} UTC)`],
            ['Relative speed', fmtNum(c.relative_speed_kms, 2, ' km/s')],
            ['Crossing angle', fmtNum(Math.abs(Number(c.crossing_angle_deg)), 1, '°'), 'Angle between the two velocity vectors at the encounter.'],
            ['Designed offset', fmtMiss(c.constructed_miss_km), 'Drawn from the stated covariance (seeded); a design input, not a result.'],
          ]}
          />
          <p className="gd-dim">
            These are design inputs only. Nobody tells the pipeline about the pair. Next, the 24 h screening has to find it on its own.
          </p>
        </>
      );
    },
  },

  {
    key: 'detection',
    title: 'Detection by screening',
    targets: ['#__gd_row__', '.tq-root'],
    panel: 'bottom',
    async run({ signal, ctx, progress }) {
      if (!ctx.pair) throw new Error('No scenario pair: run step 2 first');
      const t0 = performance.now();
      let alert = null;
      let data = null;
      while (!alert) {
        const secs = Math.round((performance.now() - t0) / 1000);
        if (secs > 300) throw new Error('The pair did not appear in /api/alerts within 300 s. The alert refresh may be slow; try Retry.');
        progress(`Waiting for the screening pipeline to report ${ctx.pair} … ${secs} s (it re-screens every 30 s and after each change)`);
        data = await pollAlerts(signal, progress, `Waiting for ${ctx.pair}`);
        alert = data ? findPairAlert(data, ctx.pair) : null;
        if (!alert) await sleep(3000, signal);
      }
      ctx.alert = alert;
      ctx.rowId = alert.id;
      progress('Found. Opening its threat card …');
      useStore.getState().setSelectedSatelliteId(ctx.satA);
      await expandRow(alert, data.alerts, signal);
      return { alert, detectedAfterS: Math.round((performance.now() - t0) / 1000) };
    },
    render({ alert: a, detectedAfterS }) {
      const pc = a.pc_checks || {};
      const d = a.decision || {};
      const ml = a.ml || {};
      return (
        <>
          <p className="gd-lead">
            The screening found <b>{a.sat1?.name}</b> × <b>{a.sat2?.name}</b>
            {detectedAfterS > 1 ? ` (${detectedAfterS} s after loading)` : ''} and ranked it <b className={`gd-sev is-${String(a.severity).toLowerCase()}`}>{a.severity}</b>.
            Its threat card is open in the queue.
          </p>
          <Kv rows={[
            ['Miss distance', fmtMiss(a.miss_distance_km), 'Closest distance at TCA, refined by Brent search on SGP4.'],
            ['TCA', `in ${fmtNum(a.tca_minutes, 1)} min (${String(a.tca_utc || '').substring(11, 19)} UTC)`],
            ['Relative speed', fmtNum(a.relative_speed_kms, 2, ' km/s')],
            ['Pc Foster', fmtPc(a.probability_of_collision), 'Authoritative: 2-D integral over the hard-body circle with TLE-age covariance.'],
            ['Pc Chan / Alfano max / MC', `${fmtPc(pc.chan)} / ${fmtPc(pc.alfano_max)} / ${pc.monte_carlo == null ? 'below MC resolution' : fmtPc(pc.monte_carlo)}`,
              'Independent formulas on the same B-plane. Alfano is an upper bound over covariance size.'],
            pc.spread_decades != null && ['Methods agree', `${pc.consistent ? 'yes' : 'no'} (spread ${fmtNum(pc.spread_decades, 2)} decades)`],
            ['Decision', `${fmtNum(d.score, 0)}/100 → ${d.action || 'n/a'}${d.model_agreement?.confidence ? ` · ${d.model_agreement.confidence} confidence` : ''}`,
              'Score = 60 % physics Pc + 15 % ML + 15 % cascade + 10 % manoeuvre cost. The action is set by the physics Pc thresholds.'],
            ['ML main factor', ml.main_factor_label || ml.main_factor
              ? `${ml.main_factor_label || ml.main_factor} (surrogate Pc ${fmtPc(ml.pc_surrogate)})` : 'not attached',
            'TreeSHAP: the feature that moved the XGBoost surrogate the most. Advisory only.'],
          ]}
          />
          {d.rationale && <p className="gd-dim">“{d.rationale}”</p>}
        </>
      );
    },
  },

  {
    key: 'avoidance',
    title: 'Avoidance options',
    targets: ['#__gd_row__', '.tq-root'],
    async run({ signal, ctx, progress }) {
      if (!ctx.pair) throw new Error('No scenario pair: run steps 2–3 first');
      progress('Fetching the current manoeuvre plan …');
      const data = await fetchAlerts(signal);
      const alert = findPairAlert(data, ctx.pair) || ctx.alert;
      if (!alert) throw new Error('The scenario alert is no longer in /api/alerts');
      ctx.alert = alert;
      await expandRow(alert, data.alerts, signal);
      const row = rowFor(alert.id);
      scrollListTo(sectionIn(row, 'Avoidance Options') || sectionIn(row, 'Recommended Maneuver'));
      return { alert };
    },
    render({ alert: a }) {
      const rec = a.recommended_maneuver;
      if (!rec) {
        return (
          <p className="gd-lead">
            No manoeuvre is planned for this pair right now ({a.maneuver_status || 'planner did not run'}). The planner only covers the
            top alerts with Pc ≥ 1e-6, within a 3 s budget per refresh.
          </p>
        );
      }
      const opts = Array.isArray(rec.options) ? rec.options : [];
      const idx = Number.isInteger(rec.chosen_index) ? rec.chosen_index : 0;
      const chosen = opts[idx] || rec;
      const engines = chosen.engines || [];
      const cryo = engines.find((e) => /cryo/i.test(e.family || ''));
      const recEng = engines.find((e) => e.engine === (chosen.recommended_engine || rec.recommended_engine));
      const safe = chosen.cascade_safe;
      return (
        <>
          <p className="gd-lead">
            The planner tried {rec.candidates_evaluated ?? 'several'} candidate burns, re-propagated each one and recomputed Pc.
            Chosen: <b>{chosen.candidate}</b> on {rec.sat_name || chosen.sat_name}.
          </p>
          <Kv rows={[
            ['Δv', fmtNum(chosen.delta_v_ms, 2, ' m/s')],
            ['New miss / Pc', `${fmtMiss(chosen.new_miss_distance_km)} / ${fmtPc(chosen.new_pc_collision)}`, 'From re-propagating the burned trajectory, not from a formula.'],
            ['Cascade-safe', safe === true ? 'yes: no new conjunction ≥ 1e-6 in 24 h'
              : safe === false ? `no: secondary Pc ${fmtPc(chosen.secondary_max_pc)}`
                : `not checked (${chosen.cascade_check || 'pending'})`,
            'The burned orbit is re-screened against the whole catalogue and live fragments for 24 h.'],
            ['Options compared', `${opts.length}${rec.selection_rule ? ` · rule: ${rec.selection_rule}` : ''}`],
            ['Engine', recEng
              ? `${recEng.engine} · ${fmtNum(recEng.prop_mass_kg * 1000, 1)} g ${recEng.propellant} · ${fmtNum(recEng.burn_time_s, 2)} s burn`
              : (chosen.recommended_engine || rec.recommended_engine || 'n/a'),
            'Lowest propellant mass among practical engines that can do the burn impulsively before TCA.'],
            ['Fuel used', `${fmtNum(chosen.fuel_cost_pct ?? rec.fuel_cost_pct, 2)} % of tank (rocket equation, ${rec.mass_source || 'assumed mass'})`],
          ]}
          />
          {cryo && (
            <p className="gd-dim">
              Why not {cryo.engine} (Isp {fmtNum(cryo.isp_s, 0)} s, would need only {fmtNum(cryo.prop_mass_kg * 1000, 1)} g)?{' '}
              {cryo.note?.split(';')[0] || 'Liquid hydrogen boils off, so cryogenic engines are used on upper stages, not satellites.'}.
            </p>
          )}
          <p className="gd-hint">“Approve” on the card would execute this burn. We leave it to you; next we show what happens if nobody acts.</p>
        </>
      );
    },
  },

  {
    key: 'collision',
    title: 'What if we do not act?',
    targets: ['section[aria-label="Live catalog globe"]'],
    async run({ signal, ctx, progress }) {
      if (!ctx.satA || !ctx.satB) throw new Error('No scenario pair: run step 2 first');
      const t0 = performance.now();
      const tick = setInterval(() => {
        const s = Math.round((performance.now() - t0) / 1000);
        progress(`Breaking up ${ctx.satA} × ${ctx.satB} at their predicted TCA (NASA Standard Breakup Model), then screening the fragments against the catalogue … ${s} s (can take 1–3 min on this laptop)`);
      }, 1000);
      let res;
      try {
        res = await gdFetch('/api/debris/simulate', {
          method: 'POST',
          body: { sat_a: ctx.satA, sat_b: ctx.satB, advance_to_impact: true },
          timeoutMs: 300000,
          signal,
        });
      } finally {
        clearInterval(tick);
      }
      const ev = res?.event;
      if (!ev?.event_id) throw new Error('Collision simulated but no event was returned');
      ctx.collision = res;
      ctx.eventId = ev.event_id;

      // Advance the sim clock to just past the collision if it is still ahead.
      progress('Checking the simulation clock …');
      const clock = await gdFetch('/api/simulation/time', { signal });
      const nowMs = Date.parse(clock?.simulation_time || '');
      const colMs = Date.parse(ev.collision_utc || '');
      let advancedTo = null;
      if (Number.isFinite(nowMs) && Number.isFinite(colMs) && colMs > nowMs) {
        const offset = Number(clock.offset_hours || 0) + (colMs - nowMs) / 3.6e6 + 0.03;
        progress(`Advancing the simulation clock by ${fmtDuration((colMs - nowMs) / 3.6e6 + 0.03)} to just after impact (alerts are recomputed) …`);
        await gdFetch('/api/simulation/time', { method: 'POST', body: { offset_hours: Number(offset.toFixed(4)) }, timeoutMs: 120000, signal });
        useStore.getState().setSimTimeOffset(Number(offset.toFixed(2)));
        advancedTo = offset;
      } else if (clock?.offset_hours != null) {
        useStore.getState().setSimTimeOffset(Number(Number(clock.offset_hours).toFixed(2)));
      }

      // Open the impact replay on this event and play it.
      progress('Loading the impact replay …');
      const st = useStore.getState();
      if (typeof st.setImpact === 'function') {
        st.setImpact({ eventId: ev.event_id, replay: null, playing: false });
        document.querySelector('.im-pill')?.click();
        let ok = false;
        for (let i = 0; i < 60 && !ok; i += 1) {
          await sleep(500, signal);
          const imp = useStore.getState().impact;
          if (imp?.eventId !== ev.event_id) useStore.getState().setImpact({ eventId: ev.event_id, replay: null, playing: false });
          if (imp?.replay?.event_id === ev.event_id) {
            const t = Number(imp.replay.t_rel_s?.[0] ?? -900);
            useStore.getState().setImpact({ playing: true, speed: 60, tRelS: t });
            ok = true;
          }
        }
        ctx.replayPlaying = ok;
      }
      return { res, advancedTo, replay: !!ctx.replayPlaying };
    },
    render({ res, advancedTo, replay }) {
      const ev = res.event || {};
      const parents = ev.parents || [];
      return (
        <>
          <p className="gd-lead">
            If nobody manoeuvres, <b>{(ev.parent_names || []).join(' and ')}</b> collide at{' '}
            {String(ev.collision_utc || '').substring(11, 19)} UTC at {fmtNum(ev.rel_vel_kms ?? ev.stats?.relative_velocity_kms, 2, ' km/s')}.
          </p>
          <Kv rows={[
            ['Energy / mass', `${fmtNum(ev.emr_j_per_g, 0)} J/g → ${ev.is_catastrophic ? 'catastrophic' : 'non-catastrophic'} (threshold ${fmtNum(ev.catastrophic_threshold_j_per_g, 0)} J/g)`,
              'NASA SBM: catastrophic when ½·m·v² / target mass ≥ 40 J/g; then both objects fragment completely.'],
            ['Masses', parents.map((p) => `${p.name?.split(' (')[0]} ${fmtNum(p.mass_kg, 0)} kg (${p.mass_source})`).join(' · '), 'SATCAT has no masses, so these are tagged assumptions.'],
            ['Fragments ≥ 10 cm', `${res.total_fragments} (N = 0.1·M^0.75·Lc^-1.71), ${res.fragments_generated} propagated`],
            advancedTo != null && ['Clock', `advanced to Sim +${fmtNum(advancedTo, 2)} h (just after impact)`],
          ]}
          />
          <p className="gd-dim">
            {replay
              ? 'The impact replay on the left globe is playing at 60× from T−15 min: real propagated fragment positions, sampled every 30 s.'
              : 'The impact replay console appears on the left globe once the replay has loaded.'}
          </p>
        </>
      );
    },
  },

  {
    key: 'debris',
    title: 'Who the debris threatens',
    targets: ['.tq-root'],
    async run({ signal, ctx, progress }) {
      if (!ctx.eventId) throw new Error('No collision event: run step 5 first');
      const t0 = performance.now();
      let mine = [];
      let data = null;
      for (;;) {
        const s = Math.round((performance.now() - t0) / 1000);
        progress(`Looking for fragment alerts of ${ctx.eventId} … ${s} s`);
        data = await pollAlerts(signal, progress, 'Looking for fragment alerts') || data;
        mine = (data?.alerts || []).filter((a) => a.source === 'debris' && a.parent_event?.event_id === ctx.eventId);
        if (mine.length || s > 120) break;
        await sleep(3000, signal);
      }
      clickAllTab();
      if (mine.length) useStore.getState().setAlerts(data.alerts);
      const bySat = new Map();
      for (const a of mine) {
        const id = a.sat1?.id;
        const prev = bySat.get(id);
        if (!prev || Number(a.probability_of_collision) > Number(prev.probability_of_collision)) bySat.set(id, a);
      }
      const sats = [...bySat.values()].sort((x, y) => Number(y.probability_of_collision) - Number(x.probability_of_collision));
      const agencies = [...new Set(sats.map((a) => a.sat1?.agency || 'Unknown'))];
      // Show the first debris row in the queue.
      const first = sats[0] && rowFor(sats[0].id);
      if (first) scrollListTo(first);
      return { alerts: mine.length, sats, agencies, collision: ctx.collision };
    },
    render({ alerts, sats, agencies, collision }) {
      const meta = collision?.debris_screening || {};
      if (!alerts) {
        return (
          <p className="gd-lead">
            The {collision?.total_fragments ?? ''} fragments were screened against {meta.satellites_screened ?? 'all'} satellites
            over {meta.window_hours ?? 6} h and none came within {meta.threshold_km ?? 5} km: no debris alerts for this event.
          </p>
        );
      }
      return (
        <>
          <p className="gd-lead">
            The NASA SBM cloud of <b>{collision?.total_fragments}</b> fragments was screened against {meta.satellites_screened ?? 'all'} satellites
            over {meta.window_hours ?? 6} h: <b>{alerts}</b> fragment alerts threaten <b>{sats.length}</b> satellites operated by{' '}
            <b>{agencies.join(', ')}</b>. They appear in the queue tagged DEBRIS.
          </p>
          <table className="gd-table">
            <thead><tr><th>Satellite</th><th>Agency</th><th>Miss</th><th>Pc</th><th>TCA</th></tr></thead>
            <tbody>
              {sats.slice(0, 6).map((a) => (
                <tr key={a.id}>
                  <td>{a.sat1?.name}</td>
                  <td>{a.sat1?.agency || 'Unknown'}</td>
                  <td>{fmtMiss(a.miss_distance_km)}</td>
                  <td>{fmtPc(a.probability_of_collision)}</td>
                  <td>{String(a.tca_utc || '').substring(11, 16)}Z</td>
                </tr>
              ))}
            </tbody>
          </table>
          {sats.length > 6 && <p className="gd-dim">… and {sats.length - 6} more.</p>}
        </>
      );
    },
  },

  {
    key: 'cascade',
    title: 'The cascade',
    targets: ['.an-cascade'],
    panel: 'top',
    async run({ signal, ctx }) {
      const data = await fetchAlerts(signal);
      const all = data?.alerts || [];
      const linked = all.filter((a) => a.parent_event?.event_id === ctx.eventId || a.upstream_event === ctx.eventId);
      const names = {};
      for (const a of all) {
        if (a.sat1?.id != null) names[a.sat1.id] = a.sat1.name;
        if (a.sat2?.id != null) names[a.sat2.id] = a.sat2.name;
      }
      const probs = Object.entries(data?.node_probabilities || {})
        .map(([id, p]) => [id, Number(typeof p === 'object' && p ? (p.p_hit ?? p.probability ?? p.p) : p)])
        .filter(([, p]) => Number.isFinite(p) && p > 0)
        .sort((a, b) => b[1] - a[1]);
      const maxDepth = linked.reduce((m, a) => Math.max(m, Number(a.cascade_depth) || 0), 0);
      return {
        depth: data?.cascade_depth ?? null,
        maxLinkedDepth: maxDepth,
        linked: linked.length,
        nodes: data?.graph?.node_count ?? data?.graph?.nodes?.length ?? null,
        edges: data?.graph?.edge_count ?? data?.graph?.edges?.length ?? null,
        top: probs.slice(0, 4).map(([id, p]) => [names[id] || id, p]),
      };
    },
    render(d) {
      return (
        <>
          <p className="gd-lead">
            The Cascade Risk Graph (highlighted) links every alert: collision event → fragments → threatened satellites → their own conjunctions.
            {d.nodes != null && <> It currently has <b>{d.nodes}</b> nodes and <b>{d.edges}</b> edges.</>}
          </p>
          <Kv rows={[
            ['Depth', `${d.maxLinkedDepth || d.depth || 'n/a'} hops from the collision${d.linked ? ` (${d.linked} alerts trace back to it)` : ''}`,
              'BFS hops from the collision event: fragment 1, threatened satellite 2, that satellite’s own conjunctions 3 …'],
            d.top.length > 0 && ['Highest P(hit)', d.top.map(([n, p]) => `${n} ${fmtPc(p)}`).join(' · '),
              'P(hit) = 1 − Π(1 − Pc) over the node’s alerts, assuming independent encounters.'],
          ]}
          />
          <p className="gd-dim">
            Drag or zoom the graph; red nodes are critical. This is a 24 h prediction graph, not a decades-long Kessler population model.
          </p>
        </>
      );
    },
  },

  {
    key: 'finish',
    title: 'Back to a clean state',
    targets: [],
    async run() { return {}; },
    render(_d, { onReset, resetState }) {
      return (
        <>
          <p className="gd-lead">
            That is the full loop: screening finds the threat, four Pc methods agree, the planner proposes a cascade-safe burn, and if
            nobody acts the breakup model shows who is hit next. Every number above came from the API at the moment you saw it.
          </p>
          <p>Reset removes the scenario object and debris and returns the clock to real time.</p>
          <div className="gd-row">
            <button type="button" className="ui-btn ui-btn--primary" onClick={onReset} disabled={resetState?.status === 'running'}>
              {resetState?.status === 'running' ? 'Resetting…' : 'Reset demo'}
            </button>
            {resetState?.status === 'done' && <span className="gd-ok">Clean: clock at real time, debris and scenario cleared.</span>}
            {resetState?.status === 'error' && <span className="gd-err">{resetState.error}</span>}
            {resetState?.status === 'running' && <span className="gd-dim">{resetState.progress}</span>}
          </div>
        </>
      );
    },
  },
];

export async function resetDemo(progress = () => {}, signal) {
  progress('Returning the simulation clock to real time …');
  await gdFetch('/api/simulation/time', { method: 'POST', body: { offset_hours: 0 }, timeoutMs: 120000, signal });
  progress('Clearing debris …');
  await gdFetch('/api/debris/active', { method: 'DELETE', timeoutMs: 60000, signal });
  progress('Removing the scenario object …');
  await gdFetch('/api/simulate', { method: 'DELETE', timeoutMs: 60000, signal });
  const st = useStore.getState();
  st.setSimTimeOffset(0);
  if (typeof st.setImpact === 'function') st.setImpact({ eventId: null, replay: null, playing: false });
  st.setSelectedSatelliteId(null);
}
