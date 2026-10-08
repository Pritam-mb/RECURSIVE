import { useEffect, useState } from 'react';
import { fmtCountdown, fmtDist, fmtPc, SEV_CLASS } from './cxGraph';

const num = (v, d = 2) => (v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toFixed(d));
const utc = (iso) => (iso ? `${String(iso).slice(0, 19).replace('T', ' ')}Z` : '—');
const pretty = (s) => (s ? String(s).replace(/_/g, ' ') : '—');

/** Static definitions — what each section's quantities ARE (the computed sentence says what they mean here). */
const DEFINE = {
  area: 'A hotspot is a region of space where predicted close approaches cluster at one instant (the TCA). Its radius comes from the encounter’s 3-σ position uncertainty (or the fragment spread for debris), so anything inside it at that moment is exposed.',
  debris: 'Fragments come from the NASA Standard Breakup Model and are propagated (RK4, two-body + J2 + drag) to the hotspot TCA. “Inside” counts the simulated fragments within the radius, scaled by how many real ≥10 cm fragments each sample represents.',
  closest: 'Miss distance is the minimum predicted separation. Pc is the probability the two hard bodies actually touch given the position uncertainty: Foster integrates the 2-D Gaussian (authoritative), Chan is an independent series, Alfano is the worst case if the covariance is mis-sized, Monte Carlo samples it directly.',
  avoid: 'Each option is a small impulsive burn (R radial, S along-track, W cross-track) that was re-propagated to get the new miss and Pc. Propellant follows the rocket equation; burn time = propellant × Isp × g₀ / thrust; “finite-burn OK” means the burn is short enough (≤10 % of the lead time and ≤1/10 orbit) to be treated as impulsive.',
  others: 'After a burn the satellite flies a new orbit, so it is re-screened for 24 h against the whole catalogue and live fragments. A secondary conjunction is a close approach the burn creates or worsens; CASCADE-SAFE means none reaches Pc ≥ 1e-6 or comes within 1 km.',
};

function Explain({ text, define }) {
  return (
    <div className="cx-explain">
      <span className="cx-explain-k">What this means</span>
      {text && <p className="cx-explain-main">{text}</p>}
      <p className="cx-explain-def">{define}</p>
    </div>
  );
}

function Section({ tag, title, children, right }) {
  return (
    <section className="cx-sec">
      <header className="cx-sec-head">
        <span className="cx-sec-tag">{tag}</span>
        <span className="cx-sec-title">{title}</span>
        {right && <span className="cx-sec-right">{right}</span>}
      </header>
      {children}
    </section>
  );
}

function KV({ items }) {
  return (
    <dl className="cx-kv">
      {items.filter(Boolean).map(([k, v, cls]) => (
        <div className="cx-kv-row" key={k}>
          <dt>{k}</dt>
          <dd className={cls ?? ''}>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Ticks once per second; minutes-to-TCA from the fetch-time value minus wall time elapsed (sim clock runs at 1×). */
function Countdown({ minutes, fetchedAt }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  if (minutes == null) return '—';
  return fmtCountdown(minutes - (now - fetchedAt) / 60000);
}

const safeLabel = (o) => {
  if (o.cascade_safe === true) return ['SAFE', 'cx-ok'];
  if (o.cascade_safe === false) return ['UNSAFE', 'cx-bad'];
  return [o.cascade_check ? pretty(o.cascade_check).toUpperCase() : 'NOT CHECKED', 'cx-dim'];
};

function EngineTable({ option }) {
  const engines = option?.engines ?? [];
  if (!engines.length) return <p className="cx-none">No engine costing for this option.</p>;
  const cryo = engines.find((e) => e.family?.startsWith('cryogenic'));
  return (
    <>
      <table className="cx-table cx-table--engines">
        <thead>
          <tr>
            <th>Engine</th><th>Family</th><th>Propellant</th><th className="r">Isp s</th><th className="r">Thrust N</th>
            <th className="r">Prop kg</th><th className="r">Burn s</th><th>Finite-burn</th>
          </tr>
        </thead>
        <tbody>
          {engines.map((e) => {
            const rec = e.engine === option.recommended_engine;
            const practical = e.practical_for_satellites !== false;
            return (
              <tr key={e.engine} className={`${rec ? 'is-rec' : ''}${practical ? '' : ' is-impractical'}`} title={e.note || ''}>
                <td>{rec && <span className="cx-rec-dot" aria-label="recommended" />}{e.engine}</td>
                <td>{pretty(e.family)}</td>
                <td>{e.propellant}</td>
                <td className="r">{num(e.isp_s, 0)}</td>
                <td className="r">{e.thrust_n >= 100 ? num(e.thrust_n, 0) : num(e.thrust_n, 3)}</td>
                <td className="r">{num(e.prop_mass_kg, 4)}</td>
                <td className="r">{e.burn_time_s >= 100 ? num(e.burn_time_s, 0) : num(e.burn_time_s, 3)}</td>
                <td className={e.finite_burn_ok ? 'cx-ok' : 'cx-bad'}>{e.finite_burn_ok ? 'OK' : 'NO'}{practical ? '' : ' · impractical'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="cx-note">
        <b>Recommended: {option.recommended_engine ?? '—'}</b>
        {option.engine_rule ? ` — ${option.engine_rule}` : ' — lowest propellant among practical engines that pass the finite-burn check.'}
      </p>
      {cryo && (
        <p className="cx-note cx-note--cryo">
          <b>Cryogenic {cryo.propellant} ({cryo.engine}):</b> {cryo.note}
        </p>
      )}
    </>
  );
}

function Avoidance({ entries, activeOpt, setActiveOpt, onHoverOption }) {
  if (!entries.length) {
    return <p className="cx-none">No operational payload in this zone can manoeuvre (debris and rocket bodies only).</p>;
  }
  return entries.map((a) => {
    if (!a.options?.length) {
      return (
        <div className="cx-avoid" key={a.alert_id + a.mover.id}>
          <div className="cx-avoid-head">{a.mover.name} <span className="cx-dim">vs {a.threat?.name}</span></div>
          <p className="cx-none">No burn planned: {pretty(a.maneuver_status)} (Pc {fmtPc(a.pc_before)}, miss {fmtDist(a.miss_before_km)}).</p>
        </div>
      );
    }
    const k = `${a.alert_id}`;
    const idx = activeOpt[k] ?? a.chosen_index ?? 0;
    const opt = a.options[idx] ?? a.options[0];
    return (
      <div className="cx-avoid" key={k}>
        <div className="cx-avoid-head">
          Move <b>{a.mover.name}</b> <span className="cx-dim">away from {a.threat?.name} · now {fmtDist(a.miss_before_km)}, Pc {fmtPc(a.pc_before)} · target Pc &lt; {fmtPc(a.target_pc)}</span>
        </div>
        <table className="cx-table cx-table--opts">
          <thead>
            <tr>
              <th>Option</th><th className="r">Δv m/s</th><th className="r">New miss</th><th className="r">New Pc</th>
              <th className="r">Fuel %</th><th>Cascade-safe</th><th className="r">Secondaries</th>
            </tr>
          </thead>
          <tbody>
            {a.options.map((o, i) => {
              const [lbl, cls] = safeLabel(o);
              return (
                <tr
                  key={o.candidate + i}
                  className={`cx-opt${i === idx ? ' is-active' : ''}${o.is_chosen ? ' is-chosen' : ''}`}
                  onClick={() => setActiveOpt({ ...activeOpt, [k]: i })}
                  onMouseEnter={() => onHoverOption({ mover: a.mover.id, option: o })}
                  onMouseLeave={() => onHoverOption(null)}
                >
                  <td>{o.is_chosen && <span className="cx-rec-dot" aria-label="chosen" />}{o.candidate}</td>
                  <td className="r">{num(o.delta_v_ms, 2)}</td>
                  <td className="r">{fmtDist(o.new_miss_distance_km)}</td>
                  <td className={`r ${o.achieves_target ? 'cx-ok' : ''}`}>{fmtPc(o.new_pc_collision)}</td>
                  <td className="r">{num(o.fuel_cost_pct, 3)}</td>
                  <td className={cls}>{lbl}</td>
                  <td className="r">{o.cascade_check === 'ok' ? (o.secondary_conjunctions?.length ?? 0) : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="cx-note">
          {a.selection_rule ? <>Selection rule: {a.selection_rule}. </> : null}
          Hover a row to draw its effect on the graph; click to see its engines.
          {opt.analytic_check?.rel_error != null && (
            <> Clohessy–Wiltshire check for {opt.candidate}: {num(opt.analytic_check.along_track_shift_km_at_tca, 3)} km vs numeric {num(opt.analytic_check.numeric_km, 3)} km ({num(opt.analytic_check.rel_error * 100, 1)} %).</>
          )}
        </p>
        <div className="cx-subhead">Engine comparison · {opt.candidate} ({num(opt.delta_v_ms, 2)} m/s, mass {a.mass_kg ?? '—'} kg{a.mass_source ? ` · ${pretty(a.mass_source)}` : ''})</div>
        <EngineTable option={opt} />
      </div>
    );
  });
}

function Effects({ hs, activeOpt, onHoverOption }) {
  const eff = hs.effect_on_others;
  const planned = hs.avoidance.filter((a) => a.options?.length);
  return (
    <>
      {planned.map((a) => (
        <div key={a.alert_id} className="cx-effects">
          {a.options.map((o, i) => {
            const active = (activeOpt[a.alert_id] ?? a.chosen_index ?? 0) === i;
            const secs = o.secondary_conjunctions ?? [];
            return (
              <div
                key={o.candidate + i}
                className={`cx-effect${active ? ' is-active' : ''}`}
                onMouseEnter={() => onHoverOption({ mover: a.mover.id, option: o })}
                onMouseLeave={() => onHoverOption(null)}
              >
                <div className="cx-effect-head">
                  <span>{a.mover.name} · {o.candidate}</span>
                  <span className={safeLabel(o)[1]}>{safeLabel(o)[0]}</span>
                </div>
                {o.cascade_check !== 'ok' ? (
                  <p className="cx-none">24 h secondary screening {pretty(o.cascade_check || 'not run')} — effect on others unknown.</p>
                ) : secs.length === 0 ? (
                  <p className="cx-none cx-ok">No new or worsened close approach within 24 h.</p>
                ) : (
                  <table className="cx-table">
                    <thead><tr><th>Object</th><th className="r">Miss</th><th className="r">Pc</th><th className="r">Before</th><th>TCA</th><th>Cause</th></tr></thead>
                    <tbody>
                      {secs.map((s) => (
                        <tr key={s.id} className={s.blocks_cascade_safe ? 'is-bad' : ''}>
                          <td>{s.name ?? s.id}</td>
                          <td className="r">{fmtDist(s.miss_km)}</td>
                          <td className="r">{fmtPc(s.pc)}</td>
                          <td className="r">{s.baseline_miss_km != null ? fmtDist(s.baseline_miss_km) : 'none'}</td>
                          <td>{s.tca_utc ? `${String(s.tca_utc).slice(11, 19)}Z` : '—'}</td>
                          <td>{s.burn_induced ? 'burn-induced' : 'pre-existing'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            );
          })}
        </div>
      ))}
      <KV items={[
        ['Options screened', `${eff.options_checked} / ${eff.options_total}`],
        ['Secondaries found', eff.secondary_total],
        ['Unsafe options', eff.unsafe_options, eff.unsafe_options ? 'cx-bad' : ''],
        ['Downstream objects', eff.downstream_count],
      ]}
      />
    </>
  );
}

export default function HotspotDetail({ hs, fetchedAt, onHoverOption, onSelectNode }) {
  const [activeOpt, setActiveOpt] = useState({});
  const { area, debris, closest_approach: ca, explanations: ex } = hs;
  const geo = area.geodetic;
  const pc = ca?.pc ?? {};
  return (
    <div className="cx-detail">
      <div className="cx-detail-title">
        <span className={`cx-badge ${SEV_CLASS[hs.severity] ?? ''}`}>{hs.severity ?? '—'}</span>
        <span className="cx-detail-name">Hotspot {Number(hs.id.split(':')[1]) + 1}</span>
        <span className="cx-dim">{pretty(hs.kind)} · score {fmtPc(hs.score)}</span>
      </div>

      <Section tag="A" title="Hotspot area" right={<Countdown minutes={area.tca_minutes} fetchedAt={fetchedAt} />}>
        <KV items={[
          ['Centre', geo ? `${num(Math.abs(geo.latitude_deg), 2)}°${geo.latitude_deg >= 0 ? 'N' : 'S'} ${num(Math.abs(geo.longitude_deg), 2)}°${geo.longitude_deg >= 0 ? 'E' : 'W'}` : '—'],
          ['Altitude', geo ? `${num(geo.altitude_km, 1)} km` : '—'],
          ['Radius', `${fmtDist(area.radius_km)} · ${pretty(area.radius_source)}`],
          ['TCA', utc(area.tca_utc)],
          ['Members', area.member_count],
        ]}
        />
        <div className="cx-members">
          {area.members.map((m) => (
            <button type="button" key={m.id} className="cx-member" onClick={() => onSelectNode(m.id)}>
              {m.name ?? m.id}<span className="cx-dim">{m.distance_km != null ? ` ${fmtDist(m.distance_km)}` : ''}</span>
            </button>
          ))}
        </div>
        <Explain text={ex?.area} define={DEFINE.area} />
      </Section>

      <Section tag="B" title="Debris layer" right={`${debris.fragments_inside_total ?? 0} frag in zone`}>
        {debris.events.length === 0 && (
          <KV items={[
            ['Breakup events', 'none simulated'],
            debris.forecast_cloud && ['If this pair collides', `~${debris.forecast_cloud.fragment_count} fragments ≥10 cm`],
            debris.forecast_cloud && ['Catastrophic', debris.forecast_cloud.is_catastrophic ? `yes (EMR ${num(debris.forecast_cloud.emr_j_per_g, 0)} J/g ≥ 40)` : 'no'],
            ['Fragment threats to members', debris.fragment_threats_to_members],
          ]}
          />
        )}
        {debris.events.map((e) => (
          <div key={e.event_id} className="cx-debris-ev">
            <div className="cx-subhead">
              {e.relation === 'own_breakup' ? 'Parent event (this pair)' : 'Other event'} · {e.event_id}
            </div>
            <KV items={[
              ['Parents', (e.parent_names || []).join(' × ')],
              ['Fragments (SBM ≥10 cm)', `${e.fragment_count_total} (simulated ${e.fragments_simulated}, ×${num(e.fragment_weight, 2)})`],
              ['State at TCA', e.state !== 'released' ? pretty(e.state)
                : Math.abs(e.t_rel_s) <= 1 ? 'breakup happens here at TCA' : `released ${num(e.t_rel_s / 60, 1)} min earlier`],
              ['Inside zone', e.represented_inside == null ? '—' : `${e.represented_inside} (${e.sampled_inside} sampled)`],
              ['Nearest fragment', fmtDist(e.nearest_fragment_km)],
              e.spread && Math.abs(e.t_rel_s) > 1 && ['Cloud spread p50 / p90', `${fmtDist(e.spread.p50_km)} / ${fmtDist(e.spread.p90_km)}`],
              e.spread?.along_track_spread_km != null && Math.abs(e.t_rel_s) > 1 && ['Along-track arc', fmtDist(e.spread.along_track_spread_km)],
            ]}
            />
            {e.spread_timeline?.length > 0 && (
              <table className="cx-table">
                <thead><tr><th>After breakup</th><th className="r">Alive</th><th className="r">p50</th><th className="r">p90</th><th className="r">Along-track</th></tr></thead>
                <tbody>
                  {e.spread_timeline.map((t) => (
                    <tr key={t.t_after_breakup_s}>
                      <td>{num(t.t_after_breakup_s / 60, 0)} min</td>
                      <td className="r">{t.n_alive}</td>
                      <td className="r">{fmtDist(t.p50_km)}</td>
                      <td className="r">{fmtDist(t.p90_km)}</td>
                      <td className="r">{fmtDist(t.along_track_spread_km)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ))}
        {debris.events.length > 0 && debris.fragment_threats_to_members > 0 && (
          <KV items={[['Fragment alerts on members', `${debris.fragment_threats_to_members} · closest ${fmtDist(debris.closest_fragment_miss_km)}`]]} />
        )}
        <Explain text={ex?.debris} define={DEFINE.debris} />
      </Section>

      <Section tag="C" title="Closest approach" right={ca ? <Countdown minutes={ca.tca_minutes} fetchedAt={fetchedAt} /> : null}>
        {ca ? (
          <>
            <div className="cx-readouts">
              <div><span className="cx-ro-k">Min distance</span><span className="cx-ro-v">{fmtDist(ca.miss_distance_km)}</span></div>
              <div><span className="cx-ro-k">Pc (Foster)</span><span className={`cx-ro-v ${SEV_CLASS[ca.severity] ?? ''}`}>{fmtPc(pc.foster)}</span></div>
              <div><span className="cx-ro-k">Rel. speed</span><span className="cx-ro-v">{num(ca.relative_speed_kms, 2)} km/s</span></div>
              <div><span className="cx-ro-k">Decision</span><span className="cx-ro-v">{ca.decision?.action ?? '—'}</span></div>
            </div>
            <KV items={[
              ['Pair', `${ca.pair[0].name} ↔ ${ca.pair[1].name}`],
              ['TCA', utc(ca.tca_utc)],
              ['B-plane (T, N)', ca.b_t_km != null ? `${fmtDist(Math.abs(ca.b_t_km))}, ${fmtDist(Math.abs(ca.b_n_km))}` : '—'],
              ['Hard-body radius', ca.hbr_km != null ? fmtDist(ca.hbr_km) : '—'],
              ['Covariance (3σ)', ca.covariance_ellipse?.a != null ? `${num(ca.covariance_ellipse.a / 1000, 2)} × ${num(ca.covariance_ellipse.b / 1000, 2)} km · ${pretty(ca.sigma_source)}` : '—'],
            ]}
            />
            <table className="cx-table cx-table--pc">
              <thead><tr><th>Pc method</th><th className="r">Value</th><th>Role</th></tr></thead>
              <tbody>
                <tr><td>Foster 2-D</td><td className="r">{fmtPc(pc.foster)}</td><td>authoritative</td></tr>
                <tr><td>Chan series</td><td className="r">{fmtPc(pc.chan)}</td><td>independent check</td></tr>
                <tr><td>Alfano max</td><td className="r">{fmtPc(pc.alfano_max)}</td><td>worst-case covariance</td></tr>
                <tr><td>Monte Carlo</td><td className="r">{fmtPc(pc.monte_carlo)}</td><td>{pc.monte_carlo == null && pc.mc_note ? pc.mc_note : 'sampling check'}</td></tr>
                <tr><td>ML surrogate</td><td className="r">{fmtPc(ca.ml_pc_surrogate)}</td><td>advisory only</td></tr>
              </tbody>
            </table>
            {pc.consistent != null && (
              <p className="cx-note">Methods {pc.consistent ? 'agree' : 'disagree'} (spread {num(pc.spread_decades, 2)} decades){ca.decision?.confidence ? ` · confidence ${ca.decision.confidence}` : ''}.</p>
            )}
          </>
        ) : <p className="cx-none">No alert in this zone has a computed closest approach.</p>}
        <Explain text={ex?.closest_approach} define={DEFINE.closest} />
      </Section>

      <Section tag="D" title="Collision-avoidance conditions">
        <Avoidance entries={hs.avoidance} activeOpt={activeOpt} setActiveOpt={setActiveOpt} onHoverOption={onHoverOption} />
        <Explain text={ex?.avoidance} define={DEFINE.avoid} />
      </Section>

      <Section tag="E" title="Effect on others">
        <Effects hs={hs} activeOpt={activeOpt} onHoverOption={onHoverOption} />
        <Explain text={ex?.effect_on_others} define={DEFINE.others} />
      </Section>
    </div>
  );
}
