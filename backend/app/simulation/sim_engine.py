"""
Simulation engine.
Loads deterministic scenario JSONs and overrides satellite states.
Applies delta-V maneuvers for demo purposes.
"""

import json
import os
import logging
from datetime import datetime, timedelta, timezone
from app.core.sgp4_propagator import SGP4Propagator
from app.core.conjunction import WARNING_DISTANCE_KM, find_tca
from app.core.agency import AGENCY_ALIASES as _AGENCY_ALIASES, infer_agency as _infer_shared_agency

try:
    from app.core.agency_authority import authority_manager
    _AUTHORITY_AVAILABLE = True
except ImportError:
    authority_manager = None
    _AUTHORITY_AVAILABLE = False

logger = logging.getLogger(__name__)

AGENCY_ALIASES = _AGENCY_ALIASES

SCENARIOS_DIR = os.path.join(os.path.dirname(__file__))


RE_KM = 6378.137
MU_WGS72 = 398600.8


def _jd(t: datetime) -> tuple[float, float]:
    from sgp4.api import jday
    return jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)


def _rv_to_elements_batch(r, v, mu: float = MU_WGS72) -> dict:
    """Vectorised osculating elements (a, e, i, raan, argp, mean anomaly), robust for e ~ 0."""
    import numpy as np
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = np.linalg.norm(r, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    energy = 0.5 * np.einsum("ij,ij->i", v, v) - mu / rn
    a = -mu / (2.0 * energy)
    e_vec = (np.cross(v, h) / mu) - r / rn[:, None]
    e = np.linalg.norm(e_vec, axis=1)
    inc = np.arccos(np.clip(h[:, 2] / hn, -1.0, 1.0))
    node = np.stack([-h[:, 1], h[:, 0], np.zeros_like(hn)], axis=1)
    nn = np.linalg.norm(node, axis=1)
    node_hat = np.where(nn[:, None] > 1e-12, node / np.maximum(nn, 1e-300)[:, None], np.array([1.0, 0, 0]))
    raan = np.arctan2(node_hat[:, 1], node_hat[:, 0]) % (2 * np.pi)
    h_hat = h / hn[:, None]
    m_hat = np.cross(h_hat, node_hat)
    # argument of latitude (well defined for circular orbits)
    u_lat = np.arctan2(np.einsum("ij,ij->i", r, m_hat), np.einsum("ij,ij->i", r, node_hat))
    argp = np.arctan2(np.einsum("ij,ij->i", e_vec, m_hat), np.einsum("ij,ij->i", e_vec, node_hat))
    nu = u_lat - argp
    E = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    M = (E - e * np.sin(E)) % (2 * np.pi)
    return {"a": a, "e": e, "i_deg": np.degrees(inc), "i": inc, "raan": raan,
            "argp": argp % (2 * np.pi), "M": M}


def _fit_satrec_to_state(r_target, v_target, epoch: datetime, satnum: int, iters: int = 25):
    """Fit SGP4 mean elements whose SGP4 state at `epoch` equals (r, v).

    Fixed-point differential correction: osculating -> elements -> sgp4init ->
    SGP4 state at tsince=0; the residual is added back to the input vector.
    SGP4 short-period terms are ~1e-3 of the state, so this converges quickly.
    """
    import numpy as np
    from sgp4.api import Satrec, WGS72

    r_t = np.asarray(r_target, float)
    v_t = np.asarray(v_target, float)
    r_in, v_in = r_t.copy(), v_t.copy()
    jd, fr = _jd(epoch)
    sat = None
    dr = dv = float("inf")
    for it in range(iters):
        el = _rv_to_elements_batch(r_in[None, :], v_in[None, :])
        n_rad_min = np.sqrt(MU_WGS72 / el["a"][0] ** 3) * 60.0
        sat = Satrec()
        sat.sgp4init(WGS72, "i", int(satnum), (jd + fr) - 2433281.5, 0.0, 0.0, 0.0,
                     float(max(el["e"][0], 1e-8)), float(el["argp"][0]), float(el["i"][0]),
                     float(el["M"][0]), float(n_rad_min), float(el["raan"][0]))
        code, r_o, v_o = sat.sgp4(jd, fr)
        if code != 0:
            raise ValueError(f"SGP4 error {code} while fitting synthetic object")
        er = r_t - np.array(r_o)
        ev = v_t - np.array(v_o)
        dr, dv = float(np.linalg.norm(er)), float(np.linalg.norm(ev))
        if dr < 1e-6 and dv < 1e-9:
            break
        r_in += er
        v_in += ev
    return sat, {"iterations": it + 1, "residual_pos_m": round(dr * 1000.0, 6),
                 "residual_vel_mm_s": round(dv * 1e6, 6)}


def _to_native(value):
    """Recursively convert numpy scalars/arrays into JSON-safe Python types."""
    import numpy as np

    if isinstance(value, dict):
        return {key: _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class SimEngine:
    """Manages deterministic simulation scenarios."""

    def __init__(self, propagator: SGP4Propagator):
        self.propagator = propagator
        self.active_scenario = None
        self._scenario_sats = []
        self._scenario_backup = {}
        self.scenario_pair = None   # (real_id, synthetic_id) of a computed crossing

    # ── Computed crossing scenario ───────────────────────────────────────
    def _load_computed_crossing(self, scenario: dict, scenario_name: str) -> dict:
        """
        Build an encounter relative to the simulation clock:
          1. host shell = densest (mean-altitude x inclination) bin of the live
             LEO catalogue (computed from SGP4 states now);
          2. object A = real catalogue object closest to that bin's median;
          3. A's state at t_enc = now + encounter_minutes (SGP4);
          4. B's velocity = A's horizontal velocity rotated about the local
             vertical by the crossing angle that yields a relative speed drawn
             (seeded) from rel_speed_kms_range; B keeps A's radius, speed and
             flight-path angle, so it is a physical orbit of the same shape;
          5. B's position = A's + an offset drawn from N(0, diag(sigma_RTN^2))
             projected on the encounter plane (perpendicular to v_rel), so the
             constructed miss distance equals |offset|;
          6. B's SGP4 mean elements are fitted (fixed-point differential
             correction) so SGP4(B, t_enc) reproduces that state to < 1 mm;
             SGP4 then back-propagates B to now.
        B is registered in the propagator like any TLE object, so screening
        has to find the conjunction itself.
        """
        import numpy as np
        from sgp4.api import SatrecArray, WGS72
        from app.core import sim_clock

        cfg = dict(scenario.get("crossing") or {})
        now = sim_clock.simulation_now()
        seed = cfg.get("seed")
        if seed is None:
            seed = int(now.timestamp()) & 0x7FFFFFFF
        rng = np.random.default_rng(int(seed))
        lead_min = float(cfg.get("encounter_minutes", 30.0))
        alt_bin = float(cfg.get("shell_alt_bin_km", 50.0))
        inc_bin = float(cfg.get("shell_inc_bin_deg", 5.0))
        syn_id = int(cfg.get("synthetic_norad_id", 99901))
        syn_name = cfg.get("synthetic_name", "SCN CROSSER-B (synthetic)")

        with self.propagator._lock:
            table = [(nid, entry) for nid, entry in self.propagator._satellites.items()
                     if nid != syn_id and nid not in self._scenario_sats]
        if not table:
            raise ValueError("No catalogue objects loaded")
        ids = np.array([nid for nid, _ in table])
        names = [entry[1] for _, entry in table]
        satrecs = [entry[0] for _, entry in table]

        jd, fr = _jd(now)
        err, r, v = SatrecArray(satrecs).sgp4(np.array([jd]), np.array([fr]))
        r, v, ok = r[:, 0, :], v[:, 0, :], err[:, 0] == 0
        el = _rv_to_elements_batch(r, v)
        mean_alt = el["a"] - RE_KM
        leo = ok & (el["e"] < 0.02) & (mean_alt > 300.0) & (mean_alt < 2000.0)
        if not leo.any():
            raise ValueError("No near-circular LEO objects to host a crossing")
        alt_idx = np.floor(mean_alt / alt_bin).astype(int)
        inc_idx = np.floor(el["i_deg"] / inc_bin).astype(int)
        keys = np.stack([alt_idx[leo], inc_idx[leo]], axis=1)
        uniq, counts = np.unique(keys, axis=0, return_counts=True)
        best = uniq[int(np.argmax(counts))]
        in_bin = leo & (alt_idx == best[0]) & (inc_idx == best[1])
        cand = np.nonzero(in_bin)[0]
        is_debris = np.array([(" DEB" in names[i].upper()) or ("R/B" in names[i].upper()) for i in cand])
        if (~is_debris).any():
            cand = cand[~is_debris]
        med_alt = float(np.median(mean_alt[cand]))
        med_inc = float(np.median(el["i_deg"][cand]))
        score = np.abs(mean_alt[cand] - med_alt) / alt_bin + np.abs(el["i_deg"][cand] - med_inc) / inc_bin
        a_row = int(cand[int(np.argmin(score))])
        a_id, a_name, a_sat = int(ids[a_row]), names[a_row], satrecs[a_row]

        t_enc = now + timedelta(minutes=lead_min)
        jd_e, fr_e = _jd(t_enc)
        e_code, r_a, v_a = a_sat.sgp4(jd_e, fr_e)
        if e_code != 0:
            raise ValueError(f"SGP4 error {e_code} for host object {a_id}")
        r_a, v_a = np.array(r_a), np.array(v_a)

        r_hat = r_a / np.linalg.norm(r_a)
        v_rad = (v_a @ r_hat) * r_hat
        v_hor = v_a - v_rad
        lo, hi = cfg.get("rel_speed_kms_range", [8.0, 12.0])
        rel_target = float(rng.uniform(lo, hi))
        ratio = min(1.0, rel_target / (2.0 * np.linalg.norm(v_hor)))
        theta = 2.0 * np.arcsin(ratio) * (1.0 if rng.random() < 0.5 else -1.0)
        v_hor_b = v_hor * np.cos(theta) + np.cross(r_hat, v_hor) * np.sin(theta)
        v_b = v_rad + v_hor_b
        u = v_b - v_a
        u_hat = u / np.linalg.norm(u)

        sig = np.array(cfg.get("miss_sigma_rtn_km", [0.02, 0.08, 0.04]), float)
        d_rtn = rng.normal(0.0, sig)
        n_hat = np.cross(r_a, v_a); n_hat /= np.linalg.norm(n_hat)
        t_hat = np.cross(n_hat, r_hat)
        delta = d_rtn[0] * r_hat + d_rtn[1] * t_hat + d_rtn[2] * n_hat
        delta_perp = delta - (delta @ u_hat) * u_hat
        r_b = r_a + delta_perp

        sat_b, fit = _fit_satrec_to_state(r_b, v_b, t_enc, syn_id)
        e_now, rb_now, vb_now = sat_b.sgp4(jd, fr)

        with self.propagator._lock:
            if syn_id in self.propagator._satellites:
                self._scenario_backup[syn_id] = self.propagator._satellites[syn_id]
            self.propagator._satellites[syn_id] = (sat_b, syn_name)
        self._scenario_sats.append(syn_id)
        self.scenario_pair = (a_id, syn_id)
        try:
            from app.core.debris_model import register_object_meta
            register_object_meta(syn_id, mass_kg=float(cfg.get("synthetic_mass_kg", 800.0)),
                                 hbr_m=float(cfg.get("synthetic_hbr_m", 2.0)),
                                 object_type=cfg.get("synthetic_object_type", "PAY"),
                                 mass_source="scenario_config")
        except Exception:
            pass

        el_b = _rv_to_elements_batch(np.array([r_b]), np.array([v_b]))
        construction = {
            "seed": int(seed),
            "encounter_utc": t_enc.isoformat(),
            "encounter_minutes": lead_min,
            "relative_speed_kms": round(float(np.linalg.norm(u)), 4),
            "crossing_angle_deg": round(float(np.degrees(theta)), 3),
            "miss_sigma_rtn_km": sig.tolist(),
            "drawn_offset_rtn_km": [round(float(x), 5) for x in d_rtn],
            "constructed_miss_km": round(float(np.linalg.norm(delta_perp)), 5),
            "sgp4_fit": fit,
        }
        host = {
            "alt_range_km": [float(best[0] * alt_bin), float((best[0] + 1) * alt_bin)],
            "inc_range_deg": [float(best[1] * inc_bin), float((best[1] + 1) * inc_bin)],
            "population": int(in_bin.sum()),
            "leo_population": int(leo.sum()),
            "catalog_size": int(len(ids)),
        }
        self.active_scenario = {**scenario, "construction": construction, "host_shell": host,
                                "pair": [a_id, syn_id]}
        logger.info("Computed crossing: host %s (%d) in shell %s, B=%d, miss %.3f km, v_rel %.2f km/s",
                    a_name, a_id, host, syn_id, construction["constructed_miss_km"],
                    construction["relative_speed_kms"])
        return {
            "name": scenario.get("name", scenario_name),
            "description": scenario.get("description", ""),
            "type": "computed_crossing",
            "tca_minutes": lead_min,
            "satellites_injected": 1,
            "host_shell": host,
            "object_a": {"norad_id": a_id, "name": a_name, "source": "catalog",
                         "mean_alt_km": round(float(mean_alt[a_row]), 2),
                         "inclination_deg": round(float(el["i_deg"][a_row]), 3)},
            "object_b": {"norad_id": syn_id, "name": syn_name, "source": "synthetic_sgp4_fit",
                         "mean_alt_km": round(float(el_b["a"][0] - RE_KM), 2),
                         "inclination_deg": round(float(el_b["i_deg"][0]), 3),
                         "state_now_ok": e_now == 0,
                         "position_now_eci_km": [round(float(x), 3) for x in rb_now]},
            "construction": construction,
            "note": "Constructed values are design inputs; the conjunction itself is detected by the screening pipeline.",
        }

    def _infer_agency(self, name: str) -> str:
        return _infer_shared_agency(name)

    def _clone_propagator(self) -> SGP4Propagator:
        # Copies TLEs AND executed burns (deviation model), so what-if burns
        # on the clone start from the trajectory the live propagator flies.
        return self.propagator.clone()

    def _assess_future_conjunctions(self, propagator: SGP4Propagator, target_id: int, start_time: datetime) -> dict:
        nearest = None

        for other_id in propagator.norad_ids:
            if other_id == target_id:
                continue

            event = find_tca(
                propagator,
                target_id,
                other_id,
                start=start_time,
                hours_ahead=24.0,
                steps=120,
            )
            if event is None:
                continue

            try:
                event_time = datetime.fromisoformat(event.tca_utc)
            except Exception:
                event_time = start_time

            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)

            event_minutes = max(0.0, (event_time - start_time).total_seconds() / 60.0)
            if nearest is None or event.miss_distance_km < nearest["min_miss_distance_km"]:
                nearest = {
                    "other_id": other_id,
                    "other_name": event.sat2_name if event.sat1_id == target_id else event.sat1_name,
                    "min_miss_distance_km": float(event.miss_distance_km),
                    "closest_tca_minutes": float(event_minutes),
                    "tca_utc": event_time.isoformat(),
                    "severity": event.severity,
                }

        return nearest or {
            "other_id": None,
            "other_name": None,
            "min_miss_distance_km": None,
            "closest_tca_minutes": None,
            "tca_utc": None,
            "severity": None,
        }

    def load_scenario(self, scenario_name: str) -> dict:
        """
        Load a scenario JSON and inject its satellites into the propagator.
        Returns the scenario metadata.
        """
        path = os.path.join(SCENARIOS_DIR, f"{scenario_name}.json")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Scenario not found: {scenario_name}")

        with open(path, "r") as f:
            scenario = json.load(f)

        if scenario.get("type") == "computed_crossing":
            if self._scenario_sats:
                self.clear_scenario()
            return self._load_computed_crossing(scenario, scenario_name)

        # Inject scenario satellites on top of the existing catalog (do NOT wipe)
        tle_list = scenario.get("satellites", [])
        if tle_list:
            from sgp4.api import Satrec, WGS72
            with self.propagator._lock:
                for tle in tle_list:
                    try:
                        sat = Satrec.twoline2rv(tle["line1"], tle["line2"], WGS72)
                        nid = tle["norad_id"]
                        if nid in self.propagator._satellites and nid not in self._scenario_sats:
                            self._scenario_backup[nid] = self.propagator._satellites[nid]
                        self.propagator._satellites[nid] = (sat, tle["name"])
                        self._scenario_sats.append(nid)
                    except Exception as e:
                        logger.warning(f"Failed to load scenario sat: {e}")

            # Honor spatial offsets so scenario descriptions match reality
            inject_offset = scenario.get("inject_offset_km") or {}
            if inject_offset:
                bravo = next((t for t in tle_list if "bravo" in (t.get("name") or "").lower()), None)
                if bravo is not None:
                    try:
                        state = self.propagator.propagate_one(bravo["norad_id"])
                        if state is not None and state.error_code == 0:
                            self.propagator.apply_state_override(
                                bravo["norad_id"],
                                state.x + float(inject_offset.get("sat_bravo_dx", 0.0)),
                                state.y + float(inject_offset.get("sat_bravo_dy", 0.0)),
                                state.z + float(inject_offset.get("sat_bravo_dz", 0.0)),
                                state.vx, state.vy, state.vz,
                            )
                    except Exception as e:
                        logger.warning(f"Failed to apply scenario offset: {e}")

        self.active_scenario = scenario
        logger.info(f"Loaded scenario: {scenario.get('name', scenario_name)}")

        return {
            "name": scenario.get("name", scenario_name),
            "description": scenario.get("description", ""),
            "tca_minutes": scenario.get("tca_minutes_from_trigger", 0),
            "satellites_injected": len(tle_list),
        }

    def clear_scenario(self):
        """Remove scenario satellites from the propagator."""
        with self.propagator._lock:
            for nid in self._scenario_sats:
                self.propagator._satellites.pop(nid, None)
            for nid, entry in self._scenario_backup.items():
                self.propagator._satellites[nid] = entry
        try:
            from app.core.debris_model import unregister_object_meta
            for nid in self._scenario_sats:
                unregister_object_meta(nid)
        except Exception:
            pass
        self._scenario_sats.clear()
        self._scenario_backup.clear()
        self.active_scenario = None
        self.scenario_pair = None
        logger.info("Scenario cleared")

    def apply_maneuver(
        self,
        norad_id: int,
        dvx: float,
        dvy: float,
        dvz: float,
        frame: str = "RSW",
        session_id: str = "DEMO_SESSION",
    ) -> dict:
        """
        Apply a delta-V impulse maneuver.
        Returns validation results.

        The session id is forwarded to the pre-flight authority gate; it used to
        be dropped here, which silently downgraded every maneuver to
        DEMO_SESSION and made the agency authorization model inert.
        """
        from app.core import sim_clock

        # One burn epoch (simulation clock) for the dry run and the real burn.
        epoch = sim_clock.simulation_now()
        preflight = self.preflight_check(
            norad_id, dvx, dvy, dvz, frame=frame, session_id=session_id, epoch=epoch
        )

        if not all(preflight["gates"].values()):
            return {
                "status": "BLOCKED",
                "preflight": preflight,
            }

        result = self.propagator.apply_delta_v(norad_id, dvx, dvy, dvz, frame=frame, epoch=epoch)
        if result.get("status") != "success":
            return {"status": "ERROR", "preflight": preflight, **result}

        return {**result, "preflight": preflight}

    def preflight_check(
        self,
        norad_id: int,
        dvx: float,
        dvy: float,
        dvz: float,
        frame: str = "RSW",
        session_id: str = "DEMO_SESSION",
        epoch: datetime | None = None,
    ) -> dict:
        """
        6-gate pre-flight validation based on cloned orbit simulation.
        Returns dict with gate statuses and the nearest future encounter.

        Gate: agency_auth
            Checked via AgencyAuthorityManager.check_authority.
            DEMO_SESSION always passes. Named agency sessions enforce ownership.
        """
        import numpy as np

        dv_magnitude = float(np.sqrt(dvx**2 + dvy**2 + dvz**2))
        from app.core import sim_clock

        now = epoch or sim_clock.simulation_now()
        current_state = self.propagator.propagate_one(norad_id, now)
        target_name = current_state.name if current_state is not None else f"NORAD-{norad_id}"
        agency = self._infer_agency(target_name)

        # ── Real agency authority check (replaces hardcoded True) ────────────
        if _AUTHORITY_AVAILABLE and authority_manager is not None:
            try:
                agency_auth = authority_manager.check_authority(
                    session_id, norad_id, target_name
                )
            except Exception as auth_err:
                logger.warning("Authority check failed: %s — defaulting to DEMO_SESSION allow", auth_err)
                agency_auth = (session_id == "DEMO_SESSION")
        else:
            # Fallback: DEMO_SESSION is always allowed
            agency_auth = (session_id == "DEMO_SESSION")

        cloned_propagator = self._clone_propagator()
        burn_result = cloned_propagator.apply_delta_v(norad_id, dvx, dvy, dvz, frame=frame, epoch=now)

        future_risk = None
        gates = {
            "trajectory_clear": False,
            "fuel_budget": dv_magnitude <= 50.0,
            "agency_auth": agency_auth,
            "tca_window": False,
            "physical_limits": False,
            "sim_dryrun": False,
        }

        if burn_result.get("status") == "success":
            future_risk = self._assess_future_conjunctions(cloned_propagator, norad_id, now)
            min_miss = future_risk.get("min_miss_distance_km")
            tca_minutes = future_risk.get("closest_tca_minutes")

            gates.update({
                "trajectory_clear": min_miss is None or min_miss > WARNING_DISTANCE_KM,
                "tca_window": tca_minutes is None or tca_minutes > 60.0,
                "physical_limits": burn_result.get("new_perigee_km", 0.0) >= 100.0 and dv_magnitude <= 100.0,
                "sim_dryrun": True,
            })

        gates = {name: bool(passed) for name, passed in gates.items()}
        all_clear = all(gates.values())

        # The burn and risk results carry numpy scalars/arrays, which the
        # JSON encoder rejects (500 on /api/preflight); convert to plain types.
        return _to_native({
            "norad_id": norad_id,
            "satellite_name": target_name,
            "agency": agency,
            "frame": frame,
            "dv_magnitude_ms": round(dv_magnitude, 3),
            "gates": gates,
            "all_clear": all_clear,
            "trajectory": future_risk,
            "burn": burn_result,
        })

    def list_scenarios(self) -> list[str]:
        """List available scenario files."""
        scenarios = []
        for f in os.listdir(SCENARIOS_DIR):
            if f.endswith(".json"):
                scenarios.append(f.replace(".json", ""))
        return scenarios
