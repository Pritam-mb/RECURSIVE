import sys
sys.path.insert(0, '.')

results = []

# Test 1: analytics.py — Foster Pc + covariance ellipse
try:
    from app.core.analytics import compute_collision_probability, nasa_fragment_count
    state_p = dict(x=6778.0, y=0.0, z=0.0, vx=0.0, vy=7.7, vz=0.0)
    state_q = dict(x=6778.0, y=0.1, z=0.0, vx=0.0, vy=-7.7, vz=0.0)
    r = compute_collision_probability(state_p, state_q, tle_age_hours=0.5)
    ce = r['covariance_ellipse']
    assert all(k in ce for k in ['a','b','angle','affection_rate','predicted_fragments'])
    n = nasa_fragment_count(500, 500, 10.0)
    assert n > 0
    results.append(('analytics.py', True, 'Pc=%s frags=%d' % (r['p_collision'], n)))
except Exception as e:
    results.append(('analytics.py', False, str(e)))

# Test 2: satellite_state_tracker.py
try:
    from app.core.satellite_state_tracker import satellite_tracker
    t1 = satellite_tracker.get_telemetry(99001, 'TESTSAT')
    fuel_before = t1['fuel_remaining_pct']
    satellite_tracker.record_maneuver(99001, 20.0, 'RSW', 'TESTSAT')
    t2 = satellite_tracker.get_telemetry(99001, 'TESTSAT')
    assert t2['fuel_remaining_pct'] < fuel_before
    assert len(satellite_tracker.get_maneuver_history(99001)) == 1
    results.append(('satellite_state_tracker.py', True, 'fuel depleted %.1f->%.1f' % (fuel_before, t2['fuel_remaining_pct'])))
except Exception as e:
    results.append(('satellite_state_tracker.py', False, str(e)))

# Test 3: agency_authority.py
try:
    from app.core.agency_authority import authority_manager
    assert authority_manager.check_authority('DEMO_SESSION', 0, 'STARLINK-1') == True
    sid = authority_manager.create_session('ESA')
    assert authority_manager.check_authority(sid, 0, 'SENTINEL-1A') == True
    assert authority_manager.check_authority(sid, 0, 'STARLINK-1234') == False
    results.append(('agency_authority.py', True, 'DEMO_SESSION=True, ESA enforced'))
except Exception as e:
    results.append(('agency_authority.py', False, str(e)))

# Test 4: conjunction.py — Foster wired, CPI calibrated
try:
    from app.core.conjunction import _ANALYTICS_AVAILABLE, compute_cpi_score, _gaussian_fallback
    assert _ANALYTICS_AVAILABLE == True
    cpi_low = compute_cpi_score(1e-6, 50.0)
    cpi_high = compute_cpi_score(1e-3, 1.0)
    assert cpi_high > cpi_low
    fb = _gaussian_fallback(0.01)
    assert 0 <= fb <= 1
    results.append(('conjunction.py', True, 'ANALYTICS=True CPI_low=%.1f CPI_high=%.1f' % (cpi_low, cpi_high)))
except Exception as e:
    results.append(('conjunction.py', False, str(e)))

# Test 5: debris_model.py — NASA EVOLVE fragments
try:
    from app.core.debris_model import debris_model
    cloud = debris_model.simulate_collision(
        [6778.0, 0.0, 0.0], [0.0, 7.7, 0.0],
        500, 500, 10.0, max_fragments_simulated=20, event_id='t_final'
    )
    assert cloud.is_catastrophic == True
    assert len(cloud.fragments) > 0  # some may fail generation — that's OK
    frags = debris_model.propagate_fragments('t_final', 3600)
    assert len(frags) > 0
    results.append(('debris_model.py', True, '%d/%d fragments propagated, catastrophic=%s' % (len(frags), len(cloud.fragments), cloud.is_catastrophic)))
except Exception as e:
    results.append(('debris_model.py', False, str(e)))

# Test 6: ws_handler.py — covariance ellipse in broadcast
try:
    from app.api.ws_handler import _build_covariance_index
    alerts = [dict(sat1=dict(id=25544), sat2=dict(id=99), covariance_ellipse=dict(a=3000, b=150, angle=0))]
    idx = _build_covariance_index(alerts)
    assert '25544' in idx and '99' in idx
    results.append(('ws_handler.py', True, 'covariance index built'))
except Exception as e:
    results.append(('ws_handler.py', False, str(e)))

# Test 7: train_risk_surrogate.py — importable (Pc surrogate on Foster labels)
try:
    from app.ml.train_risk_surrogate import train, generate_encounters
    results.append(('train_risk_surrogate.py', True, 'importable'))
except Exception as e:
    results.append(('train_risk_surrogate.py', False, str(e)))

# Test 8: lstm_predictor.py — has train_on_buffer_data
try:
    from app.ml.lstm_predictor import LSTMPredictor
    p = LSTMPredictor()
    assert hasattr(p, 'train_on_buffer_data')
    assert hasattr(p, '_background_train')
    results.append(('lstm_predictor.py', True, 'train_on_buffer_data present'))
except Exception as e:
    results.append(('lstm_predictor.py', False, str(e)))

print()
print('=== FINAL VERIFICATION ===')
all_ok = True
for name, ok, detail in results:
    tag = 'PASS' if ok else 'FAIL'
    print('  [%s] %s: %s' % (tag, name, detail))
    if not ok:
        all_ok = False

print()
if all_ok:
    print('ALL 8 CHECKS PASSED - Orbital Sentinel physics transformation complete')
else:
    print('SOME CHECKS FAILED')

sys.exit(0 if all_ok else 1)
