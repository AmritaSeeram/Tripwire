import numpy as np
from tripwire.stats import kupiec_pof, christoffersen_ind, traffic_light
from tripwire.models import forecasts
from tripwire.data import simulate_garch

def test_kupiec_accepts_expected_rate_rejects_triple():
    ok = np.zeros(1000, int); ok[::100] = 1                      # exactly 1% breaches
    assert kupiec_pof(ok, 0.01)[1] > 0.9
    bad = np.zeros(1000, int); bad[::33] = 1                     # about 3%
    assert kupiec_pof(bad, 0.01)[1] < 0.001

def test_independence_detects_clustering():
    rng = np.random.default_rng(1); iid = (rng.uniform(size=3000) < 0.02).astype(int)
    clustered = np.zeros(3000, int)
    for s in range(0, 3000, 150): clustered[s:s + 3] = 1         # same 2% rate, but in bursts
    assert christoffersen_ind(iid)[1] > 0.05 and christoffersen_ind(clustered)[1] < 0.01

def test_traffic_light_zones():
    assert [traffic_light([1] * k + [0] * (250 - k)) for k in (4, 5, 10)] == ["green", "yellow", "red"]

def test_no_lookahead_future_returns_cannot_change_past_forecasts():
    r = simulate_garch(1200, seed=3); r2 = r.copy(); r2[900:] *= 5
    a, b = forecasts(r), forecasts(r2)
    for m in a: assert np.allclose(a[m][:900], b[m][:900], equal_nan=True)

def test_normal_model_is_calibrated_on_truly_normal_iid_data():
    r = np.random.default_rng(0).normal(0, 0.01, 20000); v = forecasts(r)["Normal"]; ok = ~np.isnan(v)
    assert 0.006 < (r[ok] < v[ok]).mean() < 0.014
