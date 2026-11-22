import numpy as np
import pytest

pytest.importorskip("sklearn")

from climax_hazards import analysis as A  # noqa: E402
from climax_hazards import hazard_model as hm  # noqa: E402


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0, 0.5, size=(n, 2, 2))
    y = (rng.uniform(size=p_true.shape) < p_true).astype(np.float32)
    y[:, 0, 0] = np.nan  # an undefined cell (e.g. sea)
    return p_true, y


def test_brier_and_skill():
    p, y = _data()
    clim = np.full_like(p, np.nanmean(y))
    assert A.brier(clim, y) > A.brier(p, y)
    assert A.brier_skill(p, clim, y) > 0
    assert abs(A.brier_skill(clim, clim, y)) < 1e-12


def test_reliability_perfect_vs_inflated():
    p, y = _data(20000)
    _, ece_good = A.reliability(p, y)
    tab, ece_bad = A.reliability(np.clip(p * 1.8, 0, 1), y)   # over-confident, like pos_weight-trained heads
    assert ece_good < 0.02 and ece_bad > 0.1
    assert tab["count"].sum() == np.isfinite(y).sum()


def test_platt_fixes_inflation_and_keeps_ranking():
    p, y = _data(20000)
    bad = np.clip(p * 1.8, 0, 0.999)
    cal = A.PlattCalibrator().fit(bad[:10000], y[:10000])
    out = cal.transform(bad[10000:])
    assert A.reliability(out, y[10000:])[1] < A.reliability(bad[10000:], y[10000:])[1]
    assert abs(hm.average_precision(out, y[10000:]) - hm.average_precision(bad[10000:], y[10000:])) < 1e-6
    with pytest.raises(ValueError):
        A.PlattCalibrator().fit(p, np.zeros_like(y))


def test_per_cell_ap_masks_rare_cells():
    p, y = _data()
    ap, npos = A.per_cell_ap(p, y, min_pos=10)
    assert np.isnan(ap[0, 0]) and npos[0, 0] == 0
    assert np.all(np.isfinite(ap[[0, 1, 1], [1, 0, 1]]))


def test_blend_weight_prefers_informative_source():
    p, y = _data()
    noise = np.random.default_rng(1).uniform(size=p.shape)
    w, curve = A.choose_blend_weight(p, noise, y)
    assert w >= 0.8 and len(curve) == 11


def test_event_series_and_cells():
    lat = np.array([8.4375, 14.0625]); lon = np.array([73.125, 78.75])
    assert A.nearest_cells(lat, lon, [(13.1, 80.3), (14.0, 79.0)]) == [(1, 1)]
    times = (np.datetime64("2015-12-01", "m") + np.arange(5) * np.timedelta64(1, "D")).astype("int64")
    days = A.target_days(times)
    assert str(days[0]) == "2015-12-04"
    y = np.zeros((5, 2, 2), np.float32); y[2, 1, 1] = 1; y[:, 0, 0] = np.nan
    p = np.linspace(0, 1, 20).reshape(5, 2, 2)
    df = A.event_series({"m": p}, y, days, [(1, 1), (0, 0)], "2015-12-05", "2015-12-07")
    assert list(df["observed"]) == [0.0, 1.0, 0.0]
    assert np.allclose(df["m"], p[1:4, 1, 1])
    with pytest.raises(ValueError):
        A.event_series({"m": p}, y, days, [(1, 1)], "2016-01-01", "2016-01-02")
