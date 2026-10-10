import numpy as np

from climax_hazards import labels as L

nan = np.nan


def test_spell_labels_runs_gaps_and_boundaries():
    e = np.array([1, 1, 0, 1, 1, 1, 0, 1, nan, 0, 1, 1], dtype=np.float32)[:, None]
    out = L.spell_labels(e, 3)[:, 0]
    # 0-1: short run touching the left boundary -> undetermined
    assert np.isnan(out[:2]).all()
    assert out[2] == 0
    assert (out[3:6] == 1).all()          # complete 3-day run
    assert out[6] == 0
    assert np.isnan(out[7])               # short run next to a missing day
    assert np.isnan(out[8])               # missing stays missing
    assert np.isnan(out[10:]).all()       # short run cut by the right boundary


def test_spell_labels_closed_short_run_is_zero():
    e = np.array([0, 1, 1, 0], dtype=np.float32)[:, None]
    assert (L.spell_labels(e, 3)[:, 0] == 0).all()
    assert (L.spell_labels(e, 1)[:, 0] == e[:, 0]).all()


def test_daily_aggregate_marks_incomplete_days_missing():
    times = np.datetime64("2016-12-29T00", "h") + np.arange(72).astype("timedelta64[h]")
    x = np.arange(72, dtype=np.float32)[:, None, None]
    x[50] = nan
    dates, agg = L.daily_aggregate(times, x, "max")
    assert str(dates[0]) == "2016-12-29"
    assert agg[0, 0, 0] == 23 and np.isnan(agg[2, 0, 0]) and agg[1, 0, 0] == 47


def test_align_to_forecast():
    dates = np.datetime64("2011-01-01") + np.arange(5).astype("timedelta64[D]")
    label = np.arange(5, dtype=np.float32)
    t = np.array(["2011-01-01T00", "2011-01-01T23", "2011-01-03T05"], dtype="datetime64[h]")
    out = L.align_to_forecast(t, 72, dates, label)
    assert out[0] == 3 and out[1] == 3 and np.isnan(out[2])  # Jan 4 -> idx 3; Jan 6 outside split


def test_calendar_percentile_uses_window():
    dates = np.datetime64("2001-01-01") + np.arange(730).astype("timedelta64[D]")
    vals = np.tile(np.arange(365, dtype=np.float32), 2)[:, None]
    thr = L.calendar_percentile(dates, vals, 100, window=2)
    assert thr[10, 0] == 12 and thr[100, 0] == 102


def test_prevalence_ignores_nan():
    p = L.prevalence(np.array([1, 0, nan, 0], dtype=np.float32))
    assert p == {"n_valid": 3, "n_event": 1, "n_undetermined": 1, "prevalence": 1 / 3}


def test_season_filter():
    dates = np.array(["2001-02-28", "2001-03-01", "2001-06-30", "2001-07-01"], dtype="datetime64[D]")
    out = L.season_filter(dates, np.ones((4, 1), dtype=np.float32), (3, 4, 5, 6))[:, 0]
    assert np.isnan(out[0]) and out[1] == 1 and out[2] == 1 and np.isnan(out[3])
    assert (L.season_filter(dates, np.ones((4, 1)), None) == 1).all()
