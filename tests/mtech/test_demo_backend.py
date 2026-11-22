"""Demo backend tests.

The integration tests run only where the real data and the notebook-07 outputs exist (they are
git-ignored), and check that live inference reproduces the saved Kaggle predictions.
"""

import os

import numpy as np
import pytest

pytest.importorskip("sklearn")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAVE_DATA = all(os.path.exists(os.path.join(REPO, p)) for p in (
    "regional_climax_dataset_6x6_corrected/lat.npy",
    "outputs/kaggle_v3/stageB/preds/targets.npz",
    "docs/mtech/results/analysis/calibration.csv"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="demo data / notebook-07 outputs not present")


@pytest.fixture(scope="module")
def backend():
    from climax_hazards.demo_backend import DemoBackend, DemoConfig

    return DemoBackend(DemoConfig(repo=REPO))


@needs_data
def test_saved_mode_forecast_and_tables(backend):
    fc = backend.forecast("2015-05-22", live=False)
    assert str(fc.target_day) == "2015-05-25" and not fc.live
    assert fc.prob.shape == fc.observed.shape == fc.alerts.shape == (len(backend.hazards), 6, 6)
    assert not fc.alerts[~np.isfinite(fc.observed)].any()          # never alert where the hazard is undefined
    v = backend.verification(fc)
    assert (v["hits"] + v["misses"] == v["observed event cells"]).all()
    assert (v["hits"] + v["false alarms"] == v["alerts"]).all()
    t = backend.alert_table(fc)
    assert len(t) == int(fc.alerts.sum()) and (t["probability"] >= t["threshold"]).all()


@needs_data
def test_calibration_matches_analysis_and_is_monotonic(backend):
    import pandas as pd

    from climax_hazards import hazard_model as hm

    ov = pd.read_csv(os.path.join(REPO, "docs/mtech/results/analysis/overall_test.csv")).set_index(["source", "hazard"])
    for k, h in enumerate(backend.hazards):
        ap = hm.average_precision(backend.saved["test"][:, k], backend.y["test"][:, k])
        assert abs(ap - ov.loc[("ClimaX (calibrated)", h), "pr_auc"]) < 1e-6
    p = np.linspace(0.01, 0.99, 50)[:, None, None, None] * np.ones((1, len(backend.hazards), 1, 1))
    c = backend.calibrate(p, "ClimaX")
    assert np.all(np.diff(c[:, :, 0, 0], axis=0) > 0)


@needs_data
def test_unknown_day_raises(backend):
    with pytest.raises(KeyError):
        backend.forecast("2012-01-01", live=False)


@needs_data
def test_event_window(backend):
    from climax_hazards.analysis import CASES

    ser = backend.event(CASES[0])
    assert (ser["target_day"] - ser["issue_day"] == np.timedelta64(3, "D")).all()
    assert (ser["observed"] == 1).sum() > 0


@needs_data
@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "outputs/kaggle_v3/stageA/A_multi_s0/best_encoder.ckpt")),
                    reason="checkpoints not present")
def test_live_inference_reproduces_saved_predictions(backend):
    pytest.importorskip("timm")
    fc = backend.forecast("2015-05-22", live=True)
    assert fc.live and fc.max_abs_diff_vs_saved < 1e-4
    assert fc.weather_pred is not None and fc.weather_true is not None
    pe = backend.physics_errors("2015-05-22")
    assert set(pe["model"]) == {"multi-layer physics", "no physics"} and len(pe) == 12


@needs_data
@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "outputs/kaggle_v3/stageA/A_multi_s0/best_encoder.ckpt")),
                    reason="checkpoints not present")
def test_realtime_path_equals_validated_path(backend):
    """forecast_from_fields (used for live GFS inputs) gives the same probabilities as the validated path on an ERA5 input."""
    pytest.importorskip("timm")
    d = np.datetime64("2015-05-22")
    x = backend.x_phys[backend._day_index[d]]
    out = backend.forecast_from_fields(x, d + np.timedelta64(3, "D"))
    fc = backend.forecast(d, live=False)
    defined = backend.defined_cells()
    assert np.allclose(out["prob"][defined], fc.prob[defined], atol=1e-4)
    assert all(out["applicable"].values())                      # May: heatwave season
    nov = backend.forecast_from_fields(x, np.datetime64("2015-11-25"))
    assert not nov["applicable"]["heatwave"] and not nov["alerts"][backend.hazards.index("heatwave")].any()
    with pytest.raises(ValueError):
        backend.forecast_from_fields(x * np.nan, d)
    sk = backend.realtime_skill(backend.x_phys[:10], ["geopotential_500"])
    assert sk.loc[0, "forecasts"] == 7 and sk.loc[0, "ClimaX"] < sk.loc[0, "persistence"]
