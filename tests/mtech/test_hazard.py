import numpy as np
import pytest

torch = pytest.importorskip("torch")

from climax_hazards import hazard_data as hd  # noqa: E402
from climax_hazards import hazard_model as hm  # noqa: E402
from climax_hazards import labels as L  # noqa: E402


def test_unpatchify_matches_climax_token_order():
    pytest.importorskip("timm")
    from climax.arch import ClimaX

    net = ClimaX(["a"], img_size=[6, 6], patch_size=2, embed_dim=32, depth=1, decoder_depth=1, num_heads=4)
    heads = hm.HazardHeads(["h"], 32, [6, 6], 2)
    tok = torch.randn(3, 9, 4)
    assert torch.equal(net.unpatchify(tok)[:, 0], heads.unpatchify(tok))


def test_heads_shapes_and_masked_loss_ignores_nan():
    heads = hm.HazardHeads(["a", "b"], 16, [6, 6], 2)
    logits = heads(torch.randn(4, 9, 16))
    assert logits.shape == (4, 2, 6, 6)
    y = torch.full((4, 2, 6, 6), float("nan"))
    y[:, 0, 0, 0] = 1.0
    y[:, 1, :, :5] = 0.0
    loss = hm.masked_bce(logits, y, torch.ones(2))
    assert torch.isfinite(loss)
    loss.backward()  # NaN cells must not poison gradients
    assert all(torch.isfinite(p.grad).all() for p in heads.parameters() if p.grad is not None)


def test_scores_and_threshold():
    y = np.array([0, 0, 1, 1, np.nan], dtype=np.float32)
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.9], dtype=np.float32)
    s = hm.scores(p, y, 0.3)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (2, 1, 0, 1) and s["n"] == 4
    assert 0.3 < hm.best_f1_threshold(p, y) <= 0.8


def test_pos_weights_ignore_nan():
    y = np.array([[1, 0, 0, 0, np.nan]], dtype=np.float32).T[:, :, None, None]
    assert np.isclose(hm.pos_weights(y)[0], 3.0)


def _wind_labelset():
    dates = np.datetime64("2011-01-01") + np.arange(10).astype("timedelta64[D]")
    value = np.zeros((10, 1, 1), dtype=np.float32)
    value[4] = 20.0  # only 5 Jan exceeds
    ls = L.LabelSet("extreme_wind", {}, {"wind_threshold_ms_by_month": np.full((12, 1, 1), 10.0, np.float32)})
    lab = (value > 10).astype(np.float32)
    ls.splits = {"val": {"dates": dates, "value": value, "label": lab}, "train": {"dates": dates, "value": value, "label": lab}}
    return {"extreme_wind": ls}


def test_persistence_uses_only_past_days():
    ls = _wind_labelset()
    times = np.array(["2011-01-05T00", "2011-01-06T00"], dtype="datetime64[h]")
    p = hd.persistence_probs(ls, "val", times)[:, 0, 0, 0]
    # issued 00 UTC 5 Jan: 5 Jan is not yet observed -> 0; issued 6 Jan: sees 5 Jan -> 1
    assert list(p) == [0.0, 1.0]


def test_targets_and_climatology_alignment():
    ls = _wind_labelset()
    times = np.array(["2011-01-02T00", "2011-01-09T00"], dtype="datetime64[h]")
    y = hd.aligned_targets(ls, "val", times, 72)[:, 0, 0, 0]
    assert y[0] == 1.0 and np.isnan(y[1])  # 2 Jan + 72 h = 5 Jan; 12 Jan is outside the split
    c = hd.climatology_probs(ls, "val", times, 72, window=0)[:, 0, 0, 0]
    assert c[0] == 1.0 and c[1] == 0.0


def test_paired_block_bootstrap():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 2, 2)) < 0.1).astype(np.float32)
    good = y * 0.8 + rng.random(y.shape) * 0.2
    noise = rng.random(y.shape).astype(np.float32)
    same = hm.paired_block_bootstrap(good, good, y, n_boot=50)
    assert same["pr_auc_diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = hm.paired_block_bootstrap(good, noise, y, n_boot=200)
    assert better["ci_low"] > 0 and better["p_a_not_better"] == 0
