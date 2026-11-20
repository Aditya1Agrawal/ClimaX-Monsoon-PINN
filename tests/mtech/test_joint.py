import numpy as np
import pytest

torch = pytest.importorskip("torch")

from climax_hazards import forecast_eval as fe  # noqa: E402
from climax_hazards import joint as J  # noqa: E402


def test_uncertainty_weighting_reaches_inverse_loss():
    uw = J.UncertaintyWeighting(["a", "b"])
    opt = torch.optim.SGD(uw.parameters(), lr=0.1)
    losses = {"a": torch.tensor(0.05), "b": torch.tensor(4.0)}
    for _ in range(2000):
        total, _ = uw(losses)
        opt.zero_grad()
        total.backward()
        opt.step()
    _, w = uw(losses)
    # optimum of exp(-s) L + s is exp(-s) = 1 / L, so every task contributes ~1
    assert abs(w["a"] - 20.0) < 1.0 and abs(w["b"] - 0.25) < 0.02


def test_fixed_weighting():
    fw = J.FixedWeighting({"weather": 1.0, "physics": 0.5})
    total, w = fw({"weather": torch.tensor(2.0), "physics": torch.tensor(4.0)})
    assert float(total) == 4.0 and w == {"weather": 1.0, "physics": 0.5}


def test_hypsometric_residual_matches_eval_metric():
    v = ["geopotential_500", "geopotential_850", "temperature_500", "temperature_850", "2m_temperature"]
    mean = np.array([57000.0, 14600.0, 265.0, 290.0, 290.0], dtype=np.float32)[None, :, None, None]
    std = np.array([700.0, 400.0, 5.0, 5.0, 15.0], dtype=np.float32)[None, :, None, None]
    rng = np.random.default_rng(0)
    phys = rng.normal(size=(3, 5, 2, 2)).astype(np.float32) * std + mean
    hr = J.HypsometricResidual(v, mean, std, scale_m=1.0)
    r = hr.residual(torch.from_numpy((phys - mean) / std)).numpy()[:, 0]
    ref = fe.hypsometric_residuals(phys, v, [(500, 850)])[:, 0]
    assert np.allclose(r, ref, atol=1e-2)
    x = torch.from_numpy((phys - mean) / std)
    assert float(hr(x, x)) == 0.0


def test_hazard_losses_ignore_nan_and_empty():
    logits = torch.zeros(4, 2, 2, 2, requires_grad=True)
    y = torch.full((4, 2, 2, 2), float("nan"))
    y[:, 0, 0, 0] = 1.0
    out = J.hazard_losses(logits, y, torch.ones(2))
    assert torch.isfinite(out[0]) and float(out[1]) == 0.0
    (out[0] + out[1]).backward()
    assert torch.isfinite(logits.grad).all()


def test_split_data_pairs_and_targets(tmp_path):
    from climax_hazards import labels as L

    variables = ["2m_temperature", "geopotential_500"]
    (tmp_path / "val").mkdir()
    n = 100
    for k in range(2):
        np.savez(tmp_path / "val" / f"2011_{k}.npz", **{v: np.full((n, 1, 2, 2), float(k), np.float32) for v in variables})
    mean = np.zeros((1, 2, 1, 1), np.float32)
    std = np.ones((1, 2, 1, 1), np.float32)
    dates = np.datetime64("2011-01-01") + np.arange(400).astype("timedelta64[D]")
    ls = L.LabelSet("h", {}, {})
    ls.splits = {"val": {"dates": dates, "label": np.arange(400, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32), "value": None}}
    # shard 1 starts at hour 1095, so hours 100..1094 are missing: pairs must respect the time gap
    d = J.SplitData(tmp_path, "val", variables, mean, std, lead=72, labelsets={"h": ls})
    th = d.times.astype("int64")
    assert np.all(th[d.dst] - th[d.src] == 72)
    assert len(d) == 2 * (n - 72)
    x, y, h = d.batch(np.array([0]))
    t_target_day = (d.times[d.src[0]] + np.timedelta64(72, "h")).astype("datetime64[D]")
    assert float(h[0, 0, 0, 0]) == float((t_target_day - dates[0]).astype(int))
    assert len(d.issue_hour_indices()) > 0


def test_joint_model_shapes_and_strict_load():
    pytest.importorskip("timm")
    from climax.arch import ClimaX

    vars_ = ["a", "b", "c"]
    net = ClimaX(vars_, img_size=[6, 6], patch_size=2, embed_dim=32, depth=1, decoder_depth=1, num_heads=4)
    m = J.JointModel(net, ["heat", "wind"])
    w, h = m(torch.randn(2, 3, 6, 6), torch.full((2,), 0.72), vars_)
    assert w.shape == (2, 3, 6, 6) and h.shape == (2, 2, 6, 6)
    sd = {"net." + k: v for k, v in net.state_dict().items()}
    net2 = ClimaX(vars_, img_size=[6, 6], patch_size=2, embed_dim=32, depth=1, decoder_depth=1, num_heads=4)
    J.load_climax_state(net2, sd)
    other = ClimaX(vars_ + ["d"], img_size=[6, 6], patch_size=2, embed_dim=32, depth=1, decoder_depth=1, num_heads=4)
    with pytest.raises(RuntimeError):
        J.load_climax_state(other, sd)


def test_uncertainty_weight_cap_holds():
    uw = J.UncertaintyWeighting(["weather", "heat"], max_weight={"heat": 2.0})
    opt = torch.optim.SGD(uw.parameters(), lr=0.1)
    losses = {"weather": torch.tensor(0.05), "heat": torch.tensor(0.01)}  # tiny "overfitting" heat loss
    for _ in range(2000):
        total, _ = uw(losses)
        opt.zero_grad()
        total.backward()
        opt.step()
    _, w = uw(losses)
    assert w["heat"] <= 2.0 + 1e-5          # capped instead of growing to ~100
    assert abs(w["weather"] - 20.0) < 1.0   # uncapped task still reaches 1/L


def test_gradnorm_balances_gradient_norms_and_keeps_sum():
    torch.manual_seed(0)
    shared = torch.nn.Linear(4, 4)
    head_a, head_b = torch.nn.Linear(4, 1), torch.nn.Linear(4, 1)
    gn = J.GradNormWeighting(["a", "b"], alpha=0.0)   # alpha 0: pure gradient-norm equalisation
    opt = torch.optim.Adam(gn.parameters(), lr=0.05)
    x = torch.randn(32, 4)
    for _ in range(300):
        z = shared(x)
        losses = {"a": (head_a(z) ** 2).mean() * 100.0, "b": (head_b(z) ** 2).mean()}  # a has ~100x larger gradients
        assert gn.update(losses, shared.weight, opt)
    w = gn.w.detach()
    assert abs(float(w.sum()) - 2.0) < 1e-4
    z = shared(x)
    la, lb = (head_a(z) ** 2).mean() * 100.0, (head_b(z) ** 2).mean()
    ga = torch.autograd.grad(la, shared.weight, retain_graph=True)[0].norm() * w[0]
    gb = torch.autograd.grad(lb, shared.weight)[0].norm() * w[1]
    assert w[0] < w[1]                                   # the large-gradient task is down-weighted
    assert float(abs(ga - gb) / (ga + gb)) < 0.25        # weighted gradient norms roughly equal


def test_gradnorm_skips_non_finite():
    gn = J.GradNormWeighting(["a"])
    p = torch.nn.Parameter(torch.ones(2))
    assert gn.update({"a": (p * float("nan")).sum()}, p, torch.optim.Adam(gn.parameters())) is False


LEVELS = (50, 250, 500, 600, 700, 850, 925)


def _column(n=4, seed=0):
    """Random but physically scaled (N, V, 2, 2) fields for all 7 levels, plus normalisation stats."""
    v = [f"geopotential_{p}" for p in LEVELS] + [f"temperature_{p}" for p in LEVELS]
    zm = {50: 2.0e5, 250: 1.0e5, 500: 5.7e4, 600: 4.3e4, 700: 3.0e4, 850: 1.46e4, 925: 7.5e3}
    tm = {50: 210.0, 250: 225.0, 500: 262.0, 600: 270.0, 700: 278.0, 850: 288.0, 925: 293.0}
    mean = np.array([zm[p] for p in LEVELS] + [tm[p] for p in LEVELS], np.float32)[None, :, None, None]
    std = np.array([500.0] * 7 + [4.0] * 7, np.float32)[None, :, None, None]
    phys = np.random.default_rng(seed).normal(size=(n, 14, 2, 2)).astype(np.float32) * std + mean
    return v, mean, std, phys


def test_multi_layer_residuals_match_eval_metric():
    v, mean, std, phys = _column()
    hr = J.HypsometricResidual(v, mean, std, layers=J.ADJACENT_LAYERS, scale_m=1.0)
    r = hr.residual(torch.from_numpy((phys - mean) / std)).numpy()
    assert r.shape == (4, 6, 2, 2)
    ref = fe.hypsometric_residuals(phys, v, J.ADJACENT_LAYERS)
    assert np.allclose(r, ref, atol=2e-2)


def test_adjacent_residuals_telescope_to_trapezoid_thick_layer():
    """Sum of 850-700, 700-600, 600-500 residuals = 500-850 thickness residual with a log-p trapezoid mean T."""
    v, mean, std, phys = _column()
    hr = J.HypsometricResidual(v, mean, std, layers=((700, 850), (600, 700), (500, 600)), scale_m=1.0)
    s = hr.residual(torch.from_numpy((phys - mean) / std)).double().sum(1).numpy()
    ix = {name: i for i, name in enumerate(v)}
    p = [850, 700, 600, 500]
    t_int = sum(0.5 * (phys[:, ix[f"temperature_{a}"]] + phys[:, ix[f"temperature_{b}"]]) * np.log(a / b) for a, b in zip(p[:-1], p[1:]))
    thick = (phys[:, ix["geopotential_500"]] - phys[:, ix["geopotential_850"]]) / fe.G - (fe.R_D / fe.G) * t_int
    assert np.allclose(s, thick, atol=5e-2)


def test_multi_layer_loss_scaling_and_zero():
    v, mean, std, phys = _column()
    x = torch.from_numpy((phys - mean) / std)
    y = x.clone()
    y[:, v.index("geopotential_500")] += 0.1          # perturb Z500: only the 600-500 and 500-250 layers change
    hr = J.HypsometricResidual(v, mean, std, layers=J.ADJACENT_LAYERS, scale_m=1.0)
    assert float(hr(x, x)) == 0.0
    mse = hr.layer_mse(x, y).numpy()
    changed = [hr.layers.index((500, 600)), hr.layers.index((250, 500))]
    assert np.all(mse[changed] > 0) and np.allclose(np.delete(mse, changed), 0.0, atol=1e-6)
    scales = np.sqrt(np.maximum(mse, 1e-12))
    hr.set_scales(np.where(mse > 0, scales, 1.0))
    assert abs(float(hr(x, y)) - len(changed) / 6) < 1e-4  # each perturbed layer contributes exactly 1, averaged over 6
    with pytest.raises(ValueError):
        hr.set_scales([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])


def test_parse_layers():
    assert J.parse_layers("single") == ((500, 850),)
    assert J.parse_layers("adjacent") == J.ADJACENT_LAYERS
    assert J.parse_layers("850-500,925-850") == ((500, 850), (850, 925))
    with pytest.raises(ValueError):
        J.parse_layers("500-500")
