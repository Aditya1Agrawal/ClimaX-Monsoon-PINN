"""Lightweight CPU tests for climax_hazards.inspect_utils (synthetic data only)."""

import numpy as np
import pytest

from climax_hazards import inspect_utils as iu

VARS = ["2m_temperature", "geopotential_500"]


def _make_dataset(root, width=5, years=None):
    years = years or {"train": [2000], "val": [2001], "test": [2002]}
    rng = np.random.default_rng(0)
    np.save(root / "lat.npy", np.array([8.4375, 14.0625, 19.6875]))
    np.save(root / "lon.npy", 73.125 + 5.625 * np.arange(width))
    np.savez(root / "normalize_mean.npz", **{v: np.array([1.0]) for v in VARS})
    np.savez(root / "normalize_std.npz", **{v: np.array([2.0]) for v in VARS})
    for split, ys in years.items():
        (root / split).mkdir()
        for y in ys:
            for k in range(2):
                np.savez(root / split / f"{y}_{k}.npz", **{v: rng.normal(size=(4, 1, 3, width)).astype("f4") for v in VARS})
        np.savez(root / split / "climatology.npz", **{v: np.zeros((1, 3, width)) for v in VARS})
    return root


def test_discovery_and_summary(tmp_path):
    (tmp_path / "data").mkdir()
    ds = _make_dataset(tmp_path / "data")
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "last.ckpt").write_bytes(b"x")
    assert iu.find_dataset_dirs(tmp_path) == [ds]
    assert [p.name for p in iu.find_checkpoints(tmp_path)] == ["last.ckpt"]
    s = iu.summarize_dataset(ds)
    assert s["issues"] == []
    assert s["splits"]["train"]["shapes"]["2m_temperature"] == [(4, 1, 3, 5)]
    assert s["padding"]["looks_padded"] is False
    assert s["lat_ascending"]


def test_padding_detection_and_guard(tmp_path):
    ds = _make_dataset(tmp_path)
    for f in iu.shard_files(ds, "train"):
        d = {k: iu.pad_lon_edge(v) for k, v in np.load(f).items()}
        np.savez(f, **d)
    np.save(ds / "lon.npy", 73.125 + 5.625 * np.arange(6))
    assert iu.detect_lon_padding(ds)["looks_padded"] is True
    with pytest.raises(ValueError):
        iu.pad_lon_edge(np.zeros((1, 3, 6)))  # never pad twice


def test_issue_detection(tmp_path):
    ds = _make_dataset(tmp_path, years={"train": [2005], "val": [2001], "test": [2002]})
    f = iu.shard_files(ds, "val")[0]
    d = dict(np.load(f))
    d["2m_temperature"][0, 0, 0, 0] = np.nan
    np.savez(f, **d)
    issues = " | ".join(iu.summarize_dataset(ds)["issues"])
    assert "non-finite" in issues and "chronologically" in issues


def test_shard_timestamps():
    start, end = iu.shard_time_range(2016, 7, 1095)
    # verified against raw WeatherBench 2016 file: leap year loses Dec 31
    assert str(start) == "2016-11-15T09:00" and str(end) == "2016-12-30T23:00"
    assert iu.shard_timestamps(1979, 0, 3)[-1] == np.datetime64("1979-01-01T02:00")


def test_manifest_compare(tmp_path):
    ds = _make_dataset(tmp_path)
    m = iu.md5_manifest(ds)
    assert iu.compare_manifests(m, m)["identical"]
    m2 = dict(m)
    m2.pop("lat.npy")
    assert iu.compare_manifests(m2, m)["missing_in_actual"] == ["lat.npy"]


def test_lat_weighted_rmse():
    lat = np.array([0.0, 60.0])
    pred = np.zeros((1, 1, 2, 1))
    target = np.ones((1, 1, 2, 1))
    assert np.allclose(iu.lat_weighted_rmse_np(pred, target, lat), 1.0)


def test_checkpoint_inference(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from climax.arch import ClimaX

    vars_ = ["a", "b", "c"]
    net = ClimaX(vars_, img_size=[6, 6], patch_size=2, embed_dim=64, depth=2, decoder_depth=2, num_heads=4)
    sd = {"net." + k: v for k, v in net.state_dict().items()}
    sd["hypsometric_loss.g"] = torch.tensor(9.8)
    path = tmp_path / "fake.ckpt"
    torch.save({"epoch": 3, "global_step": 10, "state_dict": sd, "hyper_parameters": {"physics_lambda": 1e-3}}, path)
    info = iu.summarize_checkpoint(path)
    arch = info["inferred_arch"]
    assert (arch["embed_dim"], arch["patch_size"], arch["depth"], arch["decoder_depth"]) == (64, 2, 2, 2)
    assert arch["num_patches"] == 9 and arch["head_out_vars"] == 3
    assert info["non_net_keys"] == ["hypsometric_loss.g"]
    cmp = iu.compare_state_dict(net, sd)
    assert not cmp["missing_in_checkpoint"] and not cmp["unexpected_in_checkpoint"] and not cmp["shape_mismatch"]
    other = ClimaX(vars_ + ["d"], img_size=[6, 6], patch_size=2, embed_dim=64, depth=2, decoder_depth=2, num_heads=4)
    assert "var_embed" in iu.compare_state_dict(other, sd)["shape_mismatch"]


def test_hypsometric_residual_metric():
    from climax_hazards import forecast_eval as fe

    ov = ["geopotential_500", "geopotential_850", "temperature_500", "temperature_850", "2m_temperature"]
    assert fe.physics_layers(ov) == [(500, 850)]
    t = 270.0
    dz = (fe.R_D / fe.G) * t * np.log(850 / 500)
    arr = np.zeros((1, 5, 2, 2))
    arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3] = dz * fe.G + 1000.0, 1000.0, t, t
    assert np.allclose(fe.hypsometric_residuals(arr, ov, [(500, 850)]), 0, atol=1e-6)
    arr[:, 0] += 10 * fe.G  # 10 m too thick
    assert np.allclose(fe.hypsometric_residuals(arr, ov, [(500, 850)]), 10)


def test_make_padded_copy_matches_internship_and_refuses_twice(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_dataset(src)
    dst = iu.make_padded_copy(src, tmp_path / "pad")
    assert np.allclose(np.load(dst / "lon.npy"), 73.125 + 5.625 * np.arange(6))
    a = np.load(iu.shard_files(dst, "train")[0])["2m_temperature"]
    assert a.shape[-1] == 6 and np.array_equal(a[..., 5], a[..., 4])
    assert np.load(dst / "train" / "climatology.npz")["2m_temperature"].shape == (1, 3, 6)
    assert np.load(dst / "normalize_mean.npz")["2m_temperature"].shape == (1,)
    with pytest.raises(ValueError):
        iu.make_padded_copy(dst, tmp_path / "pad2")
    with pytest.raises(FileExistsError):
        iu.make_padded_copy(src, dst)
