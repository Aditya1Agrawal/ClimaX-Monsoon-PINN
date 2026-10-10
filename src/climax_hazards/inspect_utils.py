"""Dataset and checkpoint inspection helpers (Phase 2).

Everything here is read-only with respect to its inputs. Nothing is padded,
copied or modified on disk.

Facts about the regional dataset that these helpers rely on (verified locally
against the raw WeatherBench 5.625 deg NetCDF files, see docs/mtech/AUDIT_REPORT.md):

* each ``<split>/<year>_<shard>.npz`` holds 1095 consecutive **hourly** UTC steps;
* shard ``k`` of year ``Y`` starts at ``Y-01-01T00:00 + k * 1095 h``;
* every year is truncated to 8760 h, so leap years lose 31 December;
* arrays are ``(time, 1, lat, lon)``; latitude is ascending (south -> north).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

DATASET_MARKERS = ("lat.npy", "lon.npy", "normalize_mean.npz", "normalize_std.npz")
SPLITS = ("train", "val", "test")
CHECKPOINT_EXTS = (".ckpt", ".pt", ".pth")
HOURS_PER_YEAR = 8760
SHARD_RE = re.compile(r"^(\d{4})_(\d+)\.npz$")


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------


def list_tree(root: str | os.PathLike, max_files_per_dir: int = 8) -> str:
    """Return a compact text listing of ``root`` (files truncated per directory)."""
    root = Path(root)
    lines = []
    if not root.exists():
        return f"{root} does not exist"
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        depth = Path(dirpath).relative_to(root).parts
        indent = "  " * len(depth)
        lines.append(f"{indent}{Path(dirpath).name}/  ({len(filenames)} files)")
        for f in sorted(filenames)[:max_files_per_dir]:
            size = os.path.getsize(os.path.join(dirpath, f))
            lines.append(f"{indent}  {f}  [{human_bytes(size)}]")
        if len(filenames) > max_files_per_dir:
            lines.append(f"{indent}  ... {len(filenames) - max_files_per_dir} more")
    return "\n".join(lines)


def find_dataset_dirs(search_root: str | os.PathLike) -> List[Path]:
    """Directories that look like a ClimaX npz dataset (marker files + train/)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        if all(m in filenames for m in DATASET_MARKERS) and "train" in dirnames:
            found.append(Path(dirpath))
    return sorted(found)


def find_checkpoints(search_root: str | os.PathLike, exts: Iterable[str] = CHECKPOINT_EXTS) -> List[Path]:
    exts = tuple(exts)
    found = []
    for dirpath, _, filenames in os.walk(search_root):
        for f in filenames:
            if f.endswith(exts):
                found.append(Path(dirpath) / f)
    return sorted(found)


def resolve_single(candidates: List[Path], override: Optional[str], what: str) -> Path:
    """Use ``override`` if given, otherwise require exactly one candidate."""
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"{what} override does not exist: {p}")
        return p
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No {what} found. Attach the Kaggle dataset or set the override path.")
    listing = "\n".join(f"  {c}" for c in candidates)
    raise RuntimeError(f"Several {what} candidates found; set the override explicitly:\n{listing}")


# ----------------------------------------------------------------------------
# Dataset inspection
# ----------------------------------------------------------------------------


def shard_files(data_dir: str | os.PathLike, split: str) -> List[Path]:
    d = Path(data_dir) / split
    return sorted(p for p in d.glob("*.npz") if SHARD_RE.match(p.name))


def shard_time_range(year: int, shard: int, n_steps: int, n_shards: int = 8):
    """First/last UTC timestamp of a shard (see module docstring for the convention)."""
    per_shard = HOURS_PER_YEAR // n_shards
    start = np.datetime64(f"{year:04d}-01-01T00:00") + np.timedelta64(shard * per_shard, "h")
    return start, start + np.timedelta64(n_steps - 1, "h")


def shard_timestamps(year: int, shard: int, n_steps: int, n_shards: int = 8) -> np.ndarray:
    start, _ = shard_time_range(year, shard, n_steps, n_shards)
    return start + np.arange(n_steps).astype("timedelta64[h]")


def detect_lon_padding(data_dir: str | os.PathLike, sample_files: int = 4) -> Dict:
    """Detect whether the internship's edge padding (duplicated last lon column) is present.

    Returns grid width, lon spacing and whether the last two columns are identical
    for every variable in a sample of shards.
    """
    lon = np.load(Path(data_dir) / "lon.npy")
    files = shard_files(data_dir, "train")[:sample_files]
    dup_all = True
    for f in files:
        with np.load(f) as d:
            for k in d.files:
                a = d[k]
                if a.shape[-1] < 2 or not np.array_equal(a[..., -1], a[..., -2]):
                    # constant fields (e.g. all-zero orography) are trivially duplicated; skip them
                    if a.shape[-1] >= 2 and np.ptp(a) == 0:
                        continue
                    dup_all = False
                    break
        if not dup_all:
            break
    spacing = np.diff(lon)
    return {
        "lon": lon.tolist(),
        "n_lon": int(lon.size),
        "lon_spacing_unique": np.unique(np.round(spacing, 6)).tolist(),
        "last_two_columns_identical_in_sample": bool(dup_all and files),
        "looks_padded": bool(dup_all and files),
    }


def summarize_dataset(data_dir: str | os.PathLike, check_values: bool = True, max_files_per_split: Optional[int] = None) -> Dict:
    """Shapes, variables, coords, temporal coverage, stats and value checks."""
    data_dir = Path(data_dir)
    lat = np.load(data_dir / "lat.npy")
    lon = np.load(data_dir / "lon.npy")
    with np.load(data_dir / "normalize_mean.npz") as m, np.load(data_dir / "normalize_std.npz") as s:
        mean = {k: m[k].tolist() for k in m.files}
        std = {k: s[k].tolist() for k in s.files}

    summary: Dict = {
        "data_dir": str(data_dir),
        "lat": lat.tolist(),
        "lon": lon.tolist(),
        "lat_ascending": bool(np.all(np.diff(lat) > 0)),
        "normalize_mean": mean,
        "normalize_std": std,
        "zero_std_variables": [k for k, v in std.items() if np.any(np.asarray(v) == 0)],
        "splits": {},
        "issues": [],
    }

    for split in SPLITS:
        files = shard_files(data_dir, split)
        if max_files_per_split:
            files = files[:max_files_per_split]
        info: Dict = {"n_shards": len(files), "variables": None, "shapes": {}, "nonfinite": {}, "years": []}
        years = set()
        n_steps_total = 0
        for f in files:
            y, k = map(int, SHARD_RE.match(f.name).groups())
            years.add(y)
            with np.load(f) as d:
                if info["variables"] is None:
                    info["variables"] = list(d.files)
                elif list(d.files) != info["variables"]:
                    summary["issues"].append(f"{f.name}: variable list differs from first shard")
                for var in d.files:
                    a = d[var]
                    info["shapes"].setdefault(var, set()).add(tuple(a.shape))
                    if check_values:
                        bad = int(a.size - np.isfinite(a).sum())
                        if bad:
                            info["nonfinite"][var] = info["nonfinite"].get(var, 0) + bad
                n_steps_total += d[d.files[0]].shape[0]
        if files:
            first_y, first_k = map(int, SHARD_RE.match(files[0].name).groups())
            last_y, last_k = map(int, SHARD_RE.match(files[-1].name).groups())
            with np.load(files[-1]) as d:
                n_last = d[d.files[0]].shape[0]
            info["first_time_utc"] = str(shard_time_range(first_y, first_k, 1)[0])
            info["last_time_utc"] = str(shard_time_range(last_y, last_k, n_last)[1])
        info["years"] = sorted(years)
        info["n_time_steps"] = n_steps_total
        info["shapes"] = {k: sorted(v) for k, v in info["shapes"].items()}
        clim = data_dir / split / "climatology.npz"
        if clim.exists():
            with np.load(clim) as c:
                info["climatology_shapes"] = {k: list(c[k].shape) for k in c.files}
        else:
            summary["issues"].append(f"{split}: climatology.npz missing")
        for var, shapes in info["shapes"].items():
            if len(shapes) > 1:
                summary["issues"].append(f"{split}/{var}: inconsistent shapes {shapes}")
            elif shapes[0][-2:] != (lat.size, lon.size):
                summary["issues"].append(f"{split}/{var}: spatial shape {shapes[0][-2:]} != (lat, lon) {(lat.size, lon.size)}")
        if info["nonfinite"]:
            summary["issues"].append(f"{split}: non-finite values {info['nonfinite']}")
        summary["splits"][split] = info

    tr, va, te = (summary["splits"][s]["years"] for s in SPLITS)
    if tr and va and te and not (max(tr) < min(va) and max(va) < min(te)):
        summary["issues"].append("splits are not chronologically disjoint")
    summary["padding"] = detect_lon_padding(data_dir)
    return summary


def pad_lon_edge(arr: np.ndarray, expected_width: int = 5) -> np.ndarray:
    """In-memory replica of the internship padding (``np.pad(..., mode='edge')`` on lon).

    Refuses to pad anything whose last axis is not ``expected_width`` so the
    padding can never be applied twice.
    """
    if arr.shape[-1] != expected_width:
        raise ValueError(f"refusing to pad: last axis is {arr.shape[-1]}, expected {expected_width}")
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (0, 1)
    return np.pad(arr, pad, mode="edge")


def make_padded_copy(src: str | os.PathLike, dst: str | os.PathLike, lon_step: float = 5.625) -> Path:
    """Reproduce the internship notebook's 6x6 copy (edge-pad lon by one column) into ``dst``.

    Same operations as the internship cell: every array with ndim >= 2 in the split files is
    edge-padded on the last axis, normalisation scalars are left alone, and ``lon`` gets
    ``lon[-1] + lon_step`` appended. Refuses to run on an already padded source or into an
    existing destination, so padding can never happen twice.
    """
    import shutil

    src, dst = Path(src), Path(dst)
    if detect_lon_padding(src)["looks_padded"]:
        raise ValueError(f"{src} is already padded")
    if dst.exists():
        raise FileExistsError(f"{dst} exists; delete it or reuse it explicitly")
    shutil.copytree(src, dst)
    width = int(np.load(src / "lon.npy").size)
    for split in SPLITS:
        for f in (dst / split).glob("*.npz"):
            with np.load(f) as d:
                data = {k: (pad_lon_edge(v, width) if v.ndim >= 2 else v) for k, v in d.items()}
            np.savez(f, **data)
    lon = np.load(dst / "lon.npy")
    np.save(dst / "lon.npy", np.append(lon, lon[-1] + lon_step))
    return dst


def md5_manifest(root: str | os.PathLike) -> Dict[str, Dict]:
    root = Path(root)
    out = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            p = Path(dirpath) / f
            h = hashlib.md5()
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            out[p.relative_to(root).as_posix()] = {"md5": h.hexdigest(), "bytes": p.stat().st_size}
    return dict(sorted(out.items()))


def compare_manifests(actual: Dict[str, Dict], reference: Dict[str, Dict]) -> Dict:
    common = set(actual) & set(reference)
    return {
        "n_actual": len(actual),
        "n_reference": len(reference),
        "missing_in_actual": sorted(set(reference) - set(actual)),
        "extra_in_actual": sorted(set(actual) - set(reference)),
        "md5_mismatch": sorted(k for k in common if actual[k]["md5"] != reference[k]["md5"]),
        "identical": not (set(actual) ^ set(reference)) and all(actual[k]["md5"] == reference[k]["md5"] for k in common),
    }


# ----------------------------------------------------------------------------
# Checkpoint inspection
# ----------------------------------------------------------------------------


def load_checkpoint(path: str | os.PathLike):
    """Load a (trusted) Lightning checkpoint on CPU.

    ``weights_only=False`` is required because Lightning stores hyper-parameters
    and callback state as Python objects; only use this on your own checkpoints.
    """
    import torch

    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=False, mmap=True, **kwargs)
    except TypeError:  # older torch without weights_only / mmap
        return torch.load(path, **kwargs)
    except RuntimeError:  # legacy (non-zip) format cannot be mmapped
        return torch.load(path, weights_only=False, **kwargs)


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_jsonable(v) for v in x]
        if hasattr(x, "item") and getattr(x, "ndim", 1) == 0:
            return x.item()
        return repr(x)


def infer_climax_config(state_dict: Dict, prefix: str = "net.") -> Dict:
    """Infer ClimaX architecture settings from tensor shapes (no guessing from filenames)."""
    sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not sd:
        return {"error": f"no keys with prefix {prefix!r}"}
    cfg: Dict = {}
    cfg["embed_dim"] = int(sd["var_embed"].shape[-1])
    cfg["n_default_vars"] = int(sd["var_embed"].shape[1])
    cfg["num_patches"] = int(sd["pos_embed"].shape[1])
    w = sd.get("token_embeds.0.proj.weight")
    if w is not None:
        cfg["patch_size"] = int(w.shape[-1])
        cfg["parallel_patch_embed"] = False
    elif "token_embeds.proj_weights" in sd:
        cfg["patch_size"] = int(sd["token_embeds.proj_weights"].shape[-1])
        cfg["parallel_patch_embed"] = True
    cfg["depth"] = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    head_linear = sorted(int(k.split(".")[1]) for k in sd if k.startswith("head.") and k.endswith(".weight"))
    cfg["decoder_depth"] = len(head_linear) - 1
    out_dim = int(sd[f"head.{head_linear[-1]}.weight"].shape[0])
    p = cfg.get("patch_size")
    cfg["head_out_dim"] = out_dim
    if p:
        cfg["head_out_vars"] = out_dim // (p * p)
    return cfg


def summarize_checkpoint(path: str | os.PathLike, ckpt=None) -> Dict:
    """Metadata, training state, callbacks and inferred architecture of a checkpoint."""
    path = Path(path)
    own = ckpt is None
    if own:
        ckpt = load_checkpoint(path)
    info: Dict = {"path": str(path), "size": human_bytes(path.stat().st_size), "top_level_keys": sorted(ckpt.keys())}
    for k in ("epoch", "global_step", "pytorch-lightning_version"):
        if k in ckpt:
            info[k] = _jsonable(ckpt[k])
    info["hyper_parameters"] = _jsonable(ckpt.get("hyper_parameters", {}))
    info["datamodule_hyper_parameters"] = _jsonable(ckpt.get("datamodule_hyper_parameters", {}))
    callbacks = ckpt.get("callbacks", {}) or {}
    info["callbacks"] = {str(k): _jsonable(v) for k, v in callbacks.items()}
    info["has_optimizer_states"] = bool(ckpt.get("optimizer_states"))
    sd = ckpt.get("state_dict", {})
    info["n_state_dict_keys"] = len(sd)
    info["n_params"] = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    info["non_net_keys"] = sorted(k for k in sd if not k.startswith("net."))
    info["inferred_arch"] = infer_climax_config(sd)
    if own:
        del ckpt
    return info


def compare_state_dict(model, state_dict: Dict, prefix: str = "net.") -> Dict:
    """Exact key/shape comparison before any load; nothing is dropped silently."""
    sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    ref = model.state_dict()
    return {
        "missing_in_checkpoint": sorted(set(ref) - set(sd)),
        "unexpected_in_checkpoint": sorted(set(sd) - set(ref)),
        "shape_mismatch": {k: (tuple(sd[k].shape), tuple(ref[k].shape)) for k in set(sd) & set(ref) if tuple(sd[k].shape) != tuple(ref[k].shape)},
    }


# ----------------------------------------------------------------------------
# Small numeric helpers
# ----------------------------------------------------------------------------


def lat_weighted_rmse_np(pred: np.ndarray, target: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Per-variable latitude-weighted RMSE for ``(N, V, H, W)`` arrays (same convention as ClimaX)."""
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[None, None, :, None]
    return np.sqrt((((pred - target) ** 2) * w).mean(axis=(2, 3))).mean(axis=0)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
