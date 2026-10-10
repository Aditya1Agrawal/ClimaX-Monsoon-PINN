"""Forecast evaluation with reference baselines (Step B).

All forecasts are scored on exactly the same (input time, target time) pairs as the
ClimaX ``Forecast`` dataset uses: inside each shard, ``x[t] -> x[t + lead]`` for
``t = 0 .. n_steps - lead - 1``. Metrics are in physical units.

Baselines
* persistence: ``y_hat = x[t]``
* climatology: train-period mean for the target's (day-of-year, UTC hour), smoothed
  with a circular +-``doy_window`` day window. Built from train years only.

Metrics
* lat-weighted RMSE (ClimaX convention: RMSE per sample, then mean over samples);
* lat-weighted ACC of anomalies w.r.t. the train climatology above (pooled over
  samples). Note this differs from ClimaX's built-in ACC, which subtracts the
  evaluated split's own time-mean.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from climax_hazards.inspect_utils import SHARD_RE, pad_lon_edge, shard_files, shard_timestamps


def _doy_hour(ts: np.ndarray):
    ts = ts.astype("datetime64[h]")
    day = ts.astype("datetime64[D]")
    doy = (day - day.astype("datetime64[Y]")).astype(int)  # 0..365
    hour = (ts - day).astype(int)
    return np.minimum(doy, 364), hour  # fold Dec 31 of leap years (never present) / day 365 onto 364


def train_climatology(data_dir, variables: Sequence[str], doy_window: int = 7) -> np.ndarray:
    """(365, 24, V, H, W) day-of-year x hour climatology from the train split."""
    files = shard_files(data_dir, "train")
    with np.load(files[0]) as d:
        hw = d[variables[0]].shape[-2:]
    sums = np.zeros((365, 24, len(variables)) + hw, dtype=np.float64)
    counts = np.zeros((365, 24), dtype=np.int64)
    for f in files:
        y, k = map(int, SHARD_RE.match(f.name).groups())
        with np.load(f) as d:
            arr = np.concatenate([d[v] for v in variables], axis=1)
        doy, hour = _doy_hour(shard_timestamps(y, k, arr.shape[0]))
        np.add.at(sums, (doy, hour), arr)
        np.add.at(counts, (doy, hour), 1)
    # circular smoothing over day of year
    kernel_sums = np.zeros_like(sums)
    kernel_counts = np.zeros_like(counts)
    for off in range(-doy_window, doy_window + 1):
        kernel_sums += np.roll(sums, off, axis=0)
        kernel_counts += np.roll(counts, off, axis=0)
    return (kernel_sums / kernel_counts[:, :, None, None, None]).astype(np.float32)


def iter_pairs(data_dir, split: str, variables: Sequence[str], lead: int, max_files: Optional[int] = None):
    """Yield (x, y, target_times) per shard, physical units, shapes (N, V, H, W)."""
    files = shard_files(data_dir, split)[:max_files]
    for f in files:
        y, k = map(int, SHARD_RE.match(f.name).groups())
        with np.load(f) as d:
            arr = np.concatenate([d[v].astype(np.float32) for v in variables], axis=1)
        ts = shard_timestamps(y, k, arr.shape[0])
        yield arr[:-lead], arr[lead:], ts[lead:]


def _lat_w(lat):
    w = np.cos(np.deg2rad(lat))
    return (w / w.mean())[None, None, :, None]


class Scorer:
    """Accumulates lat-weighted RMSE and ACC per variable for one forecast source."""

    def __init__(self, lat, n_vars, cols: slice = slice(None)):
        self.w = _lat_w(lat)
        self.cols = cols
        self.rmse_sum = np.zeros(n_vars)
        self.n = 0
        self.acc_num = np.zeros(n_vars)
        self.acc_p = np.zeros(n_vars)
        self.acc_o = np.zeros(n_vars)

    def update(self, pred, target, clim):
        c = self.cols
        pred, target, clim = pred[..., c], target[..., c], clim[..., c]
        w = self.w
        self.rmse_sum += np.sqrt((((pred - target) ** 2) * w).mean(axis=(2, 3))).sum(axis=0)
        self.n += pred.shape[0]
        pa, oa = pred - clim, target - clim
        self.acc_num += (w * pa * oa).sum(axis=(0, 2, 3))
        self.acc_p += (w * pa**2).sum(axis=(0, 2, 3))
        self.acc_o += (w * oa**2).sum(axis=(0, 2, 3))

    def result(self) -> Dict[str, np.ndarray]:
        return {"rmse": self.rmse_sum / self.n, "acc": self.acc_num / np.sqrt(self.acc_p * self.acc_o), "n": self.n}


def evaluate(
    data_dir,
    split: str,
    variables: Sequence[str],
    out_variables: Sequence[str],
    lead: int,
    clim: np.ndarray,
    models: Optional[Dict[str, Callable[[np.ndarray], np.ndarray]]] = None,
    column_sets: Optional[Dict[str, slice]] = None,
    max_files: Optional[int] = None,
) -> List[Dict]:
    """Score persistence, climatology and optional model callables.

    ``models`` map a name to ``f(x_phys (N, V_in, H, W)) -> y_phys (N, V_out, H, W')``;
    a model may return a wider grid (e.g. the internship 6x6 padded model), in which case
    ``column_sets`` selects which columns are scored, e.g. {"real": slice(0, 5), "all": slice(None)}.
    """
    lat = np.load(f"{data_dir}/lat.npy")
    out_idx = [list(variables).index(v) for v in out_variables]
    column_sets = column_sets or {"real": slice(None)}
    models = models or {}
    scorers = {}

    for x, y, ts in iter_pairs(data_dir, split, variables, lead, max_files):
        doy, hour = _doy_hour(ts)
        c = clim[doy, hour][:, out_idx]
        yo = y[:, out_idx]
        preds = {"persistence": x[:, out_idx], "climatology": c}
        for name, fn in models.items():
            preds[name] = fn(x)
        for name, p in preds.items():
            for cs_name, cs in column_sets.items():
                w = p.shape[-1]
                if w != yo.shape[-1]:  # model grid wider than data: score only real columns, pad target for "all"
                    yy, cc = pad_lon_edge(yo, yo.shape[-1]), pad_lon_edge(c, c.shape[-1])
                else:
                    yy, cc = yo, c
                    if cs_name != "real":
                        continue
                key = (name, cs_name)
                if key not in scorers:
                    scorers[key] = Scorer(lat, len(out_variables), cs)
                scorers[key].update(p, yy, cc)

    rows = []
    for (name, cs_name), s in scorers.items():
        r = s.result()
        for i, v in enumerate(out_variables):
            rows.append({"source": name, "columns": cs_name, "variable": v, "rmse": float(r["rmse"][i]), "acc": float(r["acc"][i]), "n": r["n"]})
    return rows


def climax_predictor(model, variables, out_variables, mean, std, lead_hours, lat, device="cpu", batch_size=64, pad_to: Optional[int] = None):
    """Wrap a ClimaX net as ``f(x_phys) -> y_phys``; optionally edge-pads lon in memory (internship setup)."""
    import torch

    from climax.utils.metrics import lat_weighted_mse

    mu = np.concatenate([mean[v] for v in variables])[None, :, None, None].astype(np.float32)
    sd = np.concatenate([std[v] for v in variables])[None, :, None, None].astype(np.float32)
    oi = [list(variables).index(v) for v in out_variables]
    model = model.to(device).eval()

    def predict(x):
        if pad_to is not None and x.shape[-1] != pad_to:
            x = pad_lon_edge(x, pad_to - 1)
        xn = torch.from_numpy((x - mu) / sd)
        out = []
        with torch.no_grad():
            for i in range(0, len(xn), batch_size):
                xb = xn[i : i + batch_size].to(device)
                lt = torch.full((len(xb),), lead_hours / 100.0, device=device)
                _, p = model(xb, xb[:, oi], lt, list(variables), list(out_variables), [lat_weighted_mse], lat=lat)
                out.append(p.float().cpu().numpy())
        return np.concatenate(out) * sd[:, oi] + mu[:, oi]

    return predict
