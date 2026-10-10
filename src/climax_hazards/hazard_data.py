"""Samples, aligned labels and reference baselines for hazard prediction (Option A).

Task: a forecast issued at ``issue_hours`` UTC (default 00 UTC) on day D, from the
input fields at that hour, predicts whether a hazard occurs on the UTC day containing
``t + lead`` (D+3 for 00 UTC and 72 h). Labels come from ``climax_hazards.labels`` /
``climax_hazards.precip``; NaN labels (masked cells, missing days, undetermined spells,
targets outside the split) are excluded from losses and metrics.

Baselines use only information available at issue time:
* climatology: train-period event frequency of the target calendar day (+-15 days) per cell;
* persistence: threshold exceedance on the latest fully observed day, D-1 for ERA5 heat
  and wind, D-2 for IMD rain (its 03 UTC day boundary makes D-1 unavailable at 00 UTC).
"""

from __future__ import annotations

import warnings
from typing import Dict, Sequence

import numpy as np

from climax_hazards import labels as L
from climax_hazards import precip as P
from climax_hazards.inspect_utils import SHARD_RE, shard_files, shard_timestamps

HAZARDS = ("heatwave", "extreme_wind", "extreme_precip")
PERSISTENCE_LAG_DAYS = {"heatwave": 1, "extreme_wind": 1, "extreme_precip": 2}


def build_labelsets(data_dir, imd_dir=None, hazards: Sequence[str] = HAZARDS) -> Dict[str, L.LabelSet]:
    """Primary label definitions (docs/mtech/HAZARD_DEFINITIONS.md) keyed by hazard name."""
    cache, out = {}, {}
    if "heatwave" in hazards:
        out["heatwave"] = L.heatwave_labels(data_dir, L.HeatwaveConfig(), cache=cache)["heatwave_pct"]
    if "extreme_wind" in hazards:
        out["extreme_wind"] = L.wind_labels(data_dir, L.WindConfig(), cache=cache)
    if "extreme_precip" in hazards:
        if imd_dir is None:
            raise ValueError("extreme_precip needs imd_dir")
        out["extreme_precip"] = P.precip_labels(data_dir, imd_dir, P.PrecipConfig(), cache=cache)
    return out


def load_inputs(data_dir, split: str, variables: Sequence[str], issue_hours=(0,)):
    """Input fields at the issue hours: (times (N,), x (N, V, H, W) physical units)."""
    times, xs = [], []
    for f in shard_files(data_dir, split):
        y, k = map(int, SHARD_RE.match(f.name).groups())
        with np.load(f) as d:
            n = d[variables[0]].shape[0]
            ts = shard_timestamps(y, k, n)
            hours = (ts - ts.astype("datetime64[D]")).astype(int)
            sel = np.isin(hours, issue_hours)
            xs.append(np.concatenate([d[v][sel].astype(np.float32) for v in variables], axis=1))
        times.append(ts[sel])
    times = np.concatenate(times)
    order = np.argsort(times)
    return times[order], np.concatenate(xs)[order]


def aligned_targets(labelsets: Dict[str, L.LabelSet], split: str, times, lead_hours: int) -> np.ndarray:
    """(N, K, H, W) labels for the day containing t + lead, NaN where undefined."""
    return np.stack([L.align_to_forecast(times, lead_hours, ls.splits[split]["dates"], ls.splits[split]["label"])
                     for ls in labelsets.values()], axis=1)


def _exceedance(name, ls, split):
    """Raw daily threshold exceedance (1/0/NaN), without spell or season logic."""
    d = ls.splits[split]
    v, dates = d["value"], d["dates"]
    if name == "heatwave":
        thr = ls.thresholds["tmax_threshold_c"][L.day_of_year(dates)]
    elif name == "extreme_wind":
        thr = ls.thresholds["wind_threshold_ms_by_month"][dates.astype("datetime64[M]").astype(int) % 12]
    else:
        thr = ls.thresholds["rain_threshold_mm"][None]
    return np.where(np.isnan(v) | np.isnan(thr), np.nan, (v > thr).astype(np.float32))


def persistence_probs(labelsets, split, times) -> np.ndarray:
    """(N, K, H, W) 0/1 forecast: exceedance on the latest fully observed day before issue."""
    out = []
    for name, ls in labelsets.items():
        ex = _exceedance(name, ls, split)
        dates = ls.splits[split]["dates"]
        day = times.astype("datetime64[D]") - np.timedelta64(PERSISTENCE_LAG_DAYS[name], "D")
        idx = (day - dates[0]).astype(int)
        p = np.full((len(times),) + ex.shape[1:], np.nan, dtype=np.float32)
        ok = (idx >= 0) & (idx < len(dates))
        p[ok] = ex[idx[ok]]
        out.append(np.nan_to_num(p, nan=0.0))  # unknown past -> predict no event
    return np.stack(out, axis=1)


def climatology_probs(labelsets, split, times, lead_hours: int, window: int = 15) -> np.ndarray:
    """(N, K, H, W) train-period event frequency for the target calendar day (+-window days)."""
    target = (times.astype("datetime64[h]") + np.timedelta64(lead_hours, "h")).astype("datetime64[D]")
    tdoy = L.day_of_year(target)
    out = []
    for ls in labelsets.values():
        tr = ls.splits["train"]
        doy = L.day_of_year(tr["dates"])
        lab = tr["label"]
        freq = np.zeros((365,) + lab.shape[1:], dtype=np.float32)
        for d in range(365):
            dist = np.abs(doy - d)
            sel = np.minimum(dist, 365 - dist) <= window
            with warnings.catch_warnings():  # all-NaN cells (masked) -> 0
                warnings.simplefilter("ignore", RuntimeWarning)
                freq[d] = np.nan_to_num(np.nanmean(lab[sel], axis=0), nan=0.0)
        out.append(freq[tdoy])
    return np.stack(out, axis=1)
