"""Hazard label generation from the hourly regional dataset (Phase 4).

Conventions
* Days are UTC days. A day needs all 24 hourly values; otherwise its aggregate is NaN
  (in this dataset the only gaps are 31 December of leap years).
* Labels are float arrays: 1 = event, 0 = no event, NaN = undetermined (missing data,
  masked cell, or a spell cut off by a gap/split boundary). NaN must never be treated as 0.
* Every data-derived threshold is computed from the **train split only** and then
  applied unchanged to val/test.
* Spells are evaluated inside one split, so no label depends on another split's data.

Definitions implemented (see docs/mtech/HAZARD_DEFINITIONS.md for the rationale)
* heatwave, percentile/duration: daily Tmax > calendar-day percentile (+-window days,
  train years) on >= ``min_duration`` consecutive days (Perkins & Alexander 2013 style),
  scored only in the hot season (default March-June, IMD); other days are NaN.
* heatwave, IMD-style: Tmax >= 40 C (>= 30 C for cells with orography > 1500 m) AND
  departure from the train calendar-day mean >= 4.5 C, or Tmax >= 45 C; on
  >= ``min_duration`` days. IMD criteria are station-based; at 5.625 deg this is only a
  comparison, not the primary label.
* extreme wind: daily max of hourly sqrt(u10^2 + v10^2) > per-cell percentile of the
  train daily maxima (optionally per calendar month), or an absolute threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from climax_hazards.inspect_utils import SHARD_RE, shard_files, shard_timestamps

K0 = 273.15


# ----------------------------------------------------------------------------
# Loading and daily aggregation
# ----------------------------------------------------------------------------


def load_hourly(data_dir, split: str, variables: Sequence[str]):
    """Continuous hourly series for a split: (times, {var: (T, H, W)}) with NaN for missing hours."""
    files = shard_files(data_dir, split)
    parts, times = {v: [] for v in variables}, []
    for f in files:
        y, k = map(int, SHARD_RE.match(f.name).groups())
        with np.load(f) as d:
            n = d[variables[0]].shape[0]
            for v in variables:
                parts[v].append(d[v][:, 0].astype(np.float32))
        times.append(shard_timestamps(y, k, n))
    times = np.concatenate(times).astype("datetime64[h]")
    order = np.argsort(times)
    times = times[order]
    full = np.arange(times[0], times[-1] + np.timedelta64(1, "h"), dtype="datetime64[h]")
    idx = (times - full[0]).astype(int)
    out = {}
    for v in variables:
        a = np.concatenate(parts[v])[order]
        filled = np.full((full.size,) + a.shape[1:], np.nan, dtype=np.float32)
        filled[idx] = a
        out[v] = filled
    return full, out


def daily_aggregate(times: np.ndarray, hourly: np.ndarray, how: str = "max"):
    """UTC-day aggregate; days with any missing hour become NaN. Returns (dates, (D, H, W))."""
    first = times[0].astype("datetime64[D]")
    if times[0] != first.astype("datetime64[h]"):
        raise ValueError("hourly series must start at 00 UTC")
    n_days = hourly.shape[0] // 24
    a = hourly[: n_days * 24].reshape((n_days, 24) + hourly.shape[1:])
    agg = {"max": np.max, "mean": np.mean, "min": np.min}[how](a, axis=1)  # NaN propagates -> incomplete day = NaN
    dates = first + np.arange(n_days).astype("timedelta64[D]")
    return dates, agg


def day_of_year(dates: np.ndarray) -> np.ndarray:
    d = dates.astype("datetime64[D]")
    doy = (d - d.astype("datetime64[Y]")).astype(int)
    return np.minimum(doy, 364)


def calendar_percentile(dates, values, q: float, window: int = 7) -> np.ndarray:
    """(365, H, W) percentile of ``values`` over all days within +-window calendar days."""
    doy = day_of_year(dates)
    out = np.full((365,) + values.shape[1:], np.nan, dtype=np.float32)
    for d in range(365):
        dist = np.abs(doy - d)
        sel = np.minimum(dist, 365 - dist) <= window
        out[d] = np.nanpercentile(values[sel], q, axis=0)
    return out


def calendar_mean(dates, values, window: int = 7) -> np.ndarray:
    doy = day_of_year(dates)
    out = np.full((365,) + values.shape[1:], np.nan, dtype=np.float32)
    for d in range(365):
        dist = np.abs(doy - d)
        sel = np.minimum(dist, 365 - dist) <= window
        out[d] = np.nanmean(values[sel], axis=0)
    return out


# ----------------------------------------------------------------------------
# Spell logic
# ----------------------------------------------------------------------------


def spell_labels(exceed: np.ndarray, min_duration: int) -> np.ndarray:
    """Mark days belonging to runs of >= min_duration exceedances.

    ``exceed`` is float (1/0/NaN) of shape (D, ...). A run shorter than ``min_duration``
    that touches a NaN day or the split boundary is undetermined (NaN), because the
    unseen days could have completed it.
    """
    D = exceed.shape[0]
    flat = exceed.reshape(D, -1)
    out = np.where(np.isnan(flat), np.nan, 0.0).astype(np.float32)
    if min_duration <= 1:
        return np.where(np.isnan(flat), np.nan, flat).reshape(exceed.shape).astype(np.float32)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        t = 0
        while t < D:
            if col[t] == 1:
                s = t
                while t < D and col[t] == 1:
                    t += 1
                length = t - s
                open_left = s == 0 or np.isnan(col[s - 1])
                open_right = t == D or np.isnan(col[t])
                if length >= min_duration:
                    out[s:t, c] = 1.0
                elif open_left or open_right:
                    out[s:t, c] = np.nan
            else:
                t += 1
    return out.reshape(exceed.shape)


def season_filter(dates, label, months):
    """Set days outside ``months`` (1-12) to NaN; ``None`` keeps all days."""
    if months is None:
        return label
    m = dates.astype("datetime64[M]").astype(int) % 12 + 1
    out = label.copy()
    out[~np.isin(m, months)] = np.nan
    return out


def _exceed(values, threshold, mask=None):
    e = np.where(np.isnan(values) | np.isnan(threshold), np.nan, (values > threshold).astype(np.float32))
    if mask is not None:
        e = np.where(mask[None], e, np.nan)
    return e.astype(np.float32)


# ----------------------------------------------------------------------------
# Hazard definitions
# ----------------------------------------------------------------------------


@dataclass
class HeatwaveConfig:
    percentile: float = 90.0
    window_days: int = 7
    min_duration: int = 3
    land_fraction_min: float = 0.5
    # IMD: heat waves occur mainly March-June. Days outside these months are NaN (not scored).
    # Spells are found on all days first, so a spell crossing the season edge is still detected.
    season_months: Optional[Tuple[int, ...]] = (3, 4, 5, 6)
    imd_plains_c: float = 40.0
    imd_hills_c: float = 30.0
    imd_hill_orography_m: float = 1500.0
    imd_departure_c: float = 4.5
    imd_absolute_c: float = 45.0
    imd_min_duration: int = 2


@dataclass
class WindConfig:
    percentile: float = 99.0
    monthly: bool = False
    absolute_ms: Optional[float] = None
    min_duration: int = 1
    land_fraction_min: Optional[float] = None  # None = all cells


@dataclass
class LabelSet:
    name: str
    config: Dict
    thresholds: Dict[str, np.ndarray]
    splits: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)  # split -> {dates, value, label}


def land_mask(data_dir, min_fraction: Optional[float]) -> Optional[np.ndarray]:
    if min_fraction is None:
        return None
    with np.load(shard_files(data_dir, "train")[0]) as d:
        return d["land_sea_mask"][0, 0] >= min_fraction


def orography(data_dir) -> np.ndarray:
    with np.load(shard_files(data_dir, "train")[0]) as d:
        return d["orography"][0, 0]


def daily_tmax(data_dir, split):
    times, h = load_hourly(data_dir, split, ["2m_temperature"])
    return daily_aggregate(times, h["2m_temperature"] - K0, "max")


def daily_max_wind(data_dir, split):
    times, h = load_hourly(data_dir, split, ["10m_u_component_of_wind", "10m_v_component_of_wind"])
    ws = np.sqrt(h["10m_u_component_of_wind"] ** 2 + h["10m_v_component_of_wind"] ** 2)
    return daily_aggregate(times, ws, "max")


def heatwave_labels(data_dir, cfg: HeatwaveConfig = HeatwaveConfig(), splits=("train", "val", "test"), cache=None) -> Dict[str, LabelSet]:
    """Returns {'heatwave_pct': LabelSet, 'heatwave_imd': LabelSet}."""
    cache = cache if cache is not None else {}
    daily = {s: cache.get(("tmax", s)) or daily_tmax(data_dir, s) for s in splits}
    cache.update({("tmax", s): v for s, v in daily.items()})
    mask = land_mask(data_dir, cfg.land_fraction_min)
    oro = orography(data_dir)
    dates_tr, tmax_tr = daily["train"]
    thr = calendar_percentile(dates_tr, tmax_tr, cfg.percentile, cfg.window_days)
    normal = calendar_mean(dates_tr, tmax_tr, cfg.window_days)
    oro_ok = bool(np.ptp(oro) > 0)
    hill = oro > cfg.imd_hill_orography_m
    abs_thr = np.where(hill, cfg.imd_hills_c, cfg.imd_plains_c)

    pct = LabelSet("heatwave_pct", asdict(cfg), {"tmax_threshold_c": thr})
    imd = LabelSet("heatwave_imd", {**asdict(cfg), "orography_available": oro_ok}, {"tmax_normal_c": normal, "absolute_threshold_c": abs_thr})
    for s in splits:
        dates, tmax = daily[s]
        doy = day_of_year(dates)
        e = _exceed(tmax, thr[doy], mask)
        pct.splits[s] = {"dates": dates, "value": tmax, "label": season_filter(dates, spell_labels(e, cfg.min_duration), cfg.season_months)}
        dep = tmax - normal[doy]
        crit = ((tmax >= abs_thr[None]) & (dep >= cfg.imd_departure_c)) | (tmax >= cfg.imd_absolute_c)
        e2 = np.where(np.isnan(tmax), np.nan, crit.astype(np.float32))
        if mask is not None:
            e2 = np.where(mask[None], e2, np.nan)
        imd.splits[s] = {"dates": dates, "value": tmax, "departure": dep,
                         "label": season_filter(dates, spell_labels(e2.astype(np.float32), cfg.imd_min_duration), cfg.season_months)}
    return {"heatwave_pct": pct, "heatwave_imd": imd}


def wind_labels(data_dir, cfg: WindConfig = WindConfig(), splits=("train", "val", "test"), cache=None) -> LabelSet:
    cache = cache if cache is not None else {}
    daily = {s: cache.get(("wind", s)) or daily_max_wind(data_dir, s) for s in splits}
    cache.update({("wind", s): v for s, v in daily.items()})
    mask = land_mask(data_dir, cfg.land_fraction_min)
    dates_tr, w_tr = daily["train"]
    if cfg.absolute_ms is not None:
        thr = np.full((12,) + w_tr.shape[1:], cfg.absolute_ms, dtype=np.float32)
    elif cfg.monthly:
        months = dates_tr.astype("datetime64[M]").astype(int) % 12
        thr = np.stack([np.nanpercentile(w_tr[months == m], cfg.percentile, axis=0) for m in range(12)]).astype(np.float32)
    else:
        thr = np.repeat(np.nanpercentile(w_tr, cfg.percentile, axis=0)[None], 12, axis=0).astype(np.float32)
    ls = LabelSet("extreme_wind", asdict(cfg), {"wind_threshold_ms_by_month": thr})
    for s in splits:
        dates, w = daily[s]
        m = dates.astype("datetime64[M]").astype(int) % 12
        ls.splits[s] = {"dates": dates, "value": w, "label": spell_labels(_exceed(w, thr[m], mask), cfg.min_duration)}
    return ls


# ----------------------------------------------------------------------------
# Summaries and forecast alignment
# ----------------------------------------------------------------------------


def prevalence(label: np.ndarray) -> Dict[str, float]:
    valid = ~np.isnan(label)
    n_valid = int(valid.sum())
    n_pos = int(np.nansum(label))
    return {"n_valid": n_valid, "n_event": n_pos, "n_undetermined": int((~valid).sum()), "prevalence": n_pos / n_valid if n_valid else float("nan")}


def align_to_forecast(input_times: np.ndarray, lead_hours: int, dates: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Label of the UTC day containing ``t + lead`` for each input time (NaN if outside the split)."""
    target_day = (input_times.astype("datetime64[h]") + np.timedelta64(lead_hours, "h")).astype("datetime64[D]")
    idx = (target_day - dates[0]).astype(int)
    out = np.full((len(idx),) + label.shape[1:], np.nan, dtype=np.float32)
    ok = (idx >= 0) & (idx < len(dates))
    out[ok] = label[idx[ok]]
    return out


def save_labelset(ls: LabelSet, out_dir):
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{ls.name}_thresholds.npz"), **ls.thresholds)
    for s, d in ls.splits.items():
        np.savez_compressed(os.path.join(out_dir, f"{ls.name}_{s}.npz"), dates=d["dates"].astype("datetime64[D]").astype("int64"), label=d["label"], value=d["value"])
    meta = {"name": ls.name, "config": ls.config, "dates_encoding": "int64 days since 1970-01-01 (UTC days)",
            "label_encoding": "1 event, 0 no event, NaN undetermined/masked", "thresholds_from": "train split only",
            "prevalence": {s: prevalence(d["label"]) for s, d in ls.splits.items()}}
    json.dump(meta, open(os.path.join(out_dir, f"{ls.name}_metadata.json"), "w"), indent=1)
    return meta
