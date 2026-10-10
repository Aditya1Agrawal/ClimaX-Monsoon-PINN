"""IMD 0.25 deg gridded daily rainfall -> regional 5.625 deg boxes -> extreme-precipitation labels.

Source: Pai et al. (2014), Mausam 65(1), 1-18. Yearly binary files ``rain/<year>.grd``
(as written by ``imdlib``): little-endian float32, shape (days, 129 lat, 135 lon),
lat 6.5..38.5 N, lon 66.5..100.0 E, step 0.25 deg, missing = -999 (sea / outside India).

Regridding is area-conservative: each 0.25 deg source cell contributes to a target box in
proportion to its overlapping area (cos-latitude weighted). A box value is the mean over
its *valid* overlap, and ``coverage`` is the valid fraction of the whole box area. Boxes
with coverage below ``min_coverage`` are NaN, so a box that is mostly sea is never labelled
from a few coastal pixels.

Day convention: an IMD date refers to a 24 h accumulation ending at 08:30 IST (03 UTC).
Whether date D covers 03 UTC D-1 -> D or D -> D+1 must be checked against the IMD
documentation shipped with the data; it is a parameter (``imd_day_starts_on_date``) and
it shifts forecast alignment by one day.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

from climax_hazards import labels as L

IMD_LAT = 6.5 + 0.25 * np.arange(129)
IMD_LON = 66.5 + 0.25 * np.arange(135)
IMD_RES = 0.25
IMD_MISSING = -999.0


def read_imd_year(imd_dir, year: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dates, rain[days, 129, 135]) with NaN for missing values."""
    path = os.path.join(imd_dir, "rain", f"{year}.grd")
    raw = np.fromfile(path, dtype="<f4")
    n_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    if raw.size != n_days * IMD_LAT.size * IMD_LON.size:
        raise ValueError(f"{path}: {raw.size} values, expected {n_days} x 129 x 135 (incomplete download?)")
    rain = raw.reshape(n_days, IMD_LAT.size, IMD_LON.size)
    rain = np.where(rain <= IMD_MISSING + 1, np.nan, rain).astype(np.float32)
    dates = np.datetime64(f"{year}-01-01") + np.arange(n_days).astype("timedelta64[D]")
    return dates, rain


def _overlap_1d(src_centers, src_half, tgt_centers, tgt_half):
    """(n_tgt, n_src) length of overlap between source and target intervals."""
    lo = np.maximum(src_centers[None] - src_half, tgt_centers[:, None] - tgt_half)
    hi = np.minimum(src_centers[None] + src_half, tgt_centers[:, None] + tgt_half)
    return np.clip(hi - lo, 0, None)


def conservative_weights(tgt_lat, tgt_lon, tgt_res=5.625):
    """Weights (n_tlat, n_slat) and (n_tlon, n_slon) plus full box areas for normalisation."""
    wlat = _overlap_1d(IMD_LAT, IMD_RES / 2, np.asarray(tgt_lat), tgt_res / 2) * np.cos(np.deg2rad(IMD_LAT))[None]
    wlon = _overlap_1d(IMD_LON, IMD_RES / 2, np.asarray(tgt_lon), tgt_res / 2)
    # exact box area in the same (deg * cos-lat) units, integrating cos over each target lat band
    lat_lo = np.deg2rad(np.asarray(tgt_lat) - tgt_res / 2)
    lat_hi = np.deg2rad(np.asarray(tgt_lat) + tgt_res / 2)
    band = np.rad2deg(np.sin(lat_hi) - np.sin(lat_lo))  # = integral of cos(lat) dlat, in degrees
    box_area = band[:, None] * tgt_res
    return wlat, wlon, box_area


def regrid_to_boxes(rain, wlat, wlon, box_area):
    """rain (D, 129, 135) -> (box mean (D, H, W), coverage (D, H, W))."""
    valid = np.isfinite(rain).astype(np.float64)
    r = np.nan_to_num(rain).astype(np.float64)
    num = np.einsum("hi,dij,wj->dhw", wlat, r, wlon)
    den = np.einsum("hi,dij,wj->dhw", wlat, valid, wlon)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = num / den
    return mean.astype(np.float32), (den / box_area[None]).astype(np.float32)


def imd_boxes(imd_dir, years, tgt_lat, tgt_lon, tgt_res=5.625):
    """Daily box-mean rainfall and coverage for ``years`` (concatenated, chronological)."""
    wlat, wlon, area = conservative_weights(tgt_lat, tgt_lon, tgt_res)
    dates, means, covs = [], [], []
    for y in years:
        d, rain = read_imd_year(imd_dir, y)
        m, c = regrid_to_boxes(rain, wlat, wlon, area)
        dates.append(d), means.append(m), covs.append(c)
    return np.concatenate(dates), np.concatenate(means), np.concatenate(covs)


@dataclass
class PrecipConfig:
    percentile: float = 95.0
    wet_day_mm: float = 1.0          # ETCCDI wet-day threshold; IMD "rainy day" is 2.5 mm
    min_coverage: float = 0.5        # fraction of the box area with valid IMD data
    season_months: Optional[Tuple[int, ...]] = None   # None = all year
    absolute_mm: Optional[float] = None               # e.g. IMD heavy rain 64.5 (station-scale, comparison only)
    min_duration: int = 1
    imd_day_starts_on_date: bool = True  # see module docstring; UNVERIFIED


def precip_labels(data_dir, imd_dir, cfg: PrecipConfig = PrecipConfig(), split_years=None, cache=None) -> L.LabelSet:
    """Extreme-precipitation labels on the regional grid of ``data_dir``.

    Threshold (per box): ``percentile`` of train-period *wet-day* box-mean rainfall
    (R95p-style, ETCCDI), or ``absolute_mm`` if given.
    """
    from climax_hazards.inspect_utils import SHARD_RE, shard_files

    lat, lon = np.load(os.path.join(data_dir, "lat.npy")), np.load(os.path.join(data_dir, "lon.npy"))
    if split_years is None:
        split_years = {s: sorted({int(SHARD_RE.match(f.name).group(1)) for f in shard_files(data_dir, s)}) for s in ("train", "val", "test")}
    cache = cache if cache is not None else {}
    daily = {}
    for s, yrs in split_years.items():
        key = ("imd", s, tuple(yrs))
        if key not in cache:
            cache[key] = imd_boxes(imd_dir, yrs, lat, lon)
        dates, rain, cov = cache[key]
        daily[s] = (dates, np.where(cov >= cfg.min_coverage, rain, np.nan).astype(np.float32), cov)

    _, r_tr, _ = daily["train"]
    if cfg.absolute_mm is not None:
        thr = np.full(r_tr.shape[1:], cfg.absolute_mm, dtype=np.float32)
    else:
        wet = np.where(r_tr >= cfg.wet_day_mm, r_tr, np.nan)
        with warnings.catch_warnings():  # boxes without IMD coverage are all-NaN -> NaN threshold
            warnings.simplefilter("ignore", RuntimeWarning)
            thr = np.nanpercentile(wet, cfg.percentile, axis=0).astype(np.float32)

    ls = L.LabelSet("extreme_precip", asdict(cfg), {"rain_threshold_mm": thr})
    for s, (dates, r, cov) in daily.items():
        e = np.where(np.isnan(r) | np.isnan(thr)[None], np.nan, (r > thr[None]).astype(np.float32)).astype(np.float32)
        lab = L.season_filter(dates, L.spell_labels(e, cfg.min_duration), cfg.season_months)
        ls.splits[s] = {"dates": dates, "value": r, "label": lab, "coverage": cov}
    return ls
