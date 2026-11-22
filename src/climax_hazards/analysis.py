"""Post-hoc analysis of saved hazard predictions: calibration, skill maps, blending, case studies.

Everything works on the ``preds/*.npz`` files written by ``run_hazard_experiment.py``
(``val`` / ``test`` probabilities of shape (N, K, H, W), one sample per 00 UTC issue day)
and ``preds/targets.npz`` (labels of the target day, NaN where undefined). No model is run.
Any fitted quantity (calibration map, blend weight) is fitted on validation only and
applied unchanged to test.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from climax_hazards.hazard_model import _flat, average_precision

EPS = 1e-6


# ----------------------------------------------------------------------------
# Probabilistic scores
# ----------------------------------------------------------------------------


def brier(prob, y) -> float:
    p, t = _flat(prob, y)
    return float(np.mean((p - t) ** 2)) if t.size else float("nan")


def brier_skill(prob, ref, y) -> float:
    """1 - BS(prob)/BS(ref); > 0 means better than the reference (e.g. climatology)."""
    b_ref = brier(ref, y)
    return 1.0 - brier(prob, y) / b_ref if b_ref > 0 else float("nan")


def reliability(prob, y, n_bins: int = 10) -> Tuple[pd.DataFrame, float]:
    """Reliability table on equal-width probability bins, and the expected calibration error.

    Returns (DataFrame[bin_low, bin_high, mean_pred, obs_freq, count], ECE), where
    ECE = sum_b (n_b / n) |obs_freq_b - mean_pred_b| over non-empty bins.
    """
    p, t = _flat(prob, y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        m = b == i
        n = int(m.sum())
        mp = float(p[m].mean()) if n else float("nan")
        of = float(t[m].mean()) if n else float("nan")
        if n:
            ece += n / max(t.size, 1) * abs(of - mp)
        rows.append({"bin_low": edges[i], "bin_high": edges[i + 1], "mean_pred": mp, "obs_freq": of, "count": n})
    return pd.DataFrame(rows), float(ece)


class PlattCalibrator:
    """Platt scaling on the logit of the probability, pooled over valid cells.

    p_cal = sigmoid(a * logit(p) + b). It is monotonic, so ranking metrics (PR-AUC, ROC-AUC)
    are unchanged; only the probability values (reliability, Brier) change. Fitted on
    validation, applied to test.
    """

    def __init__(self):
        self.a, self.b = 1.0, 0.0

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
        return np.log(p / (1 - p))

    def fit(self, prob, y) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        p, t = _flat(prob, y)
        if t.size == 0 or t.min() == t.max():
            raise ValueError("calibration needs both classes")
        clf = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p)[:, None], t)
        self.a, self.b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
        return self

    def transform(self, prob):
        z = self.a * self._logit(prob) + self.b
        return 1.0 / (1.0 + np.exp(-z))   # float64: avoids float32 ties that would perturb PR-AUC


# ----------------------------------------------------------------------------
# Spatial skill
# ----------------------------------------------------------------------------


def per_cell_ap(prob, y, min_pos: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """(N, H, W) -> per-cell PR-AUC (NaN where fewer than ``min_pos`` positives) and positive counts."""
    H, W = y.shape[1:]
    ap = np.full((H, W), np.nan)
    npos = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            t = y[:, i, j]
            m = np.isfinite(t)
            npos[i, j] = int(np.nansum(t[m]))
            if npos[i, j] >= min_pos and npos[i, j] < m.sum():
                ap[i, j] = average_precision(prob[:, i, j], t)
    return ap, npos


# ----------------------------------------------------------------------------
# Blending
# ----------------------------------------------------------------------------


def blend(pa, pb, w: float):
    return w * pa + (1.0 - w) * pb


def choose_blend_weight(pa_val, pb_val, y_val, grid: Iterable[float] = np.linspace(0, 1, 11)) -> Tuple[float, pd.DataFrame]:
    """Weight w of p = w*pa + (1-w)*pb maximising validation PR-AUC (ties -> the larger w)."""
    rows = [{"w": float(w), "val_pr_auc": average_precision(blend(pa_val, pb_val, w), y_val)} for w in grid]
    df = pd.DataFrame(rows)
    best = df.loc[df["val_pr_auc"][::-1].idxmax(), "w"]
    return float(best), df


# ----------------------------------------------------------------------------
# Case studies
# ----------------------------------------------------------------------------


def target_days(issue_times_min: np.ndarray, lead_days: int = 3) -> np.ndarray:
    """Issue times (int64 minutes since epoch, as in targets.npz) -> target dates (datetime64[D])."""
    return (issue_times_min.astype("datetime64[m]") + np.timedelta64(lead_days, "D")).astype("datetime64[D]")


def nearest_cells(lat: np.ndarray, lon: np.ndarray, points: Sequence[Tuple[float, float]]) -> list:
    """(lat, lon) points -> unique (i, j) indices of the grid cells whose centres are nearest."""
    out = []
    for la, lo in points:
        c = (int(np.argmin(np.abs(lat - la))), int(np.argmin(np.abs(lon - lo))))
        if c not in out:
            out.append(c)
    return out


def event_series(probs: Dict[str, np.ndarray], y: np.ndarray, days: np.ndarray, cells: Sequence[Tuple[int, int]],
                 start: str, end: str) -> pd.DataFrame:
    """Per target day in [start, end]: observed label (max over cells; NaN if none defined) and each source's max probability."""
    m = (days >= np.datetime64(start)) & (days <= np.datetime64(end))
    if not m.any():
        raise ValueError(f"no samples between {start} and {end}")
    ii, jj = zip(*cells)
    obs = y[m][:, list(ii), list(jj)]
    df = pd.DataFrame({"target_day": days[m]})
    with np.errstate(all="ignore"):
        df["observed"] = np.where(np.isfinite(obs).any(1), np.nanmax(np.where(np.isfinite(obs), obs, -np.inf), axis=1), np.nan)
    for name, p in probs.items():
        df[name] = p[m][:, list(ii), list(jj)].max(axis=1)
    return df
