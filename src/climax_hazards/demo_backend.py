"""Backend of the live demo: run the trained ClimaX encoders + hazard heads on any test day.

What is loaded (all produced by notebook 07, never committed to git):
    <run_dir>/stageA/<encoder>/best_encoder.ckpt     grid-adapted encoders (net.* state dict)
    <run_dir>/stageB/heads/<encoder>_seed<k>.pt      frozen-encoder hazard heads (3 seeds per encoder)
    <run_dir>/stageB/preds/*.npz                     saved test/val probabilities (fallback, logistic, climatology)
    <analysis_dir>/calibration.csv                   Platt parameters fitted on validation (make_analysis.py)

"ClimaX" probabilities are the mean over the encoders and head seeds, exactly as in the analysis,
then Platt-calibrated; the alert threshold per hazard is the best validation F1 of those
calibrated probabilities. Live inference therefore reproduces the saved predictions (checked by
``live_vs_saved``). Inputs are the ERA5 fields at 00 UTC of the issue day; the forecast is for
the day 3 days later (72 h).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from climax_hazards import analysis as A
from climax_hazards import hazard_data as hd
from climax_hazards import hazard_model as hm
from climax_hazards import inspect_utils as iu
from climax_hazards.forecast_eval import hypsometric_residuals
from climax_hazards.joint import ADJACENT_LAYERS

logger = logging.getLogger(__name__)

WEATHER_VARS = {"2m_temperature": ("2 m temperature", "K"), "temperature_850": ("Temperature 850 hPa", "K"),
                "geopotential_500": ("Geopotential 500 hPa", "m²/s²"), "10m_u_component_of_wind": ("10 m zonal wind", "m/s"),
                "10m_v_component_of_wind": ("10 m meridional wind", "m/s")}


@dataclass
class DemoConfig:
    repo: str
    data_dir: str = "regional_climax_dataset_6x6_corrected"
    run_dir: str = "outputs/kaggle_v3"
    analysis_dir: str = "docs/mtech/results/analysis"
    config: str = "configs/regional_india.yaml"
    encoders: Sequence[str] = ("A_multi_s0", "A_multi_s1", "A_multi_s2")
    reference_encoder: str = "A_nophys_s0"       # physics comparison only
    reference_encoder_dir: Optional[str] = None  # default: <run_dir>/stageA, then the v2 output
    head_seeds: Sequence[int] = (0, 1, 2)
    lead_days: int = 3
    device: str = "cpu"

    def path(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.repo, p)


@dataclass
class Forecast:
    issue_day: np.datetime64
    target_day: np.datetime64
    hazards: List[str]
    prob: np.ndarray                 # (K, H, W) calibrated ClimaX probability (live or saved)
    prob_logistic: np.ndarray        # (K, H, W) calibrated logistic (saved)
    prob_climatology: np.ndarray     # (K, H, W)
    observed: np.ndarray             # (K, H, W) 1 / 0 / NaN
    alerts: np.ndarray               # (K, H, W) bool, prob >= threshold on valid cells
    thresholds: Dict[str, float]
    weather_input: Dict[str, np.ndarray]
    weather_pred: Optional[Dict[str, np.ndarray]]
    weather_true: Optional[Dict[str, np.ndarray]]
    live: bool
    seconds: float
    max_abs_diff_vs_saved: Optional[float] = None
    extra: Dict = field(default_factory=dict)


class DemoBackend:
    """Loads data once; models are loaded on demand (``load_models``) so the app can also run without them."""

    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        c = yaml.safe_load(open(cfg.path(cfg.config)))
        self.net_args = dict(c["model"]["net"]["init_args"])
        self.variables = list(c["data"]["variables"])
        self.lead_hours = c["data"]["predict_range"] * c["data"]["hrs_each_step"]
        if self.lead_hours != 24 * cfg.lead_days:
            raise ValueError(f"config lead {self.lead_hours} h != {cfg.lead_days} days")
        dd = cfg.path(cfg.data_dir)
        if iu.detect_lon_padding(dd)["looks_padded"]:
            raise ValueError("the demo needs the corrected (unpadded) 6x6 dataset")
        self.lat, self.lon = np.load(os.path.join(dd, "lat.npy")), np.load(os.path.join(dd, "lon.npy"))
        m, s = dict(np.load(os.path.join(dd, "normalize_mean.npz"))), dict(np.load(os.path.join(dd, "normalize_std.npz")))
        self.mu = np.concatenate([m[v] for v in self.variables])[None, :, None, None].astype(np.float32)
        self.sd = np.concatenate([s[v] for v in self.variables])[None, :, None, None].astype(np.float32)

        # 00 UTC test fields (inputs, and the truth 3 days later)
        self.times, self.x_phys = hd.load_inputs(dd, "test", self.variables, (0,))
        self.days = self.times.astype("datetime64[D]")
        self._day_index = {d: i for i, d in enumerate(self.days)}

        # saved predictions, targets, calibration, thresholds
        pdir = os.path.join(cfg.path(cfg.run_dir), "stageB", "preds")
        t = np.load(os.path.join(pdir, "targets.npz"))
        self.hazards = list(_load_json(os.path.join(cfg.path(cfg.run_dir), "stageB", "run_meta.json"))["hazards"])
        self.issue_days = t["test_times"].astype("datetime64[m]").astype("datetime64[D]")
        self.y = {"val": t["val_y"], "test": t["test_y"]}
        self.target_days = A.target_days(t["test_times"], cfg.lead_days)
        self._pred_index = {d: i for i, d in enumerate(self.issue_days)}
        enc_files = [os.path.join(pdir, f"{e}.npz") for e in cfg.encoders]
        missing = [f for f in enc_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"saved predictions missing: {missing}")
        raw = {sp: np.mean([np.load(f)[sp] for f in enc_files], axis=0) for sp in ("val", "test")}
        lg = dict(np.load(os.path.join(pdir, "logistic.npz")))
        self.clim = dict(np.load(os.path.join(pdir, "climatology.npz")))
        cal = pd.read_csv(os.path.join(cfg.path(cfg.analysis_dir), "calibration.csv"))
        self.platt = {(r.source, r.hazard): (float(r.a), float(r.b)) for r in cal.itertuples()}
        self.saved = {sp: self.calibrate(raw[sp], "ClimaX") for sp in ("val", "test")}
        self.logistic = {sp: self.calibrate(lg[sp], "logistic") for sp in ("val", "test")}
        self.thresholds = {h: hm.best_f1_threshold(self.saved["val"][:, k], self.y["val"][:, k]) for k, h in enumerate(self.hazards)}
        self.models = None
        self.reference = None
        logger.info("demo data ready: %d test issue days, hazards %s", len(self.issue_days), self.hazards)

    # ---------------- models ----------------
    def checkpoint_paths(self) -> Dict[str, str]:
        out = {}
        for e in list(self.cfg.encoders):
            out[e] = os.path.join(self.cfg.path(self.cfg.run_dir), "stageA", e, "best_encoder.ckpt")
        return out

    def reference_checkpoint(self) -> Optional[str]:
        cands = [self.cfg.reference_encoder_dir] if self.cfg.reference_encoder_dir else []
        cands += [os.path.join(self.cfg.path(self.cfg.run_dir), "stageA"), self.cfg.path("outputs/kaggle_v2/stageA")]
        for d in cands:
            p = os.path.join(d, self.cfg.reference_encoder, "best_encoder.ckpt")
            if os.path.exists(p):
                return p
        return None

    def models_available(self) -> bool:
        heads_dir = os.path.join(self.cfg.path(self.cfg.run_dir), "stageB", "heads")
        return all(os.path.exists(p) for p in self.checkpoint_paths().values()) and all(
            os.path.exists(os.path.join(heads_dir, f"{e}_seed{s}.pt")) for e in self.cfg.encoders for s in self.cfg.head_seeds)

    def _load_net(self, path):
        from climax.arch import ClimaX

        net = ClimaX(**self.net_args)
        ck = iu.load_checkpoint(path)
        net.load_state_dict({k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("net.")}, strict=True)
        del ck
        return net.to(self.cfg.device).eval().requires_grad_(False)

    def load_models(self) -> None:
        """Load the encoders, their heads and (if found) the no-physics reference encoder."""
        import torch

        if self.models is not None:
            return
        if not self.models_available():
            raise FileNotFoundError("checkpoints or heads not found under " + self.cfg.path(self.cfg.run_dir))
        heads_dir = os.path.join(self.cfg.path(self.cfg.run_dir), "stageB", "heads")
        models = []
        for e, p in self.checkpoint_paths().items():
            net = self._load_net(p)
            hs = []
            for s in self.cfg.head_seeds:
                h = hm.HazardHeads(self.hazards, self.net_args["embed_dim"], self.net_args["img_size"], self.net_args["patch_size"])
                h.load_state_dict(torch.load(os.path.join(heads_dir, f"{e}_seed{s}.pt"), map_location=self.cfg.device))
                hs.append(h.to(self.cfg.device).eval())
            models.append((e, net, hs))
            logger.info("loaded %s with %d heads", e, len(hs))
        self.models = models
        ref = self.reference_checkpoint()
        self.reference = (self.cfg.reference_encoder, self._load_net(ref)) if ref else None

    # ---------------- inference ----------------
    def calibrate(self, prob: np.ndarray, source: str) -> np.ndarray:
        """Apply the validation-fitted Platt map per hazard to (..., K, H, W) probabilities."""
        out = np.empty(prob.shape, dtype=np.float64)
        for k, h in enumerate(self.hazards):
            a, b = self.platt[(source, h)]
            p = np.clip(prob[..., k, :, :].astype(np.float64), A.EPS, 1 - A.EPS)
            out[..., k, :, :] = 1.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + b)))
        return out

    def _run(self, net, x_norm):
        import torch

        with torch.no_grad():
            xb = torch.from_numpy(x_norm).float().to(self.cfg.device)
            lt = torch.full((len(xb),), self.lead_hours / 100.0, device=self.cfg.device)
            tokens = net.forward_encoder(xb, lt, tuple(self.variables))
            weather = net.unpatchify(net.head(tokens)).float().cpu().numpy()
        return tokens, weather

    def predict_live(self, x_phys: np.ndarray):
        """(N, V, H, W) physical inputs -> raw ClimaX hazard probabilities (N, K, H, W), weather forecast (N, V, H, W) physical."""
        import torch

        self.load_models()
        xn = ((x_phys - self.mu) / self.sd).astype(np.float32)
        probs, weathers = [], []
        for _, net, heads in self.models:
            tokens, w = self._run(net, xn)
            feats = tokens.half().float()      # the heads were trained on float16 tokens
            with torch.no_grad():
                probs.append(np.mean([torch.sigmoid(h(feats)).cpu().numpy() for h in heads], axis=0))
            weathers.append(w)
        return np.mean(probs, axis=0), np.mean(weathers, axis=0) * self.sd + self.mu

    def predict_reference_weather(self, x_phys: np.ndarray) -> Optional[np.ndarray]:
        self.load_models()
        if self.reference is None:
            return None
        _, w = self._run(self.reference[1], ((x_phys - self.mu) / self.sd).astype(np.float32))
        return w * self.sd + self.mu

    def _fields(self, arr):
        return {v: arr[self.variables.index(v)] for v in WEATHER_VARS}

    def forecast(self, issue_day, live: bool = True) -> Forecast:
        """Hazard + weather forecast issued at 00 UTC of ``issue_day`` for 3 days later."""
        d = np.datetime64(issue_day, "D")
        if d not in self._pred_index or d not in self._day_index:
            raise KeyError(f"{d} is not a test issue day (2015-2018 with a valid 3-day target)")
        k_pred, k_in = self._pred_index[d], self._day_index[d]
        tgt = d + np.timedelta64(self.cfg.lead_days, "D")
        x = self.x_phys[k_in:k_in + 1]
        truth = self._fields(self.x_phys[self._day_index[tgt]]) if tgt in self._day_index else None
        t0 = time.time()
        diff = None
        if live and self.models_available():
            raw, w = self.predict_live(x)
            prob = self.calibrate(raw, "ClimaX")[0]
            weather = self._fields(w[0])
            diff = float(np.nanmax(np.abs(prob - self.saved["test"][k_pred])))
            used_live = True
        else:
            prob, weather, used_live = self.saved["test"][k_pred], None, False
        obs = self.y["test"][k_pred]
        valid = np.isfinite(obs)
        alerts = np.stack([(prob[k] >= self.thresholds[h]) & valid[k] for k, h in enumerate(self.hazards)])
        return Forecast(d, tgt, self.hazards, prob, self.logistic["test"][k_pred], self.clim["test"][k_pred], obs, alerts,
                        dict(self.thresholds), self._fields(x[0]), weather, truth, used_live, time.time() - t0, diff)

    def alert_table(self, fc: Forecast) -> pd.DataFrame:
        rows = []
        for k, h in enumerate(fc.hazards):
            for i, j in zip(*np.nonzero(fc.alerts[k])):
                o = fc.observed[k, i, j]
                rows.append({"hazard": h, "lat": float(self.lat[i]), "lon": float(self.lon[j]), "probability": float(fc.prob[k, i, j]),
                             "threshold": fc.thresholds[h], "observed": "event" if o == 1 else "no event"})
        return pd.DataFrame(rows, columns=["hazard", "lat", "lon", "probability", "threshold", "observed"])

    def verification(self, fc: Forecast) -> pd.DataFrame:
        """Per hazard: hits, misses, false alarms of the alerts against the observed labels of the target day."""
        rows = []
        for k, h in enumerate(fc.hazards):
            v = np.isfinite(fc.observed[k])
            ev, al = (fc.observed[k] == 1) & v, fc.alerts[k]
            rows.append({"hazard": h, "observed event cells": int(ev.sum()), "alerts": int(al.sum()), "hits": int((al & ev).sum()),
                         "misses": int((~al & ev).sum()), "false alarms": int((al & ~ev & v).sum())})
        return pd.DataFrame(rows)

    def physics_errors(self, issue_day) -> Optional[pd.DataFrame]:
        """Per adjacent layer: grid-mean |R_pred - R_true| (m), one multi-layer encoder vs the reference encoder.

        Single encoder against single encoder (same seed), so ensemble averaging does not favour either side.
        """
        fc_day = np.datetime64(issue_day, "D")
        tgt = fc_day + np.timedelta64(self.cfg.lead_days, "D")
        if not self.models_available() or tgt not in self._day_index:
            return None
        self.load_models()
        x = self.x_phys[self._day_index[fc_day]][None]
        truth = self.x_phys[self._day_index[tgt]][None]
        name0, net0, _ = self.models[0]
        _, w0 = self._run(net0, ((x - self.mu) / self.sd).astype(np.float32))
        w_multi = w0 * self.sd + self.mu
        w_ref = self.predict_reference_weather(x)
        layers = list(ADJACENT_LAYERS)
        r_true = hypsometric_residuals(truth, self.variables, layers)
        rows = []
        for name, w in (("multi-layer physics", w_multi), ("no physics", w_ref)):
            if w is None:
                continue
            err = np.abs(hypsometric_residuals(w, self.variables, layers) - r_true).mean(axis=(0, 2, 3))
            rows += [{"model": name, "layer": f"{u}-{l} hPa", "abs_error_m": float(e)} for (u, l), e in zip(layers, err)]
        return pd.DataFrame(rows)

    def event(self, case: dict) -> pd.DataFrame:
        """Saved calibrated probabilities over a case window (same numbers as live inference)."""
        k = self.hazards.index(case["hazard"])
        cells = A.nearest_cells(self.lat, self.lon, case["points"])
        ser = A.event_series({"ClimaX": self.saved["test"][:, k], "logistic": self.logistic["test"][:, k],
                              "climatology": self.clim["test"][:, k]}, self.y["test"][:, k], self.target_days, cells,
                             case["start"], case["end"])
        ser["issue_day"] = ser["target_day"] - np.timedelta64(self.cfg.lead_days, "D")
        return ser


def _load_json(p):
    import json

    with open(p) as f:
        return json.load(f)
