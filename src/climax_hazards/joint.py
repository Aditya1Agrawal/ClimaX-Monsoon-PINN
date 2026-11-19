"""Joint weather + hazard training with adaptive loss balancing.

One ClimaX encoder is fine-tuned for five tasks at once:

    weather   lat-weighted MSE of the 72 h forecast (normalised units, as in the internship)
    physics   hypsometric-residual matching, 500-850 hPa (metres, divided by a scale)
    heatwave / extreme_wind / extreme_precip   masked, class-weighted BCE of the D+3 labels

Adaptive balancing uses homoscedastic uncertainty weighting (Kendall, Gal & Cipolla,
CVPR 2018): every task i has a learnable s_i = log(sigma_i^2) and

    L_total = sum_i  exp(-s_i) * L_i  +  s_i

At the optimum exp(-s_i) = 1 / L_i, so each task contributes on a comparable scale and
no hand-tuned lambda is needed; s_i is clamped to [-clamp, clamp] for stability.
A "fixed" mode (hand-set weights, e.g. the internship's lambda = 0.001) is kept for ablation.

Samples: every hour t whose t + lead target lies in the same split. Hazard targets are the
labels of the UTC day containing t + lead (NaN where undefined), exactly as in
``labels.align_to_forecast``. Validation/test hazard scores use the 00 UTC samples so they
are comparable with the frozen-encoder experiment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from climax_hazards import labels as L
from climax_hazards.forecast_eval import G, R_D
from climax_hazards.hazard_model import HazardHeads
from climax_hazards.inspect_utils import SHARD_RE, shard_files, shard_timestamps

TASKS_WEATHER = ("weather", "physics")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


class SplitData:
    """One split held in memory as normalised hourly fields, plus forecast pairs and hazard targets."""

    def __init__(self, data_dir, split: str, variables: Sequence[str], mean: np.ndarray, std: np.ndarray,
                 lead: int, labelsets: Optional[Dict[str, L.LabelSet]] = None, dtype=np.float32):
        xs, ts = [], []
        for f in shard_files(data_dir, split):
            y, k = map(int, SHARD_RE.match(f.name).groups())
            with np.load(f) as d:
                arr = np.concatenate([d[v] for v in variables], axis=1).astype(np.float32)
            xs.append(((arr - mean) / std).astype(dtype))
            ts.append(shard_timestamps(y, k, arr.shape[0]).astype("datetime64[h]"))
        self.times = np.concatenate(ts)
        order = np.argsort(self.times)
        self.times = self.times[order]
        self.x = np.concatenate(xs)[order]
        th = self.times.astype("int64")
        j = np.searchsorted(th, th + lead)
        ok = j < len(th)
        ok[ok] = th[j[ok]] == th[ok] + lead
        self.src = np.nonzero(ok)[0]           # input index
        self.dst = j[ok]                       # target index (t + lead)
        self.lead = lead
        self.hazards = list(labelsets) if labelsets else []
        if labelsets:
            self.haz = np.stack([L.align_to_forecast(self.times[self.src], lead, ls.splits[split]["dates"], ls.splits[split]["label"])
                                 for ls in labelsets.values()], axis=1).astype(np.float32)
        else:
            self.haz = None

    def __len__(self):
        return len(self.src)

    def batch(self, idx: np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(self.x[self.src[idx]], dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(self.x[self.dst[idx]], dtype=np.float32))
        h = torch.from_numpy(self.haz[idx]) if self.haz is not None else None
        return x, y, h

    def issue_hour_indices(self, hours=(0,)) -> np.ndarray:
        """Pair indices whose input time is at one of ``hours`` UTC."""
        t = self.times[self.src]
        hr = (t - t.astype("datetime64[D]")).astype(int)
        return np.nonzero(np.isin(hr, hours))[0]


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


class JointModel(nn.Module):
    """ClimaX encoder shared by the existing weather head and one head per hazard."""

    def __init__(self, net, hazards: Sequence[str], hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = net
        self.hazards = list(hazards)
        self.heads = HazardHeads(self.hazards, net.pos_embed.shape[-1], net.img_size, net.patch_size, hidden, dropout)

    def forward(self, x, lead_times, variables):
        tokens = self.net.forward_encoder(x, lead_times, tuple(variables))   # (B, L, D)
        weather = self.net.unpatchify(self.net.head(tokens))                 # (B, V, H, W), default_vars order
        hazards = self.heads(tokens) if self.hazards else None               # (B, K, H, W) logits
        return weather, hazards


class UncertaintyWeighting(nn.Module):
    """Kendall et al. (2018) homoscedastic uncertainty weighting, with optional per-task weight caps.

    The optimum weight is exp(-s) = 1 / L, so a task whose *training* loss keeps falling
    (e.g. an overfitting hazard head) gets an ever larger weight. ``max_weight`` caps
    exp(-s) for chosen tasks (lower bound s >= -log(cap)) to stop that feedback loop.
    """

    def __init__(self, tasks: Sequence[str], clamp: float = 6.0, max_weight: Optional[Dict[str, float]] = None):
        super().__init__()
        self.tasks = list(tasks)
        self.log_vars = nn.Parameter(torch.zeros(len(self.tasks)))
        self.clamp = clamp
        lo = [-clamp] * len(self.tasks)
        for i, t in enumerate(self.tasks):
            if max_weight and t in max_weight and max_weight[t]:
                lo[i] = max(lo[i], -float(np.log(max_weight[t])))
        self.register_buffer("lower", torch.tensor(lo, dtype=torch.float32))

    def forward(self, losses: Dict[str, torch.Tensor]):
        s = torch.maximum(self.log_vars, self.lower).clamp(max=self.clamp)
        total = 0.0
        for i, t in enumerate(self.tasks):
            total = total + torch.exp(-s[i]) * losses[t] + s[i]
        return total, {t: float(torch.exp(-s[i])) for i, t in enumerate(self.tasks)}


class GradNormWeighting(nn.Module):
    """GradNorm (Chen et al., ICML 2018): balance per-task gradient norms on a shared layer.

    Weights w_i (sum = number of tasks) are updated so that the gradient norm of each
    weighted task loss on a shared parameter W approaches mean_norm * r_i**alpha, where
    r_i is the task's relative inverse training rate (L_i / L_i(0), normalised). Unlike
    uncertainty weighting it does not give more weight to a task just because its training
    loss is small. The target is treated as a constant, as in the paper, so the gradient for
    w_i is sign(w_i g_i - target_i) * g_i with g_i = ||dL_i/dW||.
    """

    def __init__(self, tasks: Sequence[str], alpha: float = 1.5, min_weight: float = 0.05):
        super().__init__()
        self.tasks = list(tasks)
        self.w = nn.Parameter(torch.ones(len(self.tasks)))
        self.alpha = alpha
        self.min_weight = min_weight
        self.register_buffer("initial", torch.zeros(len(self.tasks)))
        self.register_buffer("has_initial", torch.zeros((), dtype=torch.bool))

    def forward(self, losses: Dict[str, torch.Tensor]):
        w = self.w.detach()
        total = sum(w[i] * losses[t] for i, t in enumerate(self.tasks))
        return total, {t: float(w[i]) for i, t in enumerate(self.tasks)}

    def update(self, losses: Dict[str, torch.Tensor], shared_param: torch.nn.Parameter, optimizer: torch.optim.Optimizer) -> bool:
        """One GradNorm step on the weights; returns False (and does nothing) if any norm is non-finite."""
        lv = torch.stack([losses[t].detach().float() for t in self.tasks])
        if not torch.isfinite(lv).all():
            return False
        if not bool(self.has_initial):
            self.initial.copy_(lv.clamp_min(1e-8))
            self.has_initial.fill_(True)
        g = []
        for t in self.tasks:
            gr = torch.autograd.grad(losses[t], shared_param, retain_graph=True, allow_unused=True)[0]
            g.append(torch.zeros((), device=lv.device) if gr is None else gr.float().norm())
        g = torch.stack(g)
        if not torch.isfinite(g).all():
            return False
        w = self.w.detach()
        gw = w * g
        ratio = lv / self.initial
        r = ratio / ratio.mean()
        target = gw.mean() * r ** self.alpha
        optimizer.zero_grad(set_to_none=True)
        self.w.grad = torch.sign(gw - target) * g
        optimizer.step()
        with torch.no_grad():  # keep weights positive and sum = number of tasks
            self.w.clamp_(min=self.min_weight)
            self.w.mul_(len(self.tasks) / self.w.sum())
        return True


class FixedWeighting(nn.Module):
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = dict(weights)
        self.tasks = list(weights)

    def forward(self, losses):
        total = sum(self.weights[t] * losses[t] for t in self.tasks)
        return total, dict(self.weights)


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------


def lat_weights(lat: np.ndarray, device) -> torch.Tensor:
    w = np.cos(np.deg2rad(lat))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)[None, None, :, None]


def weather_loss(pred, target, w_lat):
    return (((pred - target) ** 2) * w_lat).mean()


class HypsometricResidual(nn.Module):
    """Residual-matching hypsometric loss on denormalised fields (same formula as the internship loss)."""

    def __init__(self, variables: Sequence[str], mean: np.ndarray, std: np.ndarray, upper=500, lower=850, scale_m: float = 18.0):
        super().__init__()
        v = list(variables)
        self.idx = [v.index(f"geopotential_{upper}"), v.index(f"geopotential_{lower}"),
                    v.index(f"temperature_{upper}"), v.index(f"temperature_{lower}")]
        m = torch.tensor(mean.reshape(-1)[self.idx], dtype=torch.float32)
        s = torch.tensor(std.reshape(-1)[self.idx], dtype=torch.float32)
        self.register_buffer("mean", m[None, :, None, None])
        self.register_buffer("std", s[None, :, None, None])
        self.coef = (R_D / G) * float(np.log(lower / upper))
        self.scale2 = scale_m ** 2  # divide m^2 by a typical residual spread so the loss is O(1)

    def residual(self, x_norm):
        z_u, z_l, t_u, t_l = (x_norm[:, self.idx] * self.std + self.mean).unbind(1)
        return (z_u - z_l) / G - self.coef * 0.5 * (t_u + t_l)

    def forward(self, pred_norm, target_norm):
        return F.mse_loss(self.residual(pred_norm), self.residual(target_norm)) / self.scale2


def hazard_losses(logits, y, pos_weight) -> List[torch.Tensor]:
    """Per-hazard masked BCE; a hazard with no valid cell in the batch contributes 0 (no gradient)."""
    out = []
    valid = torch.isfinite(y)
    for k in range(y.shape[1]):
        m = valid[:, k]
        if m.any():
            out.append(F.binary_cross_entropy_with_logits(logits[:, k][m], y[:, k][m], pos_weight=pos_weight[k]))
        else:
            out.append(logits.sum() * 0.0)
    return out


def load_climax_state(net, ckpt_state: Dict) -> None:
    """Strict load of the ``net.`` part of a Lightning state dict (fails loudly on any mismatch)."""
    sd = {k[len("net."):]: v for k, v in ckpt_state.items() if k.startswith("net.")}
    net.load_state_dict(sd, strict=True)
