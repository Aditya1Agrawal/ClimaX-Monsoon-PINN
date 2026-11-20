"""Joint weather + hazard training with adaptive loss balancing.

One ClimaX encoder is fine-tuned for five tasks at once:

    weather   lat-weighted MSE of the 72 h forecast (normalised units, as in the internship)
    physics   hypsometric-residual matching, 500-850 hPa or every adjacent layer (metres, divided by a scale)
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


SINGLE_LAYER = ((500, 850),)
# every adjacent pair of the 7 pressure levels (50 ... 925 hPa): the whole column is constrained
ADJACENT_LAYERS = ((850, 925), (700, 850), (600, 700), (500, 600), (250, 500), (50, 250))


def parse_layers(spec: str):
    """'single' | 'adjacent' | '500-850,700-850' -> tuple of (upper, lower) pressure pairs in hPa."""
    if spec == "single":
        return SINGLE_LAYER
    if spec == "adjacent":
        return ADJACENT_LAYERS
    out = []
    for part in spec.split(","):
        u, l = sorted(int(p) for p in part.split("-"))
        if u == l:
            raise ValueError(f"empty layer {part!r}")
        out.append((u, l))
    return tuple(out)


class HypsometricResidual(nn.Module):
    """Residual-matching hypsometric loss on denormalised fields, for one or several layers.

    For each layer (p_u < p_l) the residual is R = (Phi_u - Phi_l)/g - (R_d/g) * (T_u + T_l)/2 * ln(p_l/p_u)
    in metres (the internship formula). The loss matches the predicted residual to the target's,
    layer by layer, and averages  MSE_k / scale_k**2  over layers, so each layer contributes O(1)
    and thin, low-error layers are not drowned out by thick ones. With the default single
    500-850 hPa layer and scale 18 m it is identical to the v2 loss.

    Note: summing the adjacent residuals from 850 to 500 hPa gives the 500-850 thickness residual
    with a log-pressure trapezoidal layer-mean temperature over 500/600/700/850 hPa, so the adjacent
    set constrains the thick layer too, with a better mean temperature than the two-level average.
    """

    def __init__(self, variables: Sequence[str], mean: np.ndarray, std: np.ndarray,
                 layers: Sequence[Sequence[int]] = SINGLE_LAYER, scale_m=18.0):
        super().__init__()
        v = list(variables)
        self.layers = [tuple(int(p) for p in lay) for lay in layers]
        if not self.layers:
            raise ValueError("no layers")
        idx, coef = [], []
        for u, l in self.layers:
            if not u < l:
                raise ValueError(f"layer {(u, l)} must be (upper, lower) with upper < lower pressure")
            idx.append([v.index(f"geopotential_{u}"), v.index(f"geopotential_{l}"),
                        v.index(f"temperature_{u}"), v.index(f"temperature_{l}")])
            coef.append((R_D / G) * float(np.log(l / u)))
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))           # (K, 4)
        m, s = mean.reshape(-1), std.reshape(-1)
        self.register_buffer("mean", torch.tensor(m[idx], dtype=torch.float32))     # (K, 4)
        self.register_buffer("std", torch.tensor(s[idx], dtype=torch.float32))
        self.register_buffer("coef", torch.tensor(coef, dtype=torch.float32))
        self.register_buffer("scale2", torch.ones(len(self.layers)))
        self.set_scales(scale_m)

    def set_scales(self, scale_m) -> None:
        """One scale (metres) for all layers or one per layer; the loss divides each layer's MSE by scale**2."""
        s = np.broadcast_to(np.asarray(scale_m, dtype=np.float64), (len(self.layers),))
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError(f"scales must be positive, got {s}")
        self.scale2.copy_(torch.tensor(s ** 2, dtype=torch.float32))

    def residual(self, x_norm):
        """(B, V, H, W) normalised fields -> (B, K, H, W) residuals in metres."""
        f = x_norm[:, self.idx] * self.std[None, :, :, None, None] + self.mean[None, :, :, None, None]  # (B, K, 4, H, W)
        z_u, z_l, t_u, t_l = f.unbind(2)
        return (z_u - z_l) / G - self.coef[None, :, None, None] * 0.5 * (t_u + t_l)

    def layer_mse(self, pred_norm, target_norm):
        """(K,) mean squared residual mismatch per layer, in m^2."""
        return ((self.residual(pred_norm) - self.residual(target_norm)) ** 2).mean(dim=(0, 2, 3))

    def forward(self, pred_norm, target_norm):
        return (self.layer_mse(pred_norm, target_norm) / self.scale2).mean()


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
