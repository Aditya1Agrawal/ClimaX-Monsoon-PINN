"""Hazard heads on a frozen ClimaX encoder, plus metrics (Option A).

Architecture (shapes for the internship model: 9 tokens, D = 1024, patch 2, grid 6 x 6):

    x (B, 17, 6, 6) --frozen ClimaX.forward_encoder--> tokens (B, 9, 1024)
    tokens --head_k: Linear(1024, h) -> GELU -> Dropout -> Linear(h, p*p)--> (B, 9, 4)
           --unpatchify--> logits_k (B, 1, 6, 6)          one head per hazard k

The existing weather head is untouched (the encoder is frozen), so weather forecasts are
unchanged. Because the encoder is frozen and the heads share no parameters, training the
heads jointly with a summed loss is equivalent to training them separately; multi-task
interaction would only appear when the encoder is fine-tuned (a later step).
Loss: per-hazard BCE-with-logits on valid (non-NaN) cells, positive class weighted by
the train-set negative/positive ratio (capped).
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HazardHeads(nn.Module):
    def __init__(self, hazards: Sequence[str], embed_dim: int, img_size, patch_size: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hazards = list(hazards)
        self.img_size, self.p = tuple(img_size), patch_size
        self.heads = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, patch_size**2))
            for k in self.hazards
        })

    def unpatchify(self, x):  # (B, L, p*p) -> (B, H, W), same token order as ClimaX
        h, w, p = self.img_size[0] // self.p, self.img_size[1] // self.p, self.p
        x = x.reshape(x.shape[0], h, w, p, p)
        return torch.einsum("nhwpq->nhpwq", x).reshape(x.shape[0], h * p, w * p)

    def forward(self, tokens):  # (B, L, D) -> (B, K, H, W) logits
        return torch.stack([self.unpatchify(self.heads[k](tokens)) for k in self.hazards], dim=1)


@torch.no_grad()
def extract_features(net, x_norm: np.ndarray, variables, lead_hours: float, device="cpu", batch_size=128) -> np.ndarray:
    """Frozen encoder tokens (N, L, D) as float16."""
    net = net.to(device).eval()
    out = []
    for i in range(0, len(x_norm), batch_size):
        xb = torch.from_numpy(x_norm[i : i + batch_size]).float().to(device)
        lt = torch.full((len(xb),), lead_hours / 100.0, device=device)  # ClimaX lead-time encoding
        out.append(net.forward_encoder(xb, lt, tuple(variables)).half().cpu().numpy())
    return np.concatenate(out)


def masked_bce(logits, y, pos_weight):
    """Mean BCE over valid cells, per hazard, then averaged over hazards. y has NaN = ignore."""
    valid = torch.isfinite(y)
    losses = []
    for k in range(y.shape[1]):
        m = valid[:, k]
        if m.any():
            losses.append(F.binary_cross_entropy_with_logits(logits[:, k][m], y[:, k][m], pos_weight=pos_weight[k]))
    return torch.stack(losses).mean()


def pos_weights(y_train: np.ndarray, cap: float = 50.0) -> np.ndarray:
    w = []
    for k in range(y_train.shape[1]):
        v = y_train[:, k][np.isfinite(y_train[:, k])]
        p = v.mean() if v.size else 0.5
        w.append(min(cap, (1 - p) / max(p, 1e-6)))
    return np.array(w, dtype=np.float32)


def predict(heads, feats: np.ndarray, device="cpu", batch_size=512) -> np.ndarray:
    heads = heads.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            out.append(torch.sigmoid(heads(torch.from_numpy(feats[i : i + batch_size]).float().to(device))).cpu().numpy())
    return np.concatenate(out)


def train_heads(feats_tr, y_tr, feats_va, y_va, hazards, img_size, patch_size, seed=0, epochs=40, lr=1e-3,
                weight_decay=1e-4, batch_size=256, patience=8, hidden=256, dropout=0.1, device="cpu", log=print):
    """Train on frozen features; keep the epoch with the best mean validation PR-AUC."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    heads = HazardHeads(hazards, feats_tr.shape[-1], img_size, patch_size, hidden, dropout).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.from_numpy(pos_weights(y_tr)).to(device)
    ytr = torch.from_numpy(y_tr)
    best, best_score, bad, history = None, -np.inf, 0, []
    for ep in range(epochs):
        heads.train()
        perm = rng.permutation(len(feats_tr))
        tot = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb = torch.from_numpy(feats_tr[idx]).float().to(device)
            loss = masked_bce(heads(xb), ytr[idx].to(device), pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        prob = predict(heads, feats_va, device)
        aps = [average_precision(prob[:, k], y_va[:, k]) for k in range(len(hazards))]
        score = float(np.nanmean(aps))
        history.append({"epoch": ep, "train_loss": tot / len(perm), "val_ap_mean": score, **{f"val_ap_{h}": a for h, a in zip(hazards, aps)}})
        if score > best_score:
            best, best_score, bad = copy.deepcopy(heads.state_dict()), score, 0
        else:
            bad += 1
        if log:
            log(f"  epoch {ep:02d} loss {tot / len(perm):.4f} val AP {score:.4f}")
        if bad >= patience:
            break
    heads.load_state_dict(best)
    return heads, history


# ----------------------------------------------------------------------------
# Metrics (pooled over valid cell-days)
# ----------------------------------------------------------------------------


def _flat(prob, y):
    m = np.isfinite(y)
    return prob[m].astype(np.float64), y[m].astype(np.int64)


def average_precision(prob, y) -> float:
    from sklearn.metrics import average_precision_score

    p, t = _flat(prob, y)
    return float(average_precision_score(t, p)) if t.size and 0 < t.sum() < t.size else float("nan")


def best_f1_threshold(prob, y) -> float:
    from sklearn.metrics import precision_recall_curve

    p, t = _flat(prob, y)
    if not t.size or t.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(t, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if thr.size else 0.5


def scores(prob, y, threshold: float) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    p, t = _flat(prob, y)
    pred = p >= threshold
    tp, fp = int((pred & (t == 1)).sum()), int((pred & (t == 0)).sum())
    fn, tn = int((~pred & (t == 1)).sum()), int((~pred & (t == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    both = t.size and 0 < t.sum() < t.size
    return {
        "n": int(t.size), "prevalence": float(t.mean()) if t.size else float("nan"),
        "pr_auc": average_precision(prob, y), "roc_auc": float(roc_auc_score(t, p)) if both else float("nan"),
        "threshold": threshold, "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if tp else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "brier": float(np.mean((p - t) ** 2)) if t.size else float("nan"),
    }


def logistic_baseline(x_tr, y_tr, x_va, y_va, x_other: Dict[str, np.ndarray], Cs=(0.01, 0.1, 1.0), log=print):
    """Per-hazard, per-cell logistic regression on the flattened normalised input fields.

    The same inputs the encoder sees; C is chosen per hazard by pooled validation PR-AUC.
    Cells without both classes in train predict the train prevalence.
    Returns probabilities for 'val' and each key of ``x_other``: {split: (N, K, H, W)}.
    """
    from sklearn.linear_model import LogisticRegression

    flat = lambda a: a.reshape(len(a), -1)
    ftr, fva = flat(x_tr), flat(x_va)
    fo = {k: flat(v) for k, v in x_other.items()}
    K, H, W = y_tr.shape[1:]
    out = {s: np.zeros((len(v),) + (K, H, W), dtype=np.float32) for s, v in {"val": fva, **fo}.items()}
    for k in range(K):
        best = None
        for C in Cs:
            probs = {s: np.zeros((len(v), H, W), dtype=np.float32) for s, v in {"val": fva, **fo}.items()}
            for i in range(H):
                for j in range(W):
                    m = np.isfinite(y_tr[:, k, i, j])
                    t = y_tr[m, k, i, j]
                    if t.size == 0:
                        continue
                    if t.min() == t.max():
                        for s in probs:
                            probs[s][:, i, j] = t.mean()
                        continue
                    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(ftr[m], t)
                    for s, f in {"val": fva, **fo}.items():
                        probs[s][:, i, j] = clf.predict_proba(f)[:, 1]
            ap = average_precision(probs["val"], y_va[:, k])
            if log:
                log(f"  logistic hazard {k} C={C}: val PR-AUC {ap:.4f}")
            if best is None or ap > best[0]:
                best = (ap, C, probs)
        for s in out:
            out[s][:, k] = best[2][s]
    return out
