# Hazard experiment, Option A: frozen encoders + hazard heads

Code: `src/climax_hazards/hazard_data.py`, `src/climax_hazards/hazard_model.py`, `scripts/mtech/run_hazard_experiment.py`.
Kaggle notebook: `notebooks/04_hazard_heads.ipynb`. Tests: `tests/mtech/test_hazard.py`.

## Research question
Does a ClimaX encoder trained with the hypsometric physics loss (λ = 0.001) yield better extreme-event detection than the same encoder trained without it (λ = 0)
when both feed identical hazard heads? And do either beat simple, honest baselines?

Option A uses the two existing internship-setup checkpoints, so **no new weather training** is needed. This was chosen because GPU budget is limited.
The corrected 6 × 6 grid and fine-tuning are deferred (see "Later").

## Task definition (my implementation choices)
- **Input:** the 17 ERA5 fields at **00 UTC on day D**, normalised with the train statistics. Original 6 × 5 grid, edge-padded in memory to the models' 6 × 6.
- **Target:** hazard label on the UTC day **D+3**, i.e. the day containing t + 72 h, the encoder's trained lead time (`lead_time = 0.72`).
- **Labels:** primary definitions from `HAZARD_DEFINITIONS.md`:
  - heatwave: Mar–Jun, train-period p90 Tmax, ≥ 3 days, land;
  - extreme wind: p99 of daily max wind speed;
  - extreme precipitation: IMD, p95 of wet days, boxes with ≥ 50 % coverage.
- **Samples:** one per day: train 11 680, val 1 460, test 1 460. Undefined targets (NaN) are excluded from every loss and metric.
- **The duplicated longitude column** gets NaN targets: it is never trained or scored.

## Model (tensor shapes for the internship encoder)
```
x (B, 17, 6, 6) --frozen ClimaX.forward_encoder (lead 72 h)--> tokens (B, 9, 1024)   [cached once]
tokens --head_k: Linear(1024, 256) → GELU → Dropout(0.1) → Linear(256, 4)--> (B, 9, 4)
       --unpatchify (same token order as ClimaX, unit-tested)--> logits_k (B, 1, 6, 6)
```
- There is one head per hazard (~0.26 M parameters each). The weather head is untouched, so weather forecasts are unchanged.
- **Loss:** per-hazard BCE-with-logits on valid cells, with the positive class weighted by the train negative/positive ratio (capped at 50), averaged over hazards.
- **Training:** AdamW (lr 1e-3, wd 1e-4), batch 256, ≤ 40 epochs, early stopping (patience 8) on mean validation PR-AUC. **3 seeds** per encoder.
- With a frozen encoder and disjoint heads, joint training equals separate training, so there is no negative transfer by construction.
  Multi-task interaction (and possible negative transfer) only arises when the encoder is fine-tuned.

## Baselines (identical samples, information available at issue time only)
| Baseline | Definition |
|---|---|
| climatology | train-period event frequency for the target calendar day (± 15 d), per cell |
| persistence | threshold exceedance on the latest fully observed day: D−1 (ERA5 heat, wind), D−2 (IMD rain, because of its 03 UTC day boundary) |
| logistic regression | per hazard and cell, on the same flattened normalised inputs; C ∈ {0.01, 0.1, 1}, chosen on val PR-AUC; class-balanced |

## Evaluation
- Pooled over valid cell-days of the **test** split (2015–2018).
- **PR-AUC** is the primary metric; its no-skill level equals the prevalence. Also reported: ROC-AUC, Brier score, and precision / recall / F1 with the confusion counts.
- The decision threshold maximises F1 on **val** and is applied unchanged to test.
- Heatwave prevalence rises from 6.5 % (val) to 9.2 % (test) because of the warming trend. Threshold-free PR-AUC is therefore the fairest comparison;
  F1 at the val threshold is sensitive to this shift.

## Status
- The pipeline was verified locally on CPU (real data, real IMD labels, synthetic encoders).
- First Kaggle run on 2026-10-11, repo commit `6d2e7b3`, 16.5 min: final-best checkpoints only (λ=0.001 `epoch_016`, λ=0 `epoch_016`).
  Results are in `docs/mtech/results/hazard_optionA/`.
- Second Kaggle run, 2026-10-11: all 8 checkpoints of both runs, plus a paired block bootstrap. Results are in `docs/mtech/results/hazard_optionA_allckpt/`.

## First result (test 2015–2018; thresholds from val; heads: mean ± sd over 3 seeds)

| Hazard (test prevalence) | climatology | persistence | logistic | encoder λ=0 + heads | encoder λ=0.001 + heads |
|---|---|---|---|---|---|
| extreme precipitation (1.8 %) | 0.068 | 0.022 | 0.155 | **0.224 ± 0.005** | 0.155 ± 0.005 |
| extreme wind (0.93 %) | 0.033 | 0.013 | **0.075** | 0.073 ± 0.006 | 0.056 ± 0.002 |
| heatwave (8.9 %) | 0.084 | 0.138 | 0.222 | **0.253 ± 0.017** | 0.222 ± 0.008 |

PR-AUC; the no-skill level equals the prevalence. Validation PR-AUC ranks the sources in the same order (λ=0 > λ=0.001 on all three hazards).

What it shows:
1. Learned models are far better than climatology and persistence. For precipitation, the λ=0 heads reach 12× the no-skill PR-AUC (ROC-AUC 0.93).
2. **The physics-trained encoder transfers worse to hazard detection than the data-only encoder**, on all three hazards (precipitation 0.155 vs 0.224).
   This happens even though its weather RMSE is ~2.7 % better. With physics features, the heads are only on par with logistic regression on the raw fields.
3. The heads behave differently on the two encoders. On the physics features they reach a much lower train loss (0.19–0.29 vs 0.50–0.55 at the best epoch) and peak later (epochs 14–22 vs 6–8),
   but generalise worse. This is consistent with a representation that is easier to fit but less hazard-relevant. It is a **hypothesis**, not established.
4. Heatwave climatology is uninformative (ROC-AUC 0.48) by construction: the calendar-day percentile threshold makes the base rate nearly constant through the season.

What it does not show yet:
- **One encoder per λ.** The ± only covers head seeds. The gap could come from the particular training run of each encoder rather than from λ.
  The all-checkpoints run checks stability within each run; only more encoder seeds (more compute) can settle it.
- **No confidence intervals yet** (added for the next run).
- Precipitation targets depend on the unverified IMD day convention (± 1 day). This applies equally to all sources.

## Limitations
- The 5.625° grid (~625 km) gives box-mean extremes only. 5 real longitude columns. Precipitation has only 9 IMD-covered boxes.
- One encoder per λ (seed 42). Head seeds measure head variance only, not encoder-training variance.
- The internship encoders were trained on the padded grid (duplicated column), and from scratch, not from pretrained ClimaX.

## Later (after Review 2 or with more compute)
Corrected 6 × 6 grid encoders; fine-tuning the encoder with `L_weather + λ L_hyps + Σ λ_k L_hazard_k` (where negative transfer becomes testable);
several encoder seeds; pretrained ClimaX initialisation with name-aware weight loading.

## Second result: all checkpoints + bootstrap confidence intervals

**Stability across checkpoints.** Test PR-AUC, min–max over the 4 checkpoints of each run (each the mean of 3 head seeds):

| Hazard | λ=0 run (e013, e015, e016, e019) | λ=0.001 run (e011, e016, e017, e019) |
|---|---|---|
| extreme precipitation | 0.220 – 0.224 | 0.155 – 0.172 |
| extreme wind | 0.071 – 0.077 | 0.051 – 0.056 |
| heatwave | 0.253 – 0.262 | 0.222 – 0.226 |

The ranges do not overlap for any hazard. The gap does not depend on which checkpoint is picked.

**Paired 7-day block bootstrap** (final-best checkpoints, test PR-AUC difference a − b, 95 % CI, 1000 resamples):

| Comparison | heatwave | extreme wind | extreme precipitation |
|---|---|---|---|
| λ=0.001 − λ=0 | −0.032 [−0.053, −0.009] | −0.021 [−0.039, −0.005] | −0.073 [−0.130, −0.019] |
| λ=0.001 − logistic | +0.003 [−0.019, 0.028] | −0.019 [−0.040, −0.002] | +0.001 [−0.044, 0.045] |
| λ=0 − logistic | +0.035 [0.011, 0.057] | +0.002 [−0.019, 0.020] | +0.074 [0.030, 0.121] |
| λ=0.001 − climatology | +0.141 [0.089, 0.194] | +0.023 [0.011, 0.039] | +0.088 [0.054, 0.125] |
| λ=0 − climatology | +0.173 [0.113, 0.225] | +0.044 [0.025, 0.066] | +0.161 [0.093, 0.238] |

**Conclusions within these two training runs:**
1. **The λ=0 encoder transfers better than the λ=0.001 encoder on all three hazards.** The CIs exclude zero, and the result holds for every checkpoint.
2. The λ=0 encoder adds skill over a linear model on the raw fields for heatwave and precipitation, but not for wind.
   The λ=0.001 encoder adds none, and is slightly worse than logistic for wind.
3. Both clearly beat climatology and persistence.

**Remaining caveat:** the bootstrap covers test-sample uncertainty, and the checkpoint sweep covers checkpoint choice. Neither covers **training-run randomness**:
each λ is still a single training run, and checkpoints of one run are strongly correlated. The decisive test is a second encoder seed per λ (≈ 9 GPU h for the pair).
Until then, write it as: "in our runs, physics regularisation improved weather RMSE (−2.7 % median) but reduced hazard transferability of the frozen representation."

## Next experiment: joint fine-tuning with adaptive loss balancing (notebook 05)

Code: `src/climax_hazards/joint.py`, `scripts/mtech/train_joint.py`, `notebooks/05_joint_adaptive_training.ipynb`; tests `tests/mtech/test_joint.py`.

- **Grid:** the corrected 6 × 6 dataset (lon 67.5–95.625°E, no duplicated column).
- **Model:** the ClimaX encoder is **fine-tuned**, not frozen. The existing weather head and the three hazard heads share the encoder tokens.
- **Five tasks:**

  | Task | Loss |
  |---|---|
  | weather | lat-weighted MSE in normalised units |
  | physics | hypsometric residual matching, 500–850 hPa, divided by (18 m)² |
  | heatwave, extreme wind, extreme rainfall | masked, class-weighted BCE on the D+3 labels |

- **Adaptive balancing:** uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018), `L = Σ exp(−s_i)·L_i + s_i`, with learnable `s_i` clamped to ±6.
  At the optimum `exp(−s_i) = 1/L_i`, so tasks contribute on comparable scales without a hand-set λ.
  A `fixed` mode keeps the internship weights (λ = 0.001 on raw m²) for ablation.
- **Initialisation:** the λ = 0 `epoch_016` checkpoint, which transferred best to hazards in Option A. The λ = 0.001 checkpoint and training from scratch are also supported.
  The checkpoint was trained on the padded grid, so its positional embeddings are re-learned during fine-tuning (the first column now is 67.5°E).
- **Samples:** every hour t with t + 72 h in the same split (≈280 k training pairs), drawn at random, 65,536 per epoch.
  Hazard targets are the labels of the UTC day containing t + 72 h.
- **Optimiser:** AdamW, with lr 1e-4 (encoder + weather head), 5e-4 (hazard heads) and 1e-2 (task weights).
  500 warm-up steps, then cosine decay, gradient clipping 1.0, fp16 mixed precision on GPU.
- **Selection:** best mean validation PR-AUC (00 UTC samples), with early stopping (patience 4). Weather validation RMSE is logged every epoch.
- **Evaluation:** test hazards on exactly the same 00 UTC samples as the frozen baseline on the corrected grid (heads on frozen λ = 0 / λ = 0.001 encoders, logistic, climatology, persistence).
  It uses the paired 7-day block bootstrap. Weather is scored on the full test set against persistence and climatology, including the hypsometric-consistency score.
- **Status:** verified locally (CPU smoke runs with real data and labels, synthetic checkpoints, 32 unit tests). **No results yet.**
