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

## Result: joint fine-tuning v1 (Kaggle 2×T4, 2026-11-11, commit `93fb1d3`)

Files: `docs/mtech/results/joint_uw_v1/` (joint run) and `docs/mtech/results/frozen_corrected6x6/` (frozen baseline, same grid and samples).
The joint run initialised from λ=0 `epoch_016`, used uncertainty weighting, and trained for 1.3 min/epoch.
Early stopping came at epoch 6, the best epoch was 2, and the whole run took 12.7 min.

**Test PR-AUC on the corrected 6×6 grid** (frozen heads: mean of 3 seeds):

| Hazard (prevalence) | climatology | persistence | logistic | frozen λ=0 | frozen λ=0.001 | **joint** |
|---|---|---|---|---|---|---|
| heatwave (9.2 %) | 0.089 | 0.141 | 0.235 | 0.232 | 0.224 | **0.254** |
| extreme wind (0.90 %) | 0.033 | 0.012 | **0.073** | 0.059 | 0.049 | 0.052 |
| extreme rainfall (1.8 %) | 0.068 | 0.022 | 0.165 | 0.174 | **0.194** | 0.172 |

**Paired 7-day block bootstrap (95 % CI), joint − other:**

| vs | heatwave | extreme wind | extreme rainfall |
|---|---|---|---|
| frozen λ=0 (same init) | +0.018 [−0.005, 0.040] | −0.008 [−0.018, 0.002] | −0.006 [−0.047, 0.033] |
| frozen λ=0.001 | **+0.027 [0.003, 0.051]** | +0.003 [−0.005, 0.013] | −0.027 [−0.085, 0.028] |
| logistic | +0.019 [−0.007, 0.045] | **−0.021 [−0.045, −0.004]** | +0.007 [−0.035, 0.050] |
| climatology / persistence | all better, CIs exclude 0 | all better | all better |

**Frozen λ=0 − frozen λ=0.001 on the corrected grid:**
heat +0.009 [−0.018, 0.037]; wind +0.010 [−0.002, 0.024]; rain −0.022 [−0.080, 0.037]. **All CIs include 0.**

**Weather (test RMSE vs climatology, joint model):**
- Better aloft: z50 +22.9 %, z250 +17.8 %, z500 +9.8 %, t250 +14.6 %.
- **Worse near the surface:** t2m −4.7 %, u10 −4.2 %, v10 −3.7 %, z850 −2.2 %, z925 −2.4 %.
- Hypsometric error: 500–850 hPa equals climatology (11.0 m). The untrained low layers are much worse: 700–850 hPa 4.5 m vs 3.0 m; 850–925 hPa 4.3 m vs 1.5 m.

**Learned task weights** at the best epoch: weather 11.0, physics 4.1, heat 1.7, wind 2.2, rain 2.5. They tracked ≈ 1/training loss, as designed.

**Inference:**
1. Joint fine-tuning **did not significantly beat frozen heads from the same initialisation** on any hazard. Heat is borderline (+0.018, p ≈ 0.07). It does significantly beat frozen λ=0.001 on heat.
2. **Hazard overfitting, amplified by the weighting.** Training BCE fell about 70 % in 6 epochs while validation PR-AUC peaked at epoch 2 (heat) or was still rising (rain).
   Uncertainty weighting increases a task's weight as its training loss falls (exp(−s) ≈ 1/L), so the weights of the overfitting hazard tasks kept growing (heat 1.1 → 3.4).
   Hourly samples share each day's label (24 near-duplicates per day), so the effective sample size is far smaller than 280 k.
3. **Near-surface weather degraded, falling below climatology.** Selection by mean hazard PR-AUC stopped at epoch 2, while weather was still re-adapting from the padded-grid initialisation (validation t2m RMSE 3.13 → 2.40 over epochs 0–6).
   Low-level hypsometric consistency also worsened. The selection criterion ignored weather.
4. **The earlier "physics encoder transfers worse" finding does not reproduce on the corrected grid**: all frozen λ=0 vs λ=0.001 CIs include 0.
   Both encoders were trained on the padded grid and are applied here with a one-column shift, so this frozen comparison is confounded. The physics question needs encoders trained on the corrected grid.
5. Logistic regression stays a strong baseline. No variant significantly beats it except on heat, and it is the best method for wind.
6. One training epoch (5) logged NaN losses (an fp16 overflow). The gradient scaler skipped the bad steps and training recovered, but a guard and counter are needed.

## Joint training v2: staged design (notebook 06)

Each v1 cause gets a targeted fix. Whether the fixes work must be shown by the Kaggle run; no v2 results yet.

| v1 cause | v2 fix | Where |
|---|---|---|
| Encoder trained on the padded grid (1-column shift) | **Stage A**: fine-tune the λ=0 checkpoint on the corrected grid for weather ± physics only. 3 seeds per variant; select on validation weather; export `best_encoder.ckpt` | `train_joint.py --no_hazards --export_encoder` |
| Physics vs no-physics comparison confounded by the grid | **Stage B**: frozen hazard heads on the 6 Stage-A encoders. Paired bootstrap physics − no-physics per seed; mean ± sd over seeds | `run_hazard_experiment.py` |
| Uncertainty weighting rewarded overfitting hazards (exp(−s) ≈ 1/L) | Hazard weights capped (`--max_hazard_weight 2`, enforced as s ≥ −log 2). **GradNorm** (Chen et al., ICML 2018) as a second balancer: it equalises gradient norms on the last shared encoder layer and does not reward a small training loss | `joint.UncertaintyWeighting(max_weight=…)`, `joint.GradNormWeighting` |
| 24 near-duplicate hourly samples per label | Hazard losses use a separate batch of 64 samples drawn from the 00 UTC pool (one per day). Weather and physics losses still use all hours | `--hazard_hours 0 --hazard_batch 64` |
| Fast hazard overfitting | Encoder lr 3e-5 (was 1e-4), head dropout 0.3 (was 0.1), head weight decay 1e-3 (was 1e-4), patience 5 | Stage C arguments |
| Selection ignored weather | Best mean validation PR-AUC **only among epochs whose validation weather MSE is within 1 % of the starting model**; otherwise fall back to the best-weather epoch. Early stopping counts epochs without either improvement | `--weather_tolerance 0.01` |
| fp16 NaN batch | Non-finite batches are skipped, never used for updates or logs, and counted (`skipped_batches`) | training loop |

- **Stage C**: joint training from the Stage-A physics encoder with the lowest validation weather MSE. It runs once with capped uncertainty weighting and once with GradNorm.
- **Comparison**: on identical test samples, against frozen heads on the *same* encoder, on the matching no-physics encoder, and against logistic, climatology and persistence (paired 7-day block bootstrap).
- **Weather**: reported relative to the Stage-A starting encoder.
- **Caveat**: Stage-A seeds vary the fine-tuning (data order, dropout) from one shared initial checkpoint. They are not independent from-scratch trainings.

## Result: joint training v2 (Kaggle 2×T4, 2026-11-20, commit `24fe0e6`)

Files: `docs/mtech/results/joint_v2/`. Stage A: 6 runs of 11–19 min. Stage C: 20–23 min per run. One NaN batch was skipped in total.

### Stage A: encoder adapted to the corrected grid (test RMSE, mean ± sd over 3 seeds)

| Variable | no physics | physics | climatology | v1 joint (for reference) |
|---|---|---|---|---|
| t2m (K) | **2.209 ± 0.002** | 2.239 ± 0.003 | 2.263 | 2.369 |
| u10 (m/s) | **1.790 ± 0.001** | 1.803 ± 0.003 | 1.804 | 1.880 |
| t850 (K) | **1.715 ± 0.001** | 1.727 ± 0.001 | 1.825 | 1.789 |
| z500 (m²/s²) | 251.9 ± 0.2 | **251.3 ± 0.3** | 280.0 | 252.6 |
| z850 (m²/s²) | **226.8 ± 0.3** | 227.2 ± 0.3 | 227.5 | 232.5 |
| hyps. error 500–850 hPa (m), trained layer | 10.98 | 10.96 | 10.98 | 10.98 |
| hyps. error 700–850 hPa (m) | **3.01** | 4.10 ± 0.01 | 3.01 | 4.49 |
| hyps. error 850–925 hPa (m) | **1.53** | 3.64 ± 0.03 | 1.52 | 4.34 |

- Grid adaptation fixed the v1 near-surface degradation: both variants now match or beat climatology everywhere.
- **The physics loss did not help weather.** It was slightly worse near the surface (t2m +1.4 %, consistent across seeds) and marginally better at z500 (−0.2 %).
- **It did not improve consistency on its own layer** (10.96 vs 10.98 m), and it **made the untrained lower layers markedly less consistent** (700–850 hPa +36 %, 850–925 hPa ×2.4), reproducibly across seeds.
  A single-layer constraint with a two-level mean temperature appears to move the inconsistency into neighbouring layers.

### Stage B: frozen heads on the Stage-A encoders (test PR-AUC, mean ± sd over 3 encoder seeds)

| Hazard | no physics | physics | logistic | climatology |
|---|---|---|---|---|
| heatwave | 0.244 ± 0.001 | 0.248 ± 0.002 | 0.235 | 0.089 |
| extreme wind | 0.060 ± 0.000 | 0.060 ± 0.001 | **0.074** | 0.033 |
| extreme rainfall | **0.224 ± 0.004** | 0.219 ± 0.004 | 0.165 | 0.069 |

- **Physics vs no physics (paired bootstrap per seed):**
  - heat +0.005 [0.000, 0.009] (s0), +0.002 [−0.002, 0.006] (s1), +0.005 [−0.001, 0.010] (s2);
  - wind and rain: all CIs include 0.
  - **The physics loss has at most a very small effect on hazard detection** (a tiny heat gain in 1 of 3 seeds).
  - The earlier padded-grid finding ("physics transfers worse") **does not replicate** once the grid confound is removed.
- **Against logistic regression:**
  - rainfall **+0.055 to +0.066, with the CI excluding 0 in all 6 runs**;
  - heat +0.011 to +0.019 (CIs include 0);
  - wind −0.012 (CIs include 0).
- Grid adaptation alone raised frozen-head rainfall PR-AUC from 0.174 / 0.194 (v1 frozen) to 0.22.

### Stage C: joint v2 from `A_phys_s1` (test PR-AUC; paired bootstrap vs frozen heads on the same encoder)

| | heatwave | extreme wind | extreme rainfall |
|---|---|---|---|
| frozen `A_phys_s1` | 0.245 | 0.060 | 0.223 |
| joint, capped uncertainty | 0.246 (−0.002 [−0.019, 0.016]) | 0.057 (−0.004 [−0.009, 0.001]) | 0.183 (**−0.043 [−0.085, −0.005]**) |
| joint, GradNorm | **0.251** (+0.002 [−0.017, 0.020]) | 0.060 (−0.000 [−0.007, 0.009]) | 0.193 (−0.033 [−0.073, 0.004]) |

- **The fixes worked mechanically.**
  - Weather stayed within 1 % of the Stage-A start: capped uncertainty +0.2 to +0.7 %, GradNorm −0.2 to +0.2 %.
  - No NaN batches; the hazard-weight cap held at 2.0.
  - Overfitting was slower: training heat BCE fell 1.11 → 0.34, while validation PR-AUC plateaued instead of collapsing.
- **GradNorm** pushed all hazard weights to the floor (0.05) from epoch 0, because their gradients on the shared layer were large. The encoder was then driven mainly by weather and physics, and the heads trained as usual (Adam is insensitive to the loss scale of parameters that only one task uses).
  GradNorm also slightly improved low-layer consistency (−3 % / −6 % vs start).
- **Joint fine-tuning still does not beat frozen heads on a grid-adapted encoder.** Heat and wind are unchanged; rainfall is worse (significantly so with uncertainty weighting).
  Fine-tuning the encoder on the sparse rainfall labels (9 IMD boxes) appears to overfit.

### Conclusions so far
1. **Best hazard system: a grid-adapted ClimaX encoder + frozen hazard heads.** It significantly beats logistic regression for extreme rainfall and all climatology / persistence baselines. It is level with logistic for heat and below it (not significantly) for wind.
2. **Single-layer hypsometric loss: no measurable benefit** for 72 h weather or hazard detection when fine-tuning, and it **degrades thickness consistency in the unconstrained lower layers**. This is seed-replicated.
   *Caveat:* these encoders are fine-tuned from one λ=0 checkpoint, not trained from scratch.
3. **Joint multi-task fine-tuning gives no hazard gain over frozen heads** and hurts rainfall. Adaptive balancing (capped uncertainty weighting, GradNorm) keeps weather stable but does not create positive transfer.

## Multi-layer hypsometric loss (v3, notebook 07): design

**Motivation.** In v2 the single 500–850 hPa constraint did not improve its own layer and pushed the error into the layers it does not see (700–850 hPa +36 %, 850–925 hPa ×2.4).

**Loss** (`joint.HypsometricResidual`, `--physics_layers adjacent --physics_scale_m auto`):
- Residual matching on **every adjacent level pair**: 925–850, 850–700, 700–600, 600–500, 500–250 and 250–50 hPa.
  - For each layer k: R_k = ΔΦ_k/g − (R_d/g)·T̄_k·ln(p_l/p_u).
  - L_phys = mean_k MSE(R_k^pred, R_k^true) / s_k².
- **Per-layer scale s_k:** the starting model's RMS residual error for that layer, on 4,096 training samples (floor 0.5 m). Every layer then starts at about 1, so thick, high-error layers (250–50 hPa) do not swamp the thin lower ones.
- The three layers from 850 to 500 hPa telescope to the 500–850 thickness, with a log-p trapezoidal mean temperature over 500/600/700/850 hPa. The old layer is still constrained, with a better T̄ than the two-level average (unit-tested).
- The defaults (`single`, 18 m) reproduce v2 exactly.
- Validation now logs the residual-matching error of all 7 layers every epoch (`val_hyps_*`). The test evaluation adds the 500–600 hPa layer.

**Experiment.**
- **Stage A:** weather + multi-layer physics, uncertainty weighting, 3 seeds. Same recipe as v2 Stage A, from the same λ=0 checkpoint.
- **Stage B:** frozen heads on all 9 encoders (multi, plus v2's nophys and phys, reused). Paired 7-day block bootstrap, multi vs each reference, per seed, plus multi vs logistic.

**Hypotheses.**
- H1: multi-layer physics reduces the residual error on all layers vs single-layer, and on the lower layers vs no physics.
- H2: it does not degrade 72 h weather RMSE by more than 1 %.
- H3: its effect on frozen-head PR-AUC (any sign) is reported with CIs.

## Result: multi-layer physics v3 (Kaggle 2×T4, 2026-11-20, commit `7b8f94d`)
Raw outputs: `docs/mtech/results/multilayer_v3/`.
- The six v2 encoders were reused and re-evaluated on identical test samples.
- No skipped batches.
- The auto scales were consistent across seeds: 3.8 / 5.6 / 2.7 / 2.3 / 18 / 70 m, from 925–850 up to 250–50 hPa.

### Stage A: test RMSE, mean over 3 seeds (seed sd ≤ 0.004 for every row below)
| | multi-layer | no physics | single layer (v2) | climatology |
|---|---|---|---|---|
| t2m (K) | 2.213 | **2.209** | 2.239 | 2.263 |
| u10 (m/s) | 1.792 | **1.790** | 1.803 | 1.804 |
| T850 (K) | **1.713** | 1.715 | 1.727 | 1.825 |
| z500 (m²/s²) | **250.9** | 251.9 | 251.3 | 280.0 |
| hyps. error 850–925 hPa (m) | **1.48** | 1.53 | 3.64 | 1.53 |
| hyps. error 700–850 hPa (m) | **2.93** | 3.01 | 4.10 | 3.02 |
| hyps. error 600–700 hPa (m) | **1.58** | 1.63 | 1.63 | 1.65 |
| hyps. error 500–600 hPa (m) | **1.76** | 1.77 | 3.25 | 1.77 |
| hyps. error 250–500 hPa (m) | 16.86 | **16.73** | 17.22 | 16.92 |
| hyps. error 50–250 hPa (m) | 70.47 | 70.51 | **70.39** | 74.31 |
| hyps. error 500–850 hPa, two-level T̄ (m) | 11.06 | 10.98 | **10.96** | 10.98 |

- **The v2 failure is fixed.**
  - Against the single-layer loss, the multi-layer loss cuts the residual error by 59 % (850–925 hPa), 28 % (700–850 hPa) and 46 % (500–600 hPa).
  - Weather is better on every reported variable (t2m −1.2 %).
- **Against no physics:**
  - The tropospheric layers below 500 hPa are 2.7–3.0 % more consistent (500–600 hPa −0.5 %), identically in all 3 seeds.
  - The multi-layer encoder is the only model that beats climatology on all four of these layers.
  - The upper layers are unchanged or slightly worse (250–500 hPa +0.8 %; 50–250 hPa ±0).
  - The old 500–850 metric uses the two-level mean temperature, which the new loss deliberately replaces with the trapezoidal one. It is +0.75 %.
- **Weather is neutral:** all variables are within ±0.4 % of no physics. The geopotential levels are slightly better (z500 −0.4 %), t2m +0.2 %.
- *Caveat:* the improved layers are the quantities the loss trains, so the gain is expected. The finding is that it comes at no weather cost and without moving error into other layers, unlike v2.

### Stage B: frozen heads, test PR-AUC (mean ± sd over 3 encoder seeds)
| | heatwave | extreme wind | extreme rainfall |
|---|---|---|---|
| multi-layer | 0.242 ± 0.001 | 0.060 | 0.219 ± 0.004 |
| no physics | 0.244 ± 0.001 | 0.060 | **0.224** ± 0.004 |
| single layer | **0.248** ± 0.002 | 0.060 | 0.219 ± 0.004 |
| logistic | 0.235 | **0.074** | 0.165 |

- **Multi vs no physics:** all 9 per-seed CIs include 0.
- **Multi vs single layer:**
  - Heatwave is slightly lower: −0.005 and −0.009, with CIs excluding 0 in 2 of 3 seeds. The effect is tiny.
  - Wind and rainfall are not significant.
- **Multi vs logistic:**
  - Rainfall is better in all 3 seeds: +0.054 to +0.063, with CIs excluding 0.
  - Heatwave is +0.010 and wind −0.012, both not significant.

### Conclusions (v3)
4. **A column-wide (all adjacent layers) hypsometric loss with per-layer scaling makes the 72 h forecast more physically consistent throughout the troposphere, at no weather cost.**
   - The single-layer loss of v2 degraded the layers it did not constrain; this design does not.
   - The physics loss is therefore useful as a **consistency regulariser**.
5. **No physics variant improves hazard detection.**
   - Frozen-head PR-AUC is the same within noise for no physics, single layer and multi-layer.
   - The hazard gain over logistic regression (extreme rainfall) comes from the grid-adapted ClimaX encoder itself.

## Thesis analysis of the v3 predictions (notebook 08, `scripts/mtech/make_analysis.py`, CPU only)
Outputs:
- figures in `docs/mtech/figures/analysis/`
- tables in `docs/mtech/results/analysis/`

"ClimaX" here means the mean probability of the frozen heads over the 3 multi-layer encoders (`A_multi_s0..2`).
This 3-encoder ensemble scores slightly above the single-encoder mean: heat 0.247 vs 0.242, rain 0.227 vs 0.219.

All fitted quantities use the validation years (2011–2014) only:
- the Platt calibration;
- the blend weight (grid 0, 0.1, …, 1);
- the alarm threshold (best validation F1).

### Calibration
- The raw frozen-head and logistic probabilities are **strongly over-confident**: ECE 0.16–0.28, Brier skill vs climatology −1.4 to −10. This comes from the class-weighted losses (pos_weight / `class_weight="balanced"`).
- **Platt scaling fixes it** without changing PR-AUC or ROC-AUC (it is monotonic). After calibration:
  - ECE: 0.011 (heat), 0.001 (wind), 0.003 (rain);
  - Brier skill vs climatology: +0.090, +0.014, +0.084.
- The calibrated probabilities are what a user should see.

### Blend of ClimaX and logistic (calibrated probabilities)
| hazard | w (ClimaX) | blend PR-AUC | vs ClimaX | vs logistic |
|---|---|---|---|---|
| heatwave | 0.5 | **0.260** | +0.014 [−0.005, 0.030] | **+0.025 [0.013, 0.035]** |
| extreme wind | 0.6 | **0.074** | **+0.012 [0.006, 0.021]** | +0.001 [−0.014, 0.012] |
| extreme rainfall | 1.0 | 0.227 (= ClimaX) | 0 | **+0.063 [0.017, 0.117]** |

The blend is at least as good as the better single model for every hazard, and the best overall system.
- For **wind**, logistic carries the skill and ClimaX adds some.
- For **rain**, validation gives logistic no weight.

### Skill maps (per cell, ≥ 10 test events)
- **Rainfall:** ClimaX beats logistic in 8 of 9 cells (+0.01 to +0.12), over central and peninsular India. It is worse in one cell (−0.05).
- **Heatwave:** 11 of 22 cells are better. ClimaX gains most in the central-west (25°N row, +0.11 / +0.13; 20°N 79°E, +0.10) and is worse across the 31°N row (−0.03 to −0.07).
- **Wind:** 11 of 24 cells are better. The cells where logistic is far better (−0.08 to −0.17) are over sea.

### Case studies (forecasts issued 3 days ahead; calibrated; validation-F1 threshold)
| event (test years) | event days | ClimaX hits / misses / false alarms | logistic |
|---|---|---|---|
| May 2015 heatwave, Andhra Pradesh / Telangana | 13 | 9 / 4 / 2 | 10 / 3 / 5 |
| Nov–Dec 2015 Chennai floods | 8 | **5 / 3 / 6** | 1 / 7 / 2 |
| July 2017 Gujarat floods | 6 | 1 / 5 / 3 | **3 / 3 / 7** |
| Oct 2018 Cyclone Titli | 3 | 0 / 3 / 0 | 0 / 3 / 0 |

- Both models picked up the 2015 heatwave about 3 days ahead; ClimaX had fewer false alarms.
- ClimaX caught most of the Chennai rain days, which logistic missed.
- Logistic did better for Gujarat 2017.
- **Neither predicted Cyclone Titli.** A tropical cyclone is far smaller than a 5.6° cell, and its extreme winds are not resolved in the coarse inputs. This is a stated limitation of the 5.625° setup.
- The 2018 Kerala floods are **not evaluable**: no IMD box with ≥ 50 % land coverage contains Kerala at this resolution.
