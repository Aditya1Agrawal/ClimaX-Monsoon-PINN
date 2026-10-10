# Measured results

Only numbers produced by code in this repository are recorded here. Each file states the dataset it came from.

| File | What | Produced by |
|---|---|---|
| `baselines_orig6x5_test.csv` | Persistence and train day-of-year×hour climatology, test 2015–2018, 72 h lead, original 6×5 grid (n = 32 736 pairs, the same pairs ClimaX tests on) | `scripts/mtech/evaluate_forecast.py --data_dir regional_climax_dataset` |
| `baselines_corrected6x6_test.csv` | The same baselines on the corrected 6×6 grid (lon 67.5–95.625°E) | `... --data_dir regional_climax_dataset_6x6_corrected` |
| `kaggle_inspection/` | Output of notebook 01 on Kaggle: checkpoint metadata and provenance (all four match the internship log), strict-load report, and a **64-sample** val forward smoke test of `epoch_016.ckpt` (a pipeline check, not an evaluation) | `notebooks/01_inspect_data_and_checkpoints.ipynb` |
| `labels_corrected6x6/` | Hazard label prevalence, threshold sensitivity, seasonality, maps, IMD diagnostics | `scripts/mtech/make_labels.py` |

## Internship checkpoint `epoch_016.ckpt`: same-sample test evaluation (measured on Kaggle, 2026-10-10)

Files: `eval_epoch016_orig6x5_test.csv` (full output) and `eval_epoch016_skill_summary.csv`. Test 2015–2018, 72 h lead, n = 32 736 pairs.
"real" = the 5 real longitude columns; "incl_dup" = the internship's padded 6×6 grid, including the duplicated column.

**Pipeline check:** the `incl_dup` scores reproduce the internship's logged ClimaX test metrics to ≤ 5e-5 (z500 233.6048, t2m 2.1399, z925 243.1508).

| variable | model (real) | persistence | climatology | skill vs clim | skill vs pers | duplicate-column effect |
|---|---|---|---|---|---|---|
| t2m (K) | 2.134 | 2.469 | 2.284 | 6.6 % | 13.6 % | +0.3 % |
| u10 (m/s) | 1.697 | 2.151 | 1.767 | 4.0 % | 21.1 % | +2.7 % |
| v10 (m/s) | 1.603 | 2.116 | 1.637 | 2.1 % | 24.2 % | −1.5 % |
| t500 (K) | 1.685 | 2.113 | 1.882 | 10.5 % | 20.3 % | −0.1 % |
| t850 (K) | 1.455 | 1.842 | 1.632 | 10.9 % | 21.0 % | −0.5 % |
| z50 (m² s⁻²) | 264.8 | 266.7 | 344.5 | 23.1 % | 0.7 % | −0.3 % |
| z250 | 404.4 | 475.1 | 510.5 | 20.8 % | 14.9 % | −1.3 % |
| z500 | 237.4 | 282.0 | 272.8 | 13.0 % | 15.8 % | −1.6 % |
| z700 | 214.1 | 260.2 | 227.2 | 5.8 % | 17.7 % | +0.9 % |
| z850 | 228.2 | 282.3 | 235.4 | 3.1 % | 19.2 % | +2.4 % |
| z925 | 236.7 | 294.1 | 242.7 | 2.5 % | 19.5 % | +2.7 % |

Skill = 1 − RMSE_model / RMSE_baseline. All 17 variables are in the CSV.

Interpretation:
- The model beats persistence and train-climatology on all 17 variables. The margin over climatology is small (2–6 %) for near-surface fields and the lower-tropospheric geopotential.
- The duplicated column changes the reported RMSE by up to ±2.7 %. The internship's headline numbers should be quoted from the "real" columns.
- No statement about the physics loss is possible: this is the λ=0.001 model, and no λ=0 model exists yet.
- Climatology has no ACC by construction (its anomaly is zero).

## Physics-loss ablation, single seed (λ = 0 vs λ = 0.001), original padded grid

File: `physics_ablation_seed42_orig_padded_climaxmetrics.csv`. These are the ClimaX `test_step` metrics, transcribed from the two Kaggle test tables:
padded 6×6 grid including the duplicated column, ClimaX ACC (split climatology), test 2015–2018, 72 h. Both runs used seed 42 and 20 epochs, with selection on `val/w_rmse`.
Best `val/w_rmse`: λ=0 → 104.877, λ=0.001 → 103.583.

**Result:** with λ=0.001, test RMSE is lower on **17/17** variables (−1.3 % to −4.0 %, median −2.7 %) and ACC is higher on **17/17** (+0.004 to +0.021).
The largest gains are in lower-tropospheric temperature (t850 −4.0 %, t925 −3.8 %, t700 −3.6 %) and the 10 m winds (−3.0 to −3.5 %).

**What this does and does not show:**
- It is **one run per setting**. Run-to-run variance (seed, data order, DDP non-determinism) is unknown, so the 1–4 % differences cannot yet be called significant.
  The consistent sign across all 17 correlated variables is encouraging, but it is not independent evidence. **Needed:** ≥ 3 seeds per setting (notebook 03).
- The scores include the duplicated column. Re-score both checkpoints on the real columns with `evaluate_forecast.py` (notebook 03, `EXTRA_EVAL`).
- "More physically consistent" has **not been measured yet**. `evaluate_forecast.py` now reports `hyps_resid_<upper>_<lower>`, the RMSE of the hypsometric-residual error in metres,
  for the trained layer (500–850 hPa) and for five untrained layers. The test baselines are 10.5 m (climatology) and 13.0 m (persistence) for 500–850 hPa.
- Wall-clock: λ=0 took 3 h 59 m and λ=0.001 took 4 h 50 m. The physics run has 21 % more runtime with identical step counts, which is worth explaining (logging or DDP overhead?).
