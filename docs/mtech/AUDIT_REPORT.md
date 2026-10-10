# Audit report: internship regional ClimaX (Phase 1)

Branch: `mtech-extreme-events` (created from `main` @ `32d9e5d`; `main` untouched).
Date of audit: 2026-10-10. Everything below was checked against the code, the local data, the raw WeatherBench files,
or the internship notebook log (`internship-project (5).ipynb`). Items that could not be verified are marked **UNVERIFIED**.

## 1. What the internship actually ran

| Item | Finding | Evidence |
|---|---|---|
| Entry point | `src/climax/global_forecast/train.py --config configs/regional_india.yaml --trainer.max_epochs=20` | notebook cell 4 |
| Code version | GitHub `Aditya1Agrawal/ClimaX-Monsoon-PINN` cloned on Kaggle after commit `32d9e5d` (13:30 IST); run started 10:07 UTC (15:37 IST) | log timestamps, commit time |
| Hardware | 2 GPUs, DDP, fp32, per-GPU batch 32, `num_workers=1` | log |
| Steps | 4092 optimiser steps/epoch, 20 epochs, ~14 min/epoch | log |
| Seed | 42 (`seed_everything`) | log |
| Physics | `physics_lambda = 0.001`, 500/850 hPa hypsometric residual | log print, config |
| Checkpoint selection | `ModelCheckpoint(monitor=val/w_rmse, top_k=3, save_last)`; EarlyStopping patience 5 never triggered | log |
| Best checkpoint | `epoch_016.ckpt`, `val/w_rmse=103.58346`; test was run on it (`ckpt_path="best"`) | log |
| **Pretrained weights** | **Not used. The model was trained from scratch.** `regional_india.yaml` has no `pretrained_path` (default `""`), the command does not override it, and the log has no "Loading pre-trained checkpoint" line | config, module, log |
| Baseline without physics | **No run found.** There is no λ=0 checkpoint or log | notebook, Kaggle listing |

Consequence: the existing result should be described as "a ClimaX-architecture model trained from scratch on regional data,
with a hypsometric loss term". It is not fine-tuned *pretrained* ClimaX. No claim about the physics loss is possible yet
because no ablation exists.

## 2. Regional dataset

Local copy `regional_climax_dataset/` is **byte-identical** (MD5) to the manifest in `docs/mtech/local_dataset_manifest.json`.
**VERIFIED on Kaggle (2026-10-10):** the attached `climax-indian-monsoon-regional` dataset is byte-identical to this manifest
(327 of 327 files, 0 missing, 0 extra, 0 MD5 mismatches). All locally computed results therefore use exactly the Kaggle data.

* Source: WeatherBench 5.625° ERA5 (`raw_global_data/5.625deg`, 1979–2018, present locally, 78 GB). Only 5 variables:
  `2m_temperature`, `10m_u/v_component_of_wind`, `geopotential` and `temperature` on 13 levels, plus `constants.nc`.
* Built by `preprocess_and_crop.py`. Grid: lat `8.4375 … 36.5625` (6, **ascending**), lon `73.125 … 95.625` (5) → **6 × 5**.
  One cell is 5.625° (~625 km), so the whole region has 30 cells.
* Temporal: **hourly**. Shard `Y_k` contains 1095 steps starting at `Y-01-01T00 UTC + k·1095 h`. Every year is truncated to 8760 h,
  so **leap years lose 31 December**. Verified by exact match against the raw NetCDF files for 1979, 2010 and 2016.
  Shards store no timestamps; use `climax_hazards.inspect_utils.shard_timestamps`.
* Splits are chronological and disjoint: train 1979–2010 (256 shards, 280 320 h), val 2011–2014, test 2015–2018 (32 shards each).
* 22 arrays per shard, `(1095, 1, 6, 5)`. Values: no NaN/inf in any of the 320 shards.
* Normalisation: per-variable scalars from **train only**. Climatology files are per split, shape `(1, 6, 5)`.
* Bug: `orography` is **all zeros, std = 0**. `NAME_TO_VAR` maps it to `orog`, but the key in `constants.nc` is `orography`, so the
  zero fallback was used. It is not among the 17 training variables, so the internship run is unaffected. Fix it before using orography.
* Not used in training although present: `geopotential_1000`, `temperature_1000`, `land_sea_mask`, `lattitude`.
* No precipitation and no humidity in the dataset or in the local raw data.

## 3. The longitude padding

1. **Where:** only in the notebook. It copies the read-only input to `/kaggle/working/regional_climax_dataset_6x6` and edge-pads there.
   The uploaded Kaggle dataset is **unpadded** (6 × 5). This is confirmed by the local MD5 match; notebook 01 also checks it on Kaggle.
2. **Shapes:** `(T,1,6,5)` → `(T,1,6,6)`, climatology `(1,6,5)` → `(1,6,6)`. `lon.npy` becomes `[73.125 … 95.625, 101.25]`.
   The normalisation scalars `(1,)` are skipped by the `ndim >= 2` guard and stay valid. All arrays stay mutually consistent with `img_size: [6, 6]`.
3. **What the extra column is:** a copy of lon 95.625°E labelled 101.25°E. A real 101.25°E column exists in the raw data and differs from it.
   So the padded column is a fake location. The model receives its own positional embedding and must predict it.
4. **Effect on losses and metrics:** yes, it contributes. The lat-weighted MSE (training), the hypsometric loss, and the
   val/test RMSE/ACC average over all 36 cells. The 95.625°E column therefore carries double weight (2 of 6 columns instead of 1 of 5).
   Checkpoint selection (`val/w_rmse`) and the reported test metrics include the duplicate. Notebook 01 reports RMSE with and without that column.
5. **Root cause:** `LON_REGS = [68.0, 98.0]` excludes the 67.5°E grid point (`crop_india.py` used 65°E). With 67.5 included the crop is 6 × 6 of real data.
6. **Recoverable:** yes. The raw global files are present locally. The intended crop is lon `67.5 … 95.625`, which adds Gujarat, Rajasthan and Pakistan.
   That column is also aligned with the 2 × 2 patches of the global 5.625° grid (lon indices 12–17).
7. **Do not pad twice:** `inspect_utils.pad_lon_edge` refuses anything that is not 5 columns wide, and `detect_lon_padding` flags already-padded data.
8. **Reproducibility:** the original dataset and padding are left as they are. A corrected-grid dataset should be written to a **new** folder,
   with the internship setup kept as "baseline-as-run".

## 4. Model and checkpoints

* Architecture (verified by building it): ClimaX, 17 vars, `img_size [6,6]`, patch 2 → **9 tokens**, embed 1024, depth 8, decoder 2, heads 16,
  **107.26 M parameters**. This matches the log. It is a very large model for 9 tokens.
* Kaggle `checkpoints-for-regional-forecasting`: `epoch_011.ckpt`, `epoch_016.ckpt`, `epoch_017.ckpt`, `last.ckpt`, about 1.29 GB each.
  The size fits weights plus AdamW state (3 × 430 MB). The filenames are exactly the top-3 + last of the logged physics run.
  That is strong but not conclusive. Notebook 01 compares the stored `epoch` and `global_step` with the log
  (49104 / 69564 / 73656 / 81840) and prints the stored `physics_lambda` and `pretrained_path`.
  **VERIFIED on Kaggle (2026-10-10, `docs/mtech/results/kaggle_inspection/`):** all four checkpoints match the logged epoch and global step exactly.
  All store `physics_lambda=0.001`, `pretrained_path=''` (this confirms training from scratch) and `root_dir=/kaggle/working/regional_climax_dataset_6x6`.
  The stored `ModelCheckpoint` scores equal the logged `val/w_rmse` (103.62589 / 103.58346 / 103.68472), and `best_model_path` = `epoch_016.ckpt`.
  All four load **strictly** into the `regional_india.yaml` architecture (no missing, unexpected or mis-shaped keys).
  The Kaggle dataset is the unpadded 6 × 5 grid with no issues found. All four checkpoints therefore come from the single λ=0.001 physics run; none is a baseline.
* Strict-loading the internship architecture works on a synthetic checkpoint with the same layout (local test).
* Pretrained-loading hazard, relevant for the extension: `load_pretrained_weights` keeps keys whose **shapes** match.
  `token_embeds.0…16` of the 48-variable pretrained model match in shape, so they would load onto **different variables**
  (index-based, not name-based). `var_embed` and `head` would be dropped, and `pos_embed` would be interpolated from the whole globe down to 3 × 3.
  Loading pretrained weights correctly needs name-aware remapping. **UNVERIFIED:** the pretrained variable order must be taken from the
  pretraining config, because the checkpoint does not store names.

## 5. Physics-informed loss (`src/climax/utils/physics_loss.py`)

* Formula: `R = (Φ500 − Φ850)/g − (R_d/g)·½(T500+T850)·ln(850/500)`, loss = `MSE(R_pred, R_target)`.
  Units are correct: ERA5 stores geopotential Φ (m² s⁻²), and the code divides by g. The pressure order is correct (ΔZ > 0, ln > 0).
  Computed on denormalised fields. Channel indices are resolved by name.
* Residual matching is justified: on the training data, `R_target` has mean 3.7 m and std **18.0 m** (15.5 m with a 4-level log-p mean),
  because of moisture (virtual temperature) and the two-level T̄. Forcing R = 0 would penalise real atmospheric states. There is no humidity
  in the data, so a virtual-temperature correction is not possible.
* Interpretation: `R_pred − R_target = ΔΦ_err/g − 15.5·T̄_err` is a fixed linear combination of the forecast errors. So the term is a
  physically motivated **supervised** penalty on inconsistent errors. It is not a label-free physical constraint. Say this in the write-up.
* Not double-counted: the total loss is `lat_weighted_mse + λ·physics`. The MSE that `net.forward` returns is only logged.
* Magnitude: the weighted physics term was ~36 % of the total loss at epoch 0 (0.0586 / 0.163) and ~26 % at epoch 19 (0.0077 / 0.030).
  It is not negligible. The raw residual MSE fell from ~59 m² to ~8 m².
* Includes the duplicated column, and has no latitude weighting. These are minor.
* Physics is **not** used in validation or model selection.
* `physics_loss.py` (repo root) `__main__` calls `loss_fn(predictions)` with one argument, so it is broken. This is cosmetic.

## 6. Training and evaluation procedure: other issues

* `val/w_rmse` (the selection metric) averages RMSE across variables in **physical units**, so geopotential (~10²–10³ m² s⁻²) dominates selection.
* ACC uses each split's own climatology (standard ClimaX practice). That uses test-period means, but only inside the ACC metric.
* No persistence or climatology baseline was computed. Skill relative to persistence is unknown. **This is needed before any claim.**
* Forecast pairs are formed inside a shard, so there is no cross-split leakage. The last 72 h of every shard are unused as inputs.
* The LR schedule (warm-up 10k steps, cosine to 100k) was cut at 81 840 steps.
* Debug prints were added to the datamodule and climate-projection module (harmless). `climate_projection/train.py` has a stray `from pydoc import cli`.

## 7. Hazard feasibility (preliminary; data facts verified, definitions to be finalised in Phase 4)

| Hazard | Inputs available | Labels | Feasible now? | Key limitation |
|---|---|---|---|---|
| Extreme precipitation | **No precip variable** anywhere locally | none | **No** | Needs IMD 0.25° gridded daily rainfall (Pai et al. 2014; observational, land-only, 03–03 UTC day) or WeatherBench 5.625° `total_precipitation` (ERA5, same grid and timeline, model-based). Do not derive it from T, Z or wind |
| Heatwave | hourly t2m, 1979–2018 | derive daily Tmax (UTC day) | **Yes**, percentile + duration | IMD criteria (≥40 °C plains, departure ≥4.5 °C, or ≥45 °C) are station-scale. *Update after measuring:* box-mean Tmax does reach ≥40 °C on ~3 % of train land cell-days, but the full IMD-style rule fires on only ~0.3 % of them. The primary label is therefore a train-period calendar-day percentile with ≥3-day duration, on land cells only; the IMD-style label is kept for comparison (see `HAZARD_DEFINITIONS.md`) |
| Extreme wind | hourly u10, v10 | `sqrt(u²+v²)`, percentile | **Yes**, as "extreme grid-scale 10 m wind" | Box-mean wind, no gusts; ocean cells dominate. **Not** cyclone labels (needs IBTrACS) |
| Flood / cyclone / landslide | — | none | No | Need GFD/DFO, IBTrACS, NASA GLC and environmental inputs |

**Overarching limitation:** at 5.625° the whole Indian region is 30 cells (~20 land cells). Extremes are local, and grid-box means smooth them strongly.
A hazard study at this resolution is defensible only as a proof of concept of a multi-task physics-informed pipeline,
with "extreme" defined relative to grid-box climatology. A finer grid (2.8125° / 1.40625° WeatherBench, or ERA5 subsets from CDS) would make the hazards far more meaningful,
but costs much more data and compute. **This is the main decision for you (see the plan).**

## 8. Severity summary

| # | Issue | Severity | Action |
|---|---|---|---|
| 1 | Model was trained from scratch, not pretrained | High (framing) | Correct the description; optionally add a name-aware pretrained init experiment |
| 2 | No λ=0 baseline | High (claims) | Run a matched baseline before claiming any physics effect |
| 3 | Duplicated fake longitude column in train/val/test metrics | Medium | Report on real columns; build a corrected 67.5–95.625°E grid |
| 4 | No persistence/climatology baselines | Medium | **Done**: `epoch_016` beats both on all 17 variables, by 2–23 % vs climatology (`results/README.md`) |
| 5 | Selection metric dominated by geopotential | Medium | Also log normalised or per-variable selection metrics |
| 6 | Orography zeros | Low (unused) | Fix the key in the new preprocessing |
| 7 | Leap-year 31 Dec dropped; shards have no timestamps | Low | Handled by `shard_timestamps`; matters for heatwave durations |
| 8 | Index-based pretrained loading | Medium (future) | Name-aware loader in `climax_hazards` |
