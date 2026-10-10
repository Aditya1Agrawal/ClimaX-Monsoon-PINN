# Implementation plan: physics-informed regional ClimaX for multi-hazard prediction

Guiding rule: one small, validated step at a time. No GPU training before the smoke tests pass and you approve it.
The original internship code (`src/climax/**`, `configs/regional_india.yaml`) stays unchanged. New code goes in `src/climax_hazards/`, `configs/mtech/`,
`notebooks/`, `scripts/mtech/`, `docs/mtech/`.

## Status

| Phase | Status |
|---|---|
| 1 Audit | Done: `docs/mtech/AUDIT_REPORT.md` |
| 2 Data/checkpoint access | Notebook `notebooks/01_inspect_data_and_checkpoints.ipynb` written and dry-run locally (real data + synthetic checkpoint). **Kaggle run pending** |
| Step B evaluation | Done. `epoch_016` beats persistence and climatology on all 17 variables (`docs/mtech/results/README.md`) |
| Step C corrected grid | Done locally: `regional_climax_dataset_6x6_corrected/` (lon 67.5–95.625°E, orography fixed, timestamps). Columns 73.125–95.625 are byte-identical to the original in all 320 shards. MD5 manifest: `docs/mtech/corrected_dataset_manifest.json`. **Upload to Kaggle pending (you)** |
| Step D labels | Heatwave (percentile + IMD-style, March–June), extreme wind and extreme precipitation (IMD 0.25°, 9 covered boxes) done; notebook `02_labels_eda.ipynb`; see `HAZARD_DEFINITIONS.md`. IMD day convention still to verify |
| 5–9 | Not started |

## Decisions taken (2026-10-10)

5.625° first; corrected 6×6 grid as the main dataset (the internship padded setup is kept as "baseline-as-run"); IMD rainfall as the precipitation label
source once you download it; the branch is pushed to GitHub.

## Decisions originally needed from you

1. **Grid resolution for the hazard work.**
   - A. Stay at 5.625° (cheap; matches the checkpoints; ~20 land cells; hazards only as grid-box-relative extremes). *Recommended as the first iteration.*
   - B. Later, also build 2.8125° or 1.40625° (much more meaningful hazards; tens to hundreds of GB of downloads and more GPU time).
2. **Corrected grid at 5.625°:** rebuild lon `67.5 … 95.625` (real 6 × 6, no padding) from the local raw files into a new folder. *Recommended.*
   The internship 6×6-padded setup stays as "baseline-as-run".
3. **Precipitation source:** IMD 0.25° gridded daily rainfall (observations; *recommended for labels*) and/or WeatherBench 5.625° `total_precipitation` (ERA5).
   Both need a manual download by you (sources in Phase 3 below).
4. **Push the branch to GitHub?** Kaggle notebooks clone it. The alternative is uploading the repo as a private Kaggle dataset.

## Next steps (in order)

**Step A: finish Phase 2 (you, ~10 min on Kaggle, CPU is enough):** run notebook 01 with both datasets attached. Send back the printed provenance table,
the load report and `inspection_report.json`. This settles which checkpoint belongs to which run and confirms the Kaggle data equals the local data.

**Step B: evaluation baseline (CPU/GPU, small):** `scripts/mtech/evaluate_forecast.py`. Full test-set lat-weighted RMSE/ACC for `epoch_016.ckpt`,
computed both with and without the duplicated column, next to persistence and climatology on identical samples. This gives the honest reference point.

**Step C: corrected-grid dataset (local CPU):** `scripts/mtech/build_regional_dataset.py`, using the same conventions as `preprocess_and_crop.py`
with three changes: lon 67.5–95.625, the orography key fixed, and a `times.npy` per shard. It is written to a new folder and is MD5-manifested.

**Step D: Phase 4 labels (local CPU):** `src/climax_hazards/labels/` with the heatwave and extreme-wind pipelines.
- Daily aggregation over UTC days from hourly data. The missing 31 Dec in leap years is marked *missing*, not *normal*.
- Thresholds come from train years (1979–2010) only: calendar-day percentiles with a ±7-day window (heat), and per-cell high percentiles (wind).
- A sensitivity grid (90/95/99th percentile; 1/2/3/5-day duration; IMD absolute criteria checked for applicability).
- Prevalence per split, cell and month; saved metadata JSON; plots.
- Lead-time alignment: the label is for the target day(s) at `t + lead`. Windows never cross split boundaries.

**Step E: Phase 5/6 model (local CPU smoke test, then Kaggle):**
`ClimaX encoder (shared) → existing weather head + per-hazard heads`. Each head is a small MLP on the 9 decoder tokens that is unpatchified
to per-cell logits `(B, 1, H, W)`. The loss is `L_weather + λ_phys·L_hyps + Σ λ_k·BCE_k`, with pos_weight or focal loss chosen from train prevalence
and a land mask for heat. Loss components and per-component gradient norms are logged. The existing `HypsometricLoss` is reused, not changed.
Optional and separate: name-aware loading of the pretrained 5.625° ClimaX weights (remap `var_embed` and `token_embeds` by variable name, crop `pos_embed`).

**Step F: Phase 7 experiments (GPU, only after your approval):** the same splits, seed and evaluation for
(1) weather λ=0, (2) weather λ=0.001 (= internship), (3) multi-task + physics, (4) per-hazard baselines (climatology/persistence of labels,
logistic regression on the same inputs). Hazard metrics: precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, prevalence.

## Notebooks (Kaggle)

1. `01_inspect_data_and_checkpoints.ipynb`: done (pending Kaggle run)
2. `02_labels_eda.ipynb`: Step D
3. `03_model_smoke_test.ipynb`: Step E
4. `04_train_experiments.ipynb`: Step F

## References used so far

- Nguyen et al. 2023, *ClimaX: A foundation model for weather and climate*, ICML (architecture, pretrained checkpoints).
- Rasp et al. 2020, *WeatherBench*, JAMES (5.625° ERA5 dataset; also provides `total_precipitation`).
- Hersbach et al. 2020, *The ERA5 global reanalysis*, QJRMS.
- Pai et al. 2014, *Development of a new high spatial resolution (0.25° × 0.25°) long period daily gridded rainfall data set over India*, Mausam.
- IMD heatwave criteria (IMD FAQ / *Heat Wave* guidance; NDMA heatwave guidelines 2019).
- Perkins & Alexander 2013, *On the measurement of heat waves*, J. Climate (percentile/duration definitions).
- Wallace & Hobbs 2006, *Atmospheric Science*, ch. 3 (hypsometric equation).
