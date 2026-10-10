# Measured results

Only numbers produced by code in this repository are recorded here. Each file states the dataset it came from.

| File | What | Produced by |
|---|---|---|
| `baselines_orig6x5_test.csv` | Persistence and train day-of-year×hour climatology, test 2015–2018, 72 h lead, original 6×5 grid (n = 32 736 pairs, the same pairs ClimaX tests on) | `scripts/mtech/evaluate_forecast.py --data_dir regional_climax_dataset` |
| `baselines_corrected6x6_test.csv` | The same baselines on the corrected 6×6 grid (lon 67.5–95.625°E) | `... --data_dir regional_climax_dataset_6x6_corrected` |
| `kaggle_inspection/` | Output of notebook 01 on Kaggle: checkpoint metadata and provenance (all four match the internship log), strict-load report, and a **64-sample** val forward smoke test of `epoch_016.ckpt` (a pipeline check, not an evaluation) | `notebooks/01_inspect_data_and_checkpoints.ipynb` |
| `labels_corrected6x6/` | Hazard label prevalence, threshold sensitivity, seasonality, maps, IMD diagnostics | `scripts/mtech/make_labels.py` |

## Reference point for the internship checkpoint (indicative only)

Logged internship test metrics (`epoch_016.ckpt`, 6×6 padded grid **including** the duplicated column, ClimaX metric) compared with the baselines
on the 5 real columns (original grid). These are not yet a same-sample comparison: run `evaluate_forecast.py --checkpoint` on Kaggle for that.

| variable | internship log | persistence | climatology |
|---|---|---|---|
| z500 (m² s⁻²) | 233.6 | 282.0 | 272.8 |
| z850 | 233.6 | 282.3 | 235.4 |
| z925 | 243.2 | 294.1 | 242.7 |
| t850 (K) | 1.45 | 1.84 | 1.63 |
| t2m (K) | 2.14 | 2.47 | 2.28 |
| u10 (m/s) | 1.74 | 2.15 | 1.77 |
| v10 (m/s) | 1.58 | 2.12 | 1.64 |

The model beats persistence everywhere but is close to climatology for low-level geopotential and the 10 m winds. No skill claim should be made until the same-sample evaluation exists.
