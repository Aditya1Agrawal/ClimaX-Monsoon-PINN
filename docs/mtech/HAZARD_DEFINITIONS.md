# Hazard definitions and label generation (Phase 3–4)

Implementation: `src/climax_hazards/labels.py`, driver `scripts/mtech/make_labels.py`, notebook `notebooks/02_labels_eda.ipynb`.
Measured prevalences: `docs/mtech/results/labels_corrected6x6/` (the primary dataset is the corrected 6 × 6 grid, lon 67.5–95.625°E).

## Common rules (own implementation choices)

| Rule | Choice | Why |
|---|---|---|
| Time unit | UTC day built from 24 hourly values. A day with any missing hour is NaN | The dataset is hourly UTC. Indian Tmax occurs ~08–11 UTC, inside the same UTC day |
| Missing data | Label NaN = undetermined. **Never** counted as a non-event | 31 Dec of leap years is absent from the dataset |
| Thresholds | Computed on train years 1979–2010 only, applied unchanged to val/test | No future information in the labels |
| Spells | Evaluated inside one split. A spell shorter than the minimum that touches a gap or split edge is NaN | No label depends on another split; truncated spells are not called non-events |
| Spatial unit | 5.625° grid cell (~625 km), box-mean ERA5 | This is all the data resolves. "Extreme" means extreme for that grid box |
| Forecast target | Input at hour `t` → label of the UTC day containing `t + 72 h`, from the same split | `align_to_forecast`; targets outside the split are dropped |

## A. Extreme precipitation: not implemented (data missing)

Neither the regional dataset nor the local raw WeatherBench folder contains precipitation. Labels will **not** be derived from temperature, wind or geopotential.
Suitable sources:

1. **IMD gridded daily rainfall, 0.25°** (Pai et al. 2014, *Mausam* 65(1)). Observation-based (gauges), 1901–present, land only.
   The rainfall day ends at 03 UTC (08:30 IST), so the labels must be shifted by 3 h relative to UTC days.
   Regrid by area-weighted (conservative) averaging of the 0.25° cells inside each 5.625° box, and require a minimum land-cell coverage per box.
   **Recommended for labels.** IMD's categories (heavy ≥ 64.5 mm/day, very heavy ≥ 115.6, extremely heavy ≥ 204.5) are station/point thresholds.
   For box means, a train-period percentile of wet-day box-mean rainfall (e.g. 95th/99th, per season) will be needed. The absolute IMD categories are only a comparison.
2. **WeatherBench 5.625° `total_precipitation`** (ERA5 hourly accumulation, m), on the same grid and timeline (1979–2018).
   Alignment is trivial, but it is a model product with known biases over India. Use it as a second label source or as an input, not as the only "truth".

Manual step for you: download IMD gridded rainfall (`imdlib` Python package, or the IMD Pune data portal), 1979–2018. Optionally download WeatherBench
`total_precipitation_5.625deg` from the WeatherBench mediaTUM archive.

## B. Heatwave (implemented)

**Primary: percentile + duration** (published approach: Perkins & Alexander 2013, *J. Climate* 26, CTX90pct-type; the ±7-day calendar window follows the ETCCDI convention).
- Daily Tmax (UTC day) > the 90th percentile of the train-period Tmax for that calendar day (±7-day window), on **≥ 3 consecutive days**. Every day of the spell is labelled 1.
- Land cells only (land fraction ≥ 0.5; **23 of 36** cells on the corrected grid).
- **Hot season only (March–June; IMD: heat waves occur mainly in March–June).** Other days are NaN and not scored. Spells are found on all days first, so a spell crossing 1 March is still detected.
  Why: without this filter the label fires all year and peaks in Oct–Nov of the test years. That is a warm-anomaly label, not a heatwave. The all-year variant stays in the sensitivity table.

**Comparison: IMD-style** (IMD heat-wave criteria; NDMA 2019 guidelines).
- Tmax ≥ 40 °C (≥ 30 °C where orography > 1500 m) **and** departure from the train-period calendar-day mean Tmax ≥ 4.5 °C, **or** Tmax ≥ 45 °C, on ≥ 2 consecutive days, in the same March–June window.
- Adaptations, which are my choices: IMD applies these criteria to station observations and requires ≥ 2 stations in a subdivision.
  Here they are applied to box-mean ERA5 Tmax, the "normal" is the 1979–2010 train mean rather than IMD's station normal, and the hill/plain split uses grid-box orography.
  The coastal criterion (≥ 37 °C) is not applied, because coastal stations cannot be identified at this resolution.
- On the original grid, orography is all zeros (bug), so the hill criterion only works on the corrected dataset.

## C. Extreme wind (implemented)

- Wind speed = `sqrt(u10² + v10²)` per hour. Daily maximum of the hourly values.
- **Primary:** daily max > the per-cell 99th percentile of train-period daily maxima (all seasons). All cells, land and sea.
- **Sensitivity:** 95/99/99.9th percentiles, per-month thresholds, and the absolute Beaufort 7 (13.9 m/s) and 8 (17.2 m/s) thresholds.
- Interpretation: "anomalously strong grid-box-mean 10 m wind". These are **not** gust, storm or cyclone labels.
  Cyclones need IBTrACS tracks; gusts are not in the dataset.

## Not feasible now

Flood (needs inundation or impact records, e.g. Global Flood Database or DFO, and hydrological inputs), cyclone (needs IBTrACS), landslide (needs the NASA Global Landslide Catalog plus slope and soil inputs).
Precipitation labels are not flood labels, and extreme-wind labels are not cyclone labels.

## Measured results (corrected 6 × 6 grid; `docs/mtech/results/labels_corrected6x6/`)

Prevalence = event cell-days / valid cell-days. NaN cell-days (sea cells for heat, out-of-season days, missing days, truncated spells) are excluded.

| Label (primary setting) | train 1979–2010 | val 2011–14 | test 2015–18 |
|---|---|---|---|
| `heatwave_pct` (MAMJ, p90, ≥3 d, land) | 5.33 % | 6.53 % | 9.17 % |
| `heatwave_imd` (MAMJ, ≥2 d, land) | 0.89 % | 0.69 % | 0.95 % |
| `extreme_wind` (p99 of daily max, all cells) | 1.00 % | 0.83 % | 0.90 % |

Findings:
1. **Non-stationarity (heat).** Fixed 1979–2010 thresholds give 1.7× more heatwave days in 2015–2018, and the effect grows for rarer thresholds
   (p99/≥3 d: 0.19 % → 1.05 %). This is the warming trend, not leakage. Consequences: (a) class prevalence differs between training and test,
   so threshold-free metrics (PR-AUC) and calibration must be reported per split; (b) the test set is not a held-out sample from the training
   distribution. A detrended-threshold variant can be added as a sensitivity check.
2. **Wind is stationary.** It is ~1 % by construction in train and 0.8–0.9 % in val/test. Its seasonal peak is in June (monsoon onset, low-level jet).
   Only 0.02–0.03 % of cell-days exceed Beaufort 8 (17.2 m/s), which confirms that box-mean winds cannot represent gales.
3. **IMD criteria at grid scale.** Box-mean Tmax ≥ 40 °C on 7 594 land cell-days in train (max 49.0 °C). Combined with the 4.5 °C departure,
   the IMD-style label is ~6–10× rarer than the percentile label and concentrated in NW India and Pakistan.
4. **Class imbalance** of roughly 1:10 (heat) to 1:100 (wind) needs weighted or focal losses and PR-based metrics. Accuracy is meaningless here.

Full sensitivity grid: `threshold_sensitivity.csv`. Seasonal and monthly prevalence: `monthly_prevalence.csv`. Maps: `label_maps_and_seasonality.png`.
