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

## A. Extreme precipitation (implemented; `src/climax_hazards/precip.py`)

**Source:** IMD gridded daily rainfall, 0.25° (Pai et al. 2014, *Mausam* 65(1)). Observation-based (rain gauges), land only.
Downloaded 1979–2018 with `imdlib` into `raw_imd_rainfall/rain/<year>.grd` (971 MB, git-ignored).
The reader matches `imdlib` exactly, and all 40 years passed size and value checks.
Labels are **not** derived from temperature, wind or geopotential.

**Regridding (my choice):** area-conservative averaging (cos-latitude weighted overlap) of the 0.25° cells into each 5.625° box.
`coverage` is the valid fraction of the box area. Boxes with mean coverage < 0.5 are NaN.

**Spatial limitation:** IMD covers India only, so just **9 of 36 boxes** reach 50 % coverage (14 at 30 %, 6 at 70 %).
Pakistan, Nepal, Bangladesh, Tibet, Myanmar and the seas have no precipitation labels. The covered boxes are central, north and north-east India.

**Primary definition** (ETCCDI R95p style): daily box-mean rainfall > the per-box 95th percentile of train-period (1979–2010) **wet-day** (≥ 1 mm) box-mean rainfall, all year.
The thresholds come out at 14–30 mm/day.

**Sensitivity:** 90/99th percentile; 2.5 mm wet day (IMD "rainy day"); coverage 0.3 or 0.7; June–September only; IMD absolute categories.
- The IMD "heavy" (64.5 mm/day) and "very heavy" (115.6 mm/day) categories almost never occur in box means (≤ 0.04 % and 0 %).
- Those station categories therefore **cannot** be used at 5.625°. Percentile labels are the only defensible choice.

**Day convention (UNVERIFIED):** an IMD daily value is a 24 h accumulation ending at 08:30 IST (03 UTC).
Whether IMD date D means 03 UTC D-1 → D or D → D+1 has to be confirmed in the IMD data documentation before forecast alignment.
It is a parameter (`imd_day_starts_on_date`) and shifts the target by one day.

**Alternative / second source:** WeatherBench 5.625° `total_precipitation` (ERA5, same grid; model-based). Not used yet.

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
| `extreme_precip` (p95 of wet days, coverage ≥ 0.5, 9 boxes) | 2.01 % | 2.05 % | 1.79 % |

Findings:
1. **Non-stationarity (heat).** Fixed 1979–2010 thresholds give 1.7× more heatwave days in 2015–2018, and the effect grows for rarer thresholds
   (p99/≥3 d: 0.19 % → 1.05 %). This is the warming trend, not leakage. Consequences: (a) class prevalence differs between training and test,
   so threshold-free metrics (PR-AUC) and calibration must be reported per split; (b) the test set is not a held-out sample from the training
   distribution. A detrended-threshold variant can be added as a sensitivity check.
2. **Wind is stationary.** It is ~1 % by construction in train and 0.8–0.9 % in val/test. Its seasonal peak is in June (monsoon onset, low-level jet).
   Only 0.02–0.03 % of cell-days exceed Beaufort 8 (17.2 m/s), which confirms that box-mean winds cannot represent gales.
3. **IMD criteria at grid scale.** Box-mean Tmax ≥ 40 °C on 7 594 land cell-days in train (max 49.0 °C). Combined with the 4.5 °C departure,
   the IMD-style label is ~6–10× rarer than the percentile label and concentrated in NW India and Pakistan.
4. **Precipitation is stationary** (≈ 2 % in all splits) and strongly seasonal: about 7–8 % of cell-days in July and < 0.5 % from December to May.
5. **Class imbalance** of roughly 1:10 (heat) to 1:100 (wind) needs weighted or focal losses and PR-based metrics. Accuracy is meaningless here.

Full sensitivity grid: `threshold_sensitivity.csv`. Seasonal and monthly prevalence: `monthly_prevalence.csv`. Maps: `label_maps_and_seasonality.png`.
