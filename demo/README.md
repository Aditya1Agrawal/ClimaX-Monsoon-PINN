# Live demo: 3-day multi-hazard early warning for India

A Streamlit app that runs the trained physics-informed ClimaX model **live on CPU**.
- **Input:** a test day (2015–2018).
- **Output:** heatwave, extreme-wind and extreme-rainfall warnings for 3 days later, plus the 72 h weather forecast. Each is compared with what actually happened.

## Setup (once)
Run from the repository root in PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File demo\setup_env.ps1
```
This creates `.venv` (Python 3.13, CPU PyTorch), installs `demo/requirements.txt`, and patches timm 0.6.12 for Python ≥ 3.11.

The app also needs these files. They are git-ignored, so they are not in the repo:
| Path | From |
|---|---|
| `regional_climax_dataset_6x6_corrected/` | corrected dataset |
| `outputs/kaggle_v3/stageA/A_multi_s{0,1,2}/best_encoder.ckpt` | notebook 07 output (`multilayer-physics`) |
| `outputs/kaggle_v3/stageB/heads/`, `outputs/kaggle_v3/stageB/preds/`, `outputs/kaggle_v3/stageB/run_meta.json` | notebook 07 output |
| `outputs/kaggle_v2/stageA/A_nophys_s0/best_encoder.ckpt` | joint_training_v2 output (physics comparison only; optional) |

`docs/mtech/results/` and `docs/mtech/figures/` are in git.
- Without the checkpoints, the app falls back to the saved Kaggle predictions. The numbers are identical, but there is no live weather or physics tab.
- To use other locations, set `CLIMAX_DEMO_RUN_DIR` / `CLIMAX_DEMO_DATA_DIR`.

## Run
```powershell
.\.venv\Scripts\python.exe -m streamlit run demo\app.py
```
It opens at http://localhost:8501.
- The first load takes about 1 minute (data plus 4 encoders of 107 M parameters each).
- After that, each forecast takes under 1 s.
- Start it **before** the meeting and leave it running.

Check that everything works (50 tests, including "live inference = Kaggle predictions"):
```powershell
$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m pytest -q tests\mtech
```

## What each tab shows
| Tab | What it shows |
|---|---|
| Hazard warning | Observed vs ClimaX vs logistic maps for the target day. Red frame = alert. Verification table (hits / misses / false alarms). |
| Weather forecast | Input, forecast, ERA5 truth and error maps; RMSE against persistence for this forecast. |
| Physics check | Hypsometric residual error per pressure layer: multi-layer physics vs no physics (this forecast, and the whole test period). |
| Event replay | Day-by-day probabilities through a real event; **▶ Play** animates it. |
| Today (live data) | **Real-time, experimental.** Today's 00 UTC NOAA GFS analysis (via Open-Meteo, at the 36 cell centres) → hazard probabilities and weather for 3 days ahead, next to GFS's own forecast. With an earlier issue day it verifies against what happened. A 4-week verification table shows the real-time skill. Needs internet. |
| Validated results | Test PR-AUC table, weather RMSE table, reliability diagram, skill maps. |
| About | Pipeline, labels, limitations. |

## If they ask "can it forecast from today?"
Open **Today (live data)**.
- Show today's forecast for 3 days ahead, and how it compares with NOAA's operational GFS forecast.
- Explain that it is **experimental**: the model was trained on ERA5, and the live input is GFS.
- Open "How good is the real-time mode?". Over the last 4 weeks, 500 hPa geopotential and 10 m winds beat persistence by about 20 %. Temperatures do not, because GFS and ERA5 differ systematically near the surface and below ground over the Himalaya/Tibet at 5.6° (domain shift).
- The validated scores remain the 2015–2018 ERA5 test, where ClimaX beats persistence on all 17 variables.
- Heatwave alerts are off outside March–June, by the label definition.
- Each live forecast is logged in `outputs/realtime/forecast_log.csv`, so it can be checked once the target day has passed.

## Suggested 5-minute script
1. **Problem** (30 s): "One foundation model, three hazards, 3-day warning, over India."
2. **May 2015 heatwave** (preset, Hazard warning tab, 1 min):
   - The model runs live in under 1 s.
   - Alerts (red frames) cover both observed heatwave cells over Andhra Pradesh/Telangana, 3 days ahead.
   - Point at the verification table: 2 hits, 2 false alarms on this day.
3. **Event replay → ▶ Play** (1 min): the probability rises above the threshold as the heatwave starts and falls after it ends. Over the window: 9 of 13 event days caught, 2 false alarms.
4. **Chennai floods** (preset, replay, 45 s): ClimaX catches 5 of 8 rain days; logistic regression catches 1.
5. **Physics check** (45 s):
   - The hypsometric equation, applied on every layer.
   - Whole test period: the internship single-layer loss made the lower layers 2.4× worse; the multi-layer loss is the most consistent model.
6. **Validated results** (45 s):
   - Rainfall PR-AUC 0.227 vs 0.165 for logistic regression (significant).
   - Calibrated probabilities.
   - The blend is best for every hazard.
7. **Honest limitation** (30 s), with the Cyclone Titli preset: missed by every model, because one 5.6° cell is about 600 km. This motivates the higher-resolution future work.

All numbers on screen are from the test years 2015–2018, which were never used for training, calibration or threshold selection.
