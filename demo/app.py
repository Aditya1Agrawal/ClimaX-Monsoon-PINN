"""Live demo: physics-informed regional ClimaX for 3-day multi-hazard early warning over India.

Run from the repository root:
    streamlit run demo/app.py

Needs (git-ignored, from notebook 07 / joint_training_v2): outputs/kaggle_v3/stageA/A_multi_s*/best_encoder.ckpt,
outputs/kaggle_v3/stageB/{heads,preds}, the corrected 6x6 dataset, and docs/mtech/results/analysis/calibration.csv.
Without the checkpoints it falls back to the saved predictions (same numbers, no live inference).
Paths can be overridden with the environment variables CLIMAX_DEMO_RUN_DIR and CLIMAX_DEMO_DATA_DIR.
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from climax_hazards.analysis import CASES  # noqa: E402
from climax_hazards.demo_backend import WEATHER_VARS, DemoBackend, DemoConfig  # noqa: E402
import time  # noqa: E402

from climax_hazards.analysis import nearest_cells  # noqa: E402
from climax_hazards.viz import C_CLIMAX, C_LOGISTIC, C_REF, DIV, GRID, MUTED, NICE, OBS, SEQ, WARM, apply_style, draw_map, plt  # noqa: E402

apply_style()
st.set_page_config(page_title="ClimaX India hazard warning", layout="wide")


@st.cache_resource(show_spinner="Loading test data and saved predictions ...")
def get_backend():
    cfg = DemoConfig(repo=REPO, run_dir=os.environ.get("CLIMAX_DEMO_RUN_DIR", "outputs/kaggle_v3"),
                     data_dir=os.environ.get("CLIMAX_DEMO_DATA_DIR", "regional_climax_dataset_6x6_corrected"))
    return DemoBackend(cfg)


@st.cache_resource(show_spinner="Loading the ClimaX encoders and hazard heads (under a minute, once) ...")
def get_models(_b):
    _b.load_models()
    return True


def map_day(b, case):
    """Issue day for a preset: middle of the longest run of event days in the case cells (same rule as the thesis figures)."""
    ser = b.event(case)
    ev = (ser["observed"] == 1).to_numpy()
    best_len, best_mid, run = 0, 0, 0
    for i, f in enumerate(ev):
        run = run + 1 if f else 0
        if run > best_len:
            best_len, best_mid = run, i - run // 2
    return pd.Timestamp(ser["issue_day"].iloc[best_mid]).date()


def _replay_figure(b, rc, ser, day, cells, dpi=100):
    """Time series of the case window (marker on ``day``) + observed / ClimaX / logistic maps for ``day``."""
    lat, lon = b.lat, b.lon
    k = b.hazards.index(rc["hazard"])
    fig = plt.figure(figsize=(12, 3.6), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[2.3, 1, 1, 1])
    ax = fig.add_subplot(gs[0])
    x = ser["target_day"].values
    for d in x[(ser["observed"] == 1).values]:
        ax.axvspan(d - np.timedelta64(12, "h"), d + np.timedelta64(12, "h"), color="#e6e5e1", lw=0)
    ax.plot(x, ser["ClimaX"], color=C_CLIMAX, lw=2, label="ClimaX")
    ax.plot(x, ser["logistic"], color=C_LOGISTIC, lw=2, label="logistic")
    ax.plot(x, ser["climatology"], color=C_REF, lw=1.2, ls=(0, (3, 3)), label="climatology")
    ax.axhline(b.thresholds[rc["hazard"]], color=C_CLIMAX, lw=0.8, ls=":")
    ax.axvline(np.datetime64(str(day)), color=MUTED, lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel("probability (issued 3 days before)")
    ax.set_title(rc["title"], loc="left"); ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.text(0.99, 0.97, "grey = observed event day\ndotted = alert threshold", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
    fig.autofmt_xdate()
    idx = int(np.nonzero(b.target_days == np.datetime64(str(day)))[0][0])
    obs = b.y["test"][idx, k]
    valid = np.isfinite(obs)
    al = (b.saved["test"][idx, k] >= b.thresholds[rc["hazard"]]) & valid
    for c_i, (title, fld, cmap, fmt, alerts) in enumerate((("observed", obs, OBS, "{:.0f}", None),
                                                           ("ClimaX", np.where(valid, b.saved["test"][idx, k], np.nan), SEQ, "{:.2f}", al),
                                                           ("logistic", np.where(valid, b.logistic["test"][idx, k], np.nan), SEQ, "{:.2f}", None))):
        a = fig.add_subplot(gs[c_i + 1])
        draw_map(a, fld, lat, lon, cmap, 0, 1, fmt=fmt, outline=cells, alerts=alerts)
        a.set_title(f"{title}, {day}" if c_i == 0 else title)
    return fig


b = get_backend()
lat, lon = b.lat, b.lon
have_models = b.models_available()

# ---------------- sidebar ----------------
st.sidebar.title("Forecast settings")
presets = {c["title"]: c for c in CASES}
mode = st.sidebar.radio("Choose", ["Real event (preset)", "Any test day"], index=0)
if mode == "Real event (preset)":
    case = presets[st.sidebar.selectbox("Event", list(presets))]
    issue = map_day(b, case)
    st.sidebar.caption(f"Issue day chosen by a fixed rule: middle of the longest run of observed event days ({case['hazard']}).")
else:
    case = None
    issue = st.sidebar.date_input("Issue day (forecast made at 00 UTC)", value=pd.Timestamp("2015-05-22").date(),
                                  min_value=pd.Timestamp("2015-01-01").date(), max_value=pd.Timestamp("2018-12-28").date())
live = st.sidebar.toggle("Run the model live", value=have_models, disabled=not have_models,
                         help="Off: use the predictions saved by the Kaggle run (identical numbers).")
if not have_models:
    st.sidebar.warning("Checkpoints not found: showing saved predictions.")
st.sidebar.markdown("---")
st.sidebar.caption("ClimaX ViT (107 M parameters) fine-tuned on ERA5 5.625° over India with a multi-layer hypsometric "
                   "physics loss; frozen hazard heads. Test years 2015-2018 were never used for training or tuning.")

if live:
    get_models(b)
try:
    fc = b.forecast(np.datetime64(str(issue)), live=live)
except KeyError as e:
    st.error(str(e))
    st.stop()

# ---------------- header ----------------
st.title("3-day multi-hazard early warning for India")
st.markdown("Heatwave · extreme wind · extreme rainfall, from one physics-informed ClimaX foundation model")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast issued (00 UTC)", str(fc.issue_day))
c2.metric("Valid for", str(fc.target_day))
c3.metric("Live inference (CPU)" if fc.live else "Source", f"{fc.seconds:.2f} s" if fc.live else "saved")
c4.metric("Matches Kaggle run", f"Δ {fc.max_abs_diff_vs_saved:.0e}" if fc.max_abs_diff_vs_saved is not None else "n/a",
          help="Largest difference between this live forecast and the probabilities saved by the Kaggle run")

tabs = st.tabs(["Hazard warning", "Weather forecast", "Physics check", "Event replay", "Today (live data)", "Validated results", "About"])

# ---------------- hazard warning ----------------
with tabs[0]:
    st.subheader(f"Hazard probabilities for {fc.target_day} (issued 3 days earlier)")
    st.caption("Probabilities are calibrated on 2011-2014. Red frame = alert (probability ≥ the threshold with the best F1 on 2011-2014). "
               "Blank cells: hazard not defined there (heatwave: land only; rainfall: IMD boxes with ≥ 50 % land coverage).")
    for k, h in enumerate(fc.hazards):
        valid = np.isfinite(fc.observed[k])
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
        draw_map(axes[0], fc.observed[k], lat, lon, OBS, 0, 1, fmt="{:.0f}")
        axes[0].set_title(f"{NICE[h]}: observed")
        im = draw_map(axes[1], np.where(valid, fc.prob[k], np.nan), lat, lon, SEQ, 0, 1, alerts=fc.alerts[k])
        axes[1].set_title(f"ClimaX (alert ≥ {fc.thresholds[h]:.2f})")
        draw_map(axes[2], np.where(valid, fc.prob_logistic[k], np.nan), lat, lon, SEQ, 0, 1)
        axes[2].set_title("logistic regression baseline")
        fig.colorbar(im, ax=axes[1:], shrink=0.85, label="probability")
        st.pyplot(fig)
        plt.close(fig)
    cA, cB = st.columns(2)
    cA.markdown("**Verification of the alerts**")
    cA.dataframe(b.verification(fc), hide_index=True, use_container_width=True)
    cB.markdown("**Alert list**")
    al = b.alert_table(fc)
    cB.dataframe(al.round(3) if len(al) else pd.DataFrame({"": ["no alerts"]}), hide_index=True, use_container_width=True)

# ---------------- weather ----------------
with tabs[1]:
    st.subheader(f"72 h weather forecast for {fc.target_day} 00 UTC")
    if fc.weather_pred is None:
        st.info("Weather forecasts need live inference (turn on 'Run the model live').")
    elif fc.weather_true is None:
        st.info("No ERA5 truth for this target day in the test data.")
    else:
        var = st.selectbox("Variable", list(WEATHER_VARS), format_func=lambda v: f"{WEATHER_VARS[v][0]} ({WEATHER_VARS[v][1]})")
        name, unit = WEATHER_VARS[var]
        x0, pr, tr = fc.weather_input[var], fc.weather_pred[var], fc.weather_true[var]
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(tr)
        rmse = lambda a: float(np.sqrt(np.sum(w * (a - tr) ** 2) / w.sum()))
        m1, m2, m3 = st.columns(3)
        m1.metric("ClimaX RMSE (this forecast)", f"{rmse(pr):.2f} {unit}")
        m2.metric("Persistence RMSE", f"{rmse(x0):.2f} {unit}", help="'Tomorrow = today': the issue-day field used as the forecast")
        m3.metric("Improvement", f"{100 * (1 - rmse(pr) / rmse(x0)):.0f} %")
        lo, hi = float(np.min([x0, pr, tr])), float(np.max([x0, pr, tr]))
        fmt = "{:.0f}" if abs(hi) >= 1000 else "{:.1f}"
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
        cmap = WARM if "temperature" in var else SEQ
        for ax, f, t in zip(axes[:3], (x0, pr, tr), (f"input {fc.issue_day}", "ClimaX forecast", f"ERA5 truth {fc.target_day}")):
            im = draw_map(ax, f, lat, lon, cmap, lo, hi, fmt=fmt)
            ax.set_title(t)
        fig.colorbar(im, ax=axes[:3], shrink=0.85, label=unit)
        err = pr - tr
        m = float(np.max(np.abs(err))) or 1.0
        im = draw_map(axes[3], err, lat, lon, DIV, -m, m, fmt="{:+.1f}" if m < 1000 else "{:+.0f}")
        axes[3].set_title("forecast − truth")
        fig.colorbar(im, ax=axes[3], shrink=0.85, label=unit)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Forecast = mean of the three grid-adapted encoders. Test-set RMSE over 2015-2018 is in 'Validated results'.")

# ---------------- physics ----------------
with tabs[2]:
    st.subheader("Is the forecast physically consistent? (hypsometric equation)")
    st.latex(r"R = \frac{\Phi_{\mathrm{top}} - \Phi_{\mathrm{bottom}}}{g} - \frac{R_d}{g}\,\bar T \,\ln\frac{p_{\mathrm{bottom}}}{p_{\mathrm{top}}}")
    st.markdown("The thickness of each pressure layer must match its mean temperature. We train on the error of $R$ "
                "on **every adjacent layer** (925 → 50 hPa), each scaled separately. Lower = more consistent with physics.")
    if not fc.live:
        st.info("Needs live inference.")
    else:
        pe = b.physics_errors(fc.issue_day)
        if pe is None or pe.empty:
            st.info("No truth for this target day.")
        else:
            piv = pe.pivot(index="layer", columns="model", values="abs_error_m").loc[[f"{u}-{l} hPa" for u, l in
                  ((850, 925), (700, 850), (600, 700), (500, 600), (250, 500), (50, 250))]]
            low = piv.iloc[:4]
            fig, ax = plt.subplots(figsize=(7, 2.8), constrained_layout=True)
            yy = np.arange(len(low))
            cols = list(low.columns)
            for n, (cname, colr) in enumerate(zip(cols, (C_CLIMAX, C_REF))):
                ax.barh(yy + (n - 0.5) * 0.38, low[cname], height=0.34, color=colr, label=cname)
            ax.set_yticks(yy, low.index); ax.invert_yaxis(); ax.set_xlabel("mean |residual error| for this forecast (m)")
            ax.grid(axis="x", color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8)
            cl, cr = st.columns([3, 2])
            cl.pyplot(fig)
            plt.close(fig)
            cr.dataframe(piv.round(2), use_container_width=True)
            cr.caption("This forecast only, encoder A_multi_s0 vs A_nophys_s0 (same seed). Single days vary; the whole-period table below is the evidence.")
    p = os.path.join(REPO, "docs/mtech/results/multilayer_v3/stageA_weather_all.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d[(d.source == "model") & d.variable.str.startswith("hyps_resid")]
        t = d.groupby(["variable", "variant"]).rmse.mean().unstack("variant").rename(
            columns={"multi": "multi-layer physics", "nophys": "no physics", "phys": "single layer (internship)"})
        order = ["850_925", "700_850", "600_700", "500_600", "250_500", "50_250", "500_850"]
        t = t.loc[[f"hyps_resid_{o}" for o in order if f"hyps_resid_{o}" in t.index]]
        t.index = [v.replace("hyps_resid_", "").replace("_", "-") + " hPa" + (" (two-level T̄, old metric)" if v.endswith("500_850") else "")
                   for v in t.index]
        st.markdown("**Whole test period 2015-2018** (RMSE of the residual, m, mean of 3 seeds):")
        st.dataframe(t.round(2), use_container_width=True)
        st.caption("The internship's single-layer loss made the lower layers much worse (×2.4 at 850-925 hPa); "
                   "the multi-layer loss is the most consistent model in the lower troposphere at no weather cost.")

# ---------------- replay ----------------
with tabs[3]:
    st.subheader("Replay a real event day by day")
    rc = presets[st.selectbox("Event", list(presets), index=list(presets).index(case["title"]) if case else 0, key="replay")]
    ser = b.event(rc)
    k = b.hazards.index(rc["hazard"])
    days = [pd.Timestamp(d).date() for d in ser["target_day"]]
    tday = pd.Timestamp(str(fc.target_day)).date()
    default = tday if case is not None and rc is case and tday in days else days[len(days) // 2]
    day = st.select_slider("Target day", options=days, value=default)
    play = st.button("▶ Play the event", help="Steps through every day of the window")
    slot = st.empty()
    cells = nearest_cells(lat, lon, rc["points"])

    def replay_figure(d, dpi=100):
        return _replay_figure(b, rc, ser, d, cells, dpi)

    if play:
        for d in days:
            f_ = replay_figure(d, dpi=60)
            slot.pyplot(f_, dpi=60)
            plt.close(f_)
            time.sleep(0.05)
    f_ = replay_figure(day)
    slot.pyplot(f_)
    plt.close(f_)
    cs_p = os.path.join(REPO, "docs/mtech/results/analysis/cases_summary.csv")
    if os.path.exists(cs_p):
        cs = pd.read_csv(cs_p)
        cs = cs[cs.case == rc["name"]][["source", "event_days", "hits", "misses", "false_alarms", "quiet_days"]]
        st.markdown("**Whole event window** (alerts in the boxed cells):")
        st.dataframe(cs, hide_index=True, use_container_width=True)


# ---------------- live ----------------
with tabs[4]:
    import live_tab

    live_tab.render(b, REPO, have_models and live)

# ---------------- results ----------------
with tabs[5]:
    st.subheader("Validated results on the test years 2015-2018")
    p = os.path.join(REPO, "docs/mtech/results/analysis/overall_test.csv")
    if os.path.exists(p):
        ov = pd.read_csv(p)
        keep = ["blend ClimaX+logistic", "ClimaX", "logistic", "persistence", "climatology"]
        t = ov[ov.source.isin(keep)].pivot_table(index="source", columns="hazard", values="pr_auc").reindex(keep)
        t.columns = [NICE.get(c, c) for c in t.columns]
        st.markdown("**PR-AUC** (area under the precision-recall curve; higher is better; climatology ≈ event frequency)")
        st.dataframe(t.round(3), use_container_width=True)
        st.caption("ClimaX beats logistic regression on extreme rainfall by +0.06 PR-AUC (significant in all seeds, 7-day block "
                   "bootstrap). The blend with logistic regression is the best system for every hazard.")
    wp = os.path.join(REPO, "docs/mtech/results/multilayer_v3/stageA_weather_all.csv")
    if os.path.exists(wp):
        d = pd.read_csv(wp)
        keyv = ["2m_temperature", "10m_u_component_of_wind", "temperature_850", "geopotential_500"]
        d = d[d.variable.isin(keyv)]
        mod = d[d.source == "model"].groupby(["variable", "variant"]).rmse.mean().unstack("variant")
        ref = d[d.source.isin(["climatology", "persistence"])].groupby(["variable", "source"]).rmse.first().unstack("source")
        wt = pd.concat([mod.rename(columns={"multi": "ClimaX multi-layer physics", "nophys": "ClimaX no physics",
                                            "phys": "ClimaX single layer"}), ref], axis=1).reindex(keyv)
        wt.index = [WEATHER_VARS[v][0] + f" ({WEATHER_VARS[v][1]})" for v in wt.index]
        st.markdown("**72 h weather forecast RMSE**, test years (lower is better; mean of 3 training seeds)")
        st.dataframe(wt.round(2), use_container_width=True)
    for f, cap in (("reliability.png", "Calibration: raw probabilities are over-confident; Platt scaling fitted on 2011-2014 fixes it."),
                   ("skill_maps.png", "Where ClimaX beats logistic regression (blue).")):
        fp = os.path.join(REPO, "docs/mtech/figures/analysis", f)
        if os.path.exists(fp):
            st.image(fp, caption=cap, use_container_width=True)

# ---------------- about ----------------
with tabs[6]:
    st.subheader("How it works")
    st.markdown("""
1. **Input**: ERA5 reanalysis at 00 UTC: 17 variables (2 m temperature, 10 m wind, temperature and geopotential at 7 levels) on a 6 × 6 grid (5.625°) over India.
2. **Encoder**: ClimaX vision transformer (107 M parameters, 8 blocks): variable tokenisation, cross-attention aggregation, lead-time embedding (72 h).
   Fine-tuned on 1979-2010 with a **multi-layer hypsometric physics loss**, balanced against the forecast loss by uncertainty weighting.
3. **Heads**: the weather head (17 variables at +72 h) and three light **hazard heads** on the frozen encoder tokens.
4. **Labels**: heatwave = ≥ 3 days above the calendar-day 90th percentile of max temperature (March-June, land); extreme wind = daily max wind above the 99th percentile;
   extreme rainfall = IMD gridded rain above the 95th percentile of wet days. All thresholds from training years only.
5. **Calibration and alert thresholds**: fitted on 2011-2014. Everything shown here is 2015-2018, never used for training or tuning.

**Limitations**: at 5.625° one grid cell is about 600 km, so tropical cyclones (e.g. Titli 2018) and coastal states like Kerala are not resolved;
wind skill is low for all models; results are from one region and one lead time.
""")
