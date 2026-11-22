"""'Today (live data)' tab: forecast from the latest NOAA GFS analysis (via Open-Meteo), 3 days ahead.

Experimental and outside the validated 2015-2018 evaluation (the model was trained on ERA5); the tab
says so, and shows the recent real-time verification against what actually happened.
"""

import csv
import datetime as dt
import os

import numpy as np
import pandas as pd
import streamlit as st

from climax_hazards import realtime as R
from climax_hazards.demo_backend import WEATHER_VARS
from climax_hazards.viz import DIV, NICE, SEQ, WARM, draw_map, plt

LOG = os.path.join("outputs", "realtime", "forecast_log.csv")


@st.cache_data(ttl=3600, show_spinner="Downloading GFS data from Open-Meteo ...")
def _fetch(lat: tuple, lon: tuple, start: dt.date, end: dt.date):
    return R.fetch(np.array(lat), np.array(lon), start, end)


def _fields(b, data, day):
    return R.fields_at(data, f"{day}T00:00", b.variables, (len(b.lat), len(b.lon)))


@st.cache_data(ttl=6 * 3600, show_spinner="Verifying the last 4 weeks of real-time forecasts ...")
def _recent_skill(_b, today: dt.date, n_days: int = 31):
    start = today - dt.timedelta(days=n_days - 1)
    data = _fetch(tuple(_b.lat), tuple(_b.lon), start, today)
    X = np.stack([_fields(_b, data, start + dt.timedelta(days=i)) for i in range(n_days)])
    return _b.realtime_skill(X, list(WEATHER_VARS)), start


def _log(repo, row):
    p = os.path.join(repo, LOG)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def render(b, repo, have_models):
    st.subheader("Forecast from today's atmosphere (real-time, experimental)")
    st.warning("**Experimental.** Input: NOAA GFS 00 UTC analysis via Open-Meteo, at the 36 grid-cell centres. "
               "The model was trained on ERA5, so this is a domain shift: the validated scores are the 2015-2018 test results. "
               "Only the trained 3-day lead is offered. Needs internet.")
    if not have_models:
        st.info("Live mode needs the model checkpoints.")
        return
    now = dt.datetime.now(dt.timezone.utc)
    today = now.date()
    issue = st.date_input("Issue day (00 UTC)", value=today, min_value=today - dt.timedelta(days=60), max_value=today,
                          help="Today = a real forecast of the future. An earlier day lets you check the forecast against what happened.")
    target = issue + dt.timedelta(days=b.cfg.lead_days)
    try:
        data = _fetch(tuple(b.lat), tuple(b.lon), issue, target)
        x = _fields(b, data, issue)
        ref = _fields(b, data, target)
    except Exception as e:  # network, service or parsing problem: explain, never crash the demo
        st.error(f"Could not get real-time data ({type(e).__name__}: {e}). The 2015-2018 tabs still work offline.")
        return
    out = b.forecast_from_fields(x, np.datetime64(target))
    verified = target <= today
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast issued (00 UTC)", str(issue))
    c2.metric("Valid for", str(target))
    c3.metric("Inference (CPU)", f"{out['seconds']:.2f} s")
    c4.metric("Reference for the target day", "GFS analysis" if verified else "GFS forecast")

    z = R.standardised_anomaly(x, b.mu, b.sd, b.variables)
    outside = [v for v in b.variables if z[v] > b.era5_zmax[v]]
    label = ("Input check: all 17 variables within the range of the ERA5 data" if not outside
             else f"Input check: {len(outside)} variable(s) outside the ERA5 range")
    with st.expander(label):
        st.dataframe(pd.DataFrame({"today max |z|": z, "ERA5 2015-2018 max |z|": b.era5_zmax}).round(1).T, use_container_width=True)
        st.caption("|z| = |value − training mean| / training std, largest over the 36 cells.")

    # ---- hazards ----
    st.markdown(f"#### Hazard probabilities for {target}")
    fig, axes = plt.subplots(1, len(b.hazards), figsize=(11, 3.4), constrained_layout=True)
    im = None
    for ax, (k, h) in zip(axes, enumerate(b.hazards)):
        if not out["applicable"][h]:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{NICE[h]}\nnot defined in {target:%B}\n(heatwave season: March-June)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            continue
        im = draw_map(ax, out["prob"][k], b.lat, b.lon, SEQ, 0, 1, alerts=out["alerts"][k])
        ax.set_title(f"{NICE[h]} (alert ≥ {b.thresholds[h]:.2f})")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, label="probability")
    st.pyplot(fig)
    plt.close(fig)
    n_alerts = {h: int(out["alerts"][k].sum()) for k, h in enumerate(b.hazards)}
    if sum(n_alerts.values()):
        rows = [{"hazard": NICE[h], "lat": float(b.lat[i]), "lon": float(b.lon[j]), "probability": round(float(out["prob"][k, i, j]), 3)}
                for k, h in enumerate(b.hazards) for i, j in zip(*np.nonzero(out["alerts"][k]))]
        st.error(f"{len(rows)} alert(s) for {target}.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.success(f"No alerts for {target}: every probability is below its alert threshold.")

    # ---- weather ----
    st.markdown(f"#### 72 h weather forecast for {target} 00 UTC")
    var = st.selectbox("Variable", list(WEATHER_VARS), key="live_var", index=list(WEATHER_VARS).index("geopotential_500"),
                       format_func=lambda v: f"{WEATHER_VARS[v][0]} ({WEATHER_VARS[v][1]})")
    name, unit = WEATHER_VARS[var]
    i = b.variables.index(var)
    x0, pr, rf = x[i], out["weather_pred"][var], ref[i]
    lo, hi = float(np.min([x0, pr, rf])), float(np.max([x0, pr, rf]))
    fmt = "{:.0f}" if abs(hi) >= 1000 else "{:.1f}"
    cmap = WARM if "temperature" in var else SEQ
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
    ref_name = "GFS analysis (what happened)" if verified else "GFS's own 3-day forecast"
    for ax, f, t in zip(axes[:3], (x0, pr, rf), (f"input {issue}", "ClimaX forecast", ref_name)):
        im = draw_map(ax, f, b.lat, b.lon, cmap, lo, hi, fmt=fmt)
        ax.set_title(t)
    fig.colorbar(im, ax=axes[:3], shrink=0.85, label=unit)
    d = pr - rf
    m = float(np.max(np.abs(d))) or 1.0
    im = draw_map(axes[3], d, b.lat, b.lon, DIV, -m, m, fmt="{:+.1f}" if m < 1000 else "{:+.0f}")
    axes[3].set_title("ClimaX − " + ("analysis" if verified else "GFS forecast"))
    fig.colorbar(im, ax=axes[3], shrink=0.85, label=unit)
    st.pyplot(fig)
    plt.close(fig)
    if verified:
        w = np.cos(np.deg2rad(b.lat))[:, None] * np.ones_like(rf)
        rmse = lambda a: float(np.sqrt((w * (a - rf) ** 2).sum() / w.sum()))
        m1, m2 = st.columns(2)
        m1.metric("ClimaX RMSE vs what happened", f"{rmse(pr):.2f} {unit}")
        m2.metric("Persistence RMSE", f"{rmse(x0):.2f} {unit}")
    else:
        st.caption(f"{target} is in the future: the right-hand maps compare ClimaX with NOAA's operational GFS forecast "
                   "(not with the truth). Pick an issue day at least 3 days ago to verify.")

    # ---- recent real-time verification ----
    with st.expander("How good is the real-time mode? Verification over the last 4 weeks", expanded=False):
        try:
            sk, start = _recent_skill(b, today)
            sk = sk.assign(variable=[WEATHER_VARS[v][0] for v in sk.variable])
            better = ", ".join(sk.loc[sk["improvement_%"] > 0, "variable"]) or "none"
            worse = ", ".join(sk.loc[sk["improvement_%"] <= 0, "variable"]) or "none"
            st.dataframe(sk.round(2), hide_index=True, use_container_width=True)
            st.caption(f"3-day forecasts issued {start} to {today - dt.timedelta(days=b.cfg.lead_days)}, verified against the GFS analysis "
                       f"(lower RMSE is better). Better than persistence: {better}. Not better: {worse}. "
                       "Temperatures suffer most because GFS and ERA5 differ systematically near the surface and below ground over "
                       "the Himalaya/Tibet at this resolution: the ERA5 → GFS domain shift. In the 2015-2018 ERA5 test, ClimaX beats "
                       "persistence on every variable.")
        except Exception as e:
            st.info(f"Verification unavailable ({type(e).__name__}).")

    logged = st.session_state.setdefault("live_logged", set())
    if str(issue) in logged:   # Streamlit reruns the script on every click: log each issue day once per session
        return
    logged.add(str(issue))
    _log(repo, {"logged_utc": now.strftime("%Y-%m-%d %H:%M"), "issue_day": str(issue), "target_day": str(target),
                **{f"max_prob_{h}": (round(float(np.nanmax(out["prob"][k])), 4) if out["applicable"][h] else "") for k, h in enumerate(b.hazards)},
                **{f"alerts_{h}": n_alerts[h] for h in b.hazards}})
