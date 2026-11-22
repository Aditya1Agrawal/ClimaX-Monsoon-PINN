"""Thesis analysis of saved hazard predictions (CPU, no model run).

    python scripts/mtech/make_analysis.py --preds_dir outputs/kaggle_v3/stageB/preds \
        --data_dir regional_climax_dataset_6x6_corrected

Inputs: ``preds/*.npz`` and ``preds/targets.npz`` from run_hazard_experiment.py, and lat/lon of the grid.
Sources: "ClimaX" = mean probability of the frozen heads over the encoder seeds matching
--encoder (default the multi-layer physics encoders A_multi_s*); the no-physics encoders,
logistic regression, climatology and persistence for comparison.

Outputs (--out_dir, --fig_dir):
* overall_test.csv       PR-AUC, ROC-AUC, Brier, Brier skill vs climatology, ECE, raw and calibrated
* calibration.csv        Platt parameters (fitted on validation)
* reliability.csv / reliability.png
* skill_map.csv / skill_maps.png    per-cell PR-AUC, ClimaX vs logistic
* blend.csv / blend_bootstrap.csv   ClimaX + logistic blend, weight chosen on validation
* case_<name>.csv / case_<name>.png  forecasts issued 3 days ahead for real events in the test years
* cases_summary.csv      hits / misses / false alarms at the validation-chosen F1 threshold
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from climax_hazards import analysis as A  # noqa: E402
from climax_hazards import hazard_model as hm  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# colours (validated categorical slots 1-3; neutral grey for the reference)
C_CLIMAX, C_LOGISTIC, C_CAL, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc3ee", "#2a78d6", "#0d3b75"])
DIV = LinearSegmentedColormap.from_list("div", ["#c2481a", "#eb6834", "#e6e5e1", "#2a78d6", "#0d3b75"])
OBS = LinearSegmentedColormap.from_list("obs", ["#f3f2ef", "#3a3936"])
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": "white"})

NICE = {"heatwave": "Heatwave", "extreme_wind": "Extreme wind", "extreme_precip": "Extreme rainfall"}

# Real events in the test years (2015-2018). Points are (lat, lon) of the affected place;
# the nearest 5.625-degree cell centre is used. The labels (not these dates) decide what is an event.
CASES = [
    {"name": "heatwave_2015_andhra_telangana", "hazard": "heatwave", "title": "May 2015 heatwave, Andhra Pradesh / Telangana",
     "points": [(16.5, 79.0), (18.0, 79.5)], "start": "2015-05-05", "end": "2015-06-10"},
    {"name": "rain_2015_chennai", "hazard": "extreme_precip", "title": "Nov-Dec 2015 Chennai floods",
     "points": [(13.08, 80.27)], "start": "2015-11-01", "end": "2015-12-15"},
    {"name": "rain_2017_gujarat", "hazard": "extreme_precip", "title": "July 2017 Gujarat floods",
     "points": [(24.2, 72.4)], "start": "2017-07-05", "end": "2017-08-10"},
    {"name": "wind_2018_cyclone_titli", "hazard": "extreme_wind", "title": "October 2018 Cyclone Titli (Odisha / north Andhra)",
     "points": [(18.9, 84.6)], "start": "2018-10-01", "end": "2018-10-20"},
]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--preds_dir", required=True)
    p.add_argument("--data_dir", required=True, help="dataset with lat.npy / lon.npy")
    p.add_argument("--encoder", default="A_multi", help="prefix of the encoder prediction files averaged as 'ClimaX'")
    p.add_argument("--reference_encoder", default="A_nophys")
    p.add_argument("--lead_days", type=int, default=3)
    p.add_argument("--min_pos", type=int, default=10, help="per-cell PR-AUC only where the test set has this many events")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--out_dir", default="docs/mtech/results/analysis")
    p.add_argument("--fig_dir", default="docs/mtech/figures/analysis")
    return p.parse_args()


def load_mean(preds_dir, prefix):
    files = sorted(glob.glob(os.path.join(preds_dir, f"{prefix}_s*.npz")))
    if not files:
        sys.exit(f"no {prefix}_s*.npz in {preds_dir}")
    arrs = [np.load(f) for f in files]
    return {sp: np.mean([a[sp] for a in arrs], axis=0) for sp in ("val", "test")}, [os.path.basename(f)[:-4] for f in files]


def cell_edges(c):
    d = np.diff(c).mean()
    return np.concatenate([c - d / 2, [c[-1] + d / 2]])


def draw_map(ax, field, lat, lon, cmap, vmin, vmax, annotate=True, fmt="{:.2f}"):
    im = ax.pcolormesh(cell_edges(lon), cell_edges(lat), np.ma.masked_invalid(field), cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolors="white", linewidth=1.5)
    if annotate:
        for i in range(len(lat)):
            for j in range(len(lon)):
                v = field[i, j]
                if np.isfinite(v):
                    frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
                    dark = frac > 0.6 or (cmap is DIV and abs(v) > 0.6 * vmax)
                    ax.text(lon[j], lat[i], fmt.format(v), ha="center", va="center", fontsize=7, color="white" if dark else TEXT)
    ax.set_aspect("equal")
    ax.set_xticks(lon[::2]); ax.set_yticks(lat[::2])
    ax.set_xticklabels([f"{x:.0f}°E" for x in lon[::2]]); ax.set_yticklabels([f"{y:.0f}°N" for y in lat[::2]])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    t = np.load(os.path.join(args.preds_dir, "targets.npz"))
    hazards = json.load(open(os.path.join(args.preds_dir, "..", "run_meta.json")))["hazards"]
    lat = np.load(os.path.join(args.data_dir, "lat.npy"))
    lon = np.load(os.path.join(args.data_dir, "lon.npy"))
    Y = {sp: t[f"{sp}_y"] for sp in ("val", "test")}
    days = A.target_days(t["test_times"], args.lead_days)

    climax, enc_files = load_mean(args.preds_dir, args.encoder)
    ref_enc, _ = load_mean(args.preds_dir, args.reference_encoder)
    base = {s: dict(np.load(os.path.join(args.preds_dir, f"{s}.npz"))) for s in ("logistic", "climatology", "persistence")}
    print("ClimaX = mean of", enc_files)

    # ---- calibration (fitted on validation) ----
    cal, cal_rows = {}, []
    for src, P in (("ClimaX", climax), ("logistic", base["logistic"])):
        cal[src] = {sp: np.zeros_like(P[sp], dtype=np.float64) for sp in ("val", "test")}
        for k, h in enumerate(hazards):
            c = A.PlattCalibrator().fit(P["val"][:, k], Y["val"][:, k])
            for sp in ("val", "test"):
                cal[src][sp][:, k] = c.transform(P[sp][:, k])
            cal_rows.append({"source": src, "hazard": h, "a": c.a, "b": c.b})
    pd.DataFrame(cal_rows).to_csv(os.path.join(args.out_dir, "calibration.csv"), index=False)

    sources = {"ClimaX": climax, "ClimaX (calibrated)": cal["ClimaX"], f"{args.reference_encoder} encoders": ref_enc,
               "logistic": base["logistic"], "logistic (calibrated)": cal["logistic"],
               "climatology": base["climatology"], "persistence": base["persistence"]}

    # ---- blend ClimaX + logistic (calibrated probabilities, weight from validation) ----
    blend_rows, boot_rows = [], []
    blended = {sp: np.zeros_like(cal["ClimaX"][sp]) for sp in ("val", "test")}
    for k, h in enumerate(hazards):
        w, curve = A.choose_blend_weight(cal["ClimaX"]["val"][:, k], cal["logistic"]["val"][:, k], Y["val"][:, k])
        for sp in ("val", "test"):
            blended[sp][:, k] = A.blend(cal["ClimaX"][sp][:, k], cal["logistic"][sp][:, k], w)
        curve["hazard"] = h
        curve["chosen"] = curve["w"] == w
        blend_rows.append(curve)
        for other in ("ClimaX", "logistic"):
            r = hm.paired_block_bootstrap(blended["test"][:, k], sources[other]["test"][:, k], Y["test"][:, k], n_boot=args.n_boot)
            boot_rows.append({"a": f"blend (w_ClimaX={w:.1f})", "b": other, "hazard": h, **r})
    sources["blend ClimaX+logistic"] = blended
    pd.concat(blend_rows).to_csv(os.path.join(args.out_dir, "blend.csv"), index=False)
    bt = pd.DataFrame(boot_rows)
    bt.to_csv(os.path.join(args.out_dir, "blend_bootstrap.csv"), index=False)

    # ---- overall scores ----
    rows = []
    for name, P in sources.items():
        for k, h in enumerate(hazards):
            p, y = P["test"][:, k], Y["test"][:, k]
            pt, yt = hm._flat(p, y)
            from sklearn.metrics import roc_auc_score
            rows.append({"source": name, "hazard": h, "pr_auc": hm.average_precision(p, y), "roc_auc": float(roc_auc_score(yt, pt)),
                         "brier": A.brier(p, y), "brier_skill_vs_clim": A.brier_skill(p, base["climatology"]["test"][:, k], y),
                         "ece": A.reliability(p, y)[1], "prevalence": float(yt.mean()), "n": int(yt.size)})
    ov = pd.DataFrame(rows)
    ov.to_csv(os.path.join(args.out_dir, "overall_test.csv"), index=False)
    with pd.option_context("display.width", 200):
        print("\nTEST scores:\n" + ov.round(4).to_string(index=False))
        print("\nblend vs single sources (paired 7-day block bootstrap):\n" + bt[["a", "b", "hazard", "pr_auc_diff", "ci_low", "ci_high"]].round(4).to_string(index=False))

    # ---- reliability ----
    rel_rows = []
    fig, axes = plt.subplots(1, len(hazards), figsize=(3.3 * len(hazards), 3.4), constrained_layout=True)
    for ax, (k, h) in zip(np.atleast_1d(axes), enumerate(hazards)):
        ax.plot([0, 1], [0, 1], color=C_REF, lw=1, ls=(0, (3, 3)), label="perfect")
        for name, col in (("ClimaX", C_CLIMAX), ("ClimaX (calibrated)", C_CAL), ("logistic", C_LOGISTIC)):
            tab, ece = A.reliability(sources[name]["test"][:, k], Y["test"][:, k])
            tab["source"], tab["hazard"], tab["ece"] = name, h, ece
            rel_rows.append(tab)
            ok = tab["count"] >= 20
            ax.plot(tab.loc[ok, "mean_pred"], tab.loc[ok, "obs_freq"], marker="o", ms=4, lw=2, color=col, label=f"{name} (ECE {ece:.3f})")
        ax.set_title(NICE.get(h, h)); ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(color=GRID, lw=0.6)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Reliability on the test years (2015-2018), bins with ≥ 20 cell-days", fontsize=10)
    fig.savefig(os.path.join(args.fig_dir, "reliability.png"), dpi=200)
    plt.close(fig)
    pd.concat(rel_rows).to_csv(os.path.join(args.out_dir, "reliability.csv"), index=False)

    # ---- per-cell skill maps ----
    map_rows = []
    fig, axes = plt.subplots(len(hazards), 3, figsize=(10, 3.2 * len(hazards)), constrained_layout=True)
    for k, h in enumerate(hazards):
        ap_c, npos = A.per_cell_ap(climax["test"][:, k], Y["test"][:, k], args.min_pos)
        ap_l, _ = A.per_cell_ap(base["logistic"]["test"][:, k], Y["test"][:, k], args.min_pos)
        diff = ap_c - ap_l
        for i in range(len(lat)):
            for j in range(len(lon)):
                map_rows.append({"hazard": h, "lat": lat[i], "lon": lon[j], "n_events": int(npos[i, j]),
                                 "pr_auc_climax": ap_c[i, j], "pr_auc_logistic": ap_l[i, j], "diff": diff[i, j]})
        vmax = np.nanmax([np.nanmax(ap_c), np.nanmax(ap_l)])
        m = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 0.1
        draw_map(axes[k, 0], ap_c, lat, lon, SEQ, 0, vmax)
        im = draw_map(axes[k, 1], ap_l, lat, lon, SEQ, 0, vmax)
        fig.colorbar(im, ax=axes[k, :2], shrink=0.8, label="PR-AUC")
        im = draw_map(axes[k, 2], diff, lat, lon, DIV, -m, m, fmt="{:+.2f}")
        fig.colorbar(im, ax=axes[k, 2], shrink=0.8, label="ClimaX − logistic")
        axes[k, 0].set_title(f"{NICE.get(h, h)}: ClimaX"); axes[k, 1].set_title("logistic"); axes[k, 2].set_title("difference (blue = ClimaX better)")
    fig.suptitle(f"Per-cell test PR-AUC (cells with ≥ {args.min_pos} events; blank = too few events or undefined)", fontsize=10)
    fig.savefig(os.path.join(args.fig_dir, "skill_maps.png"), dpi=200)
    plt.close(fig)
    pd.DataFrame(map_rows).to_csv(os.path.join(args.out_dir, "skill_map.csv"), index=False)

    # ---- case studies (calibrated probabilities; threshold = best F1 on validation) ----
    show = {"ClimaX": cal["ClimaX"]["test"], "logistic": cal["logistic"]["test"], "climatology": base["climatology"]["test"]}
    thr = {h: hm.best_f1_threshold(cal["ClimaX"]["val"][:, k], Y["val"][:, k]) for k, h in enumerate(hazards)}
    thr_l = {h: hm.best_f1_threshold(cal["logistic"]["val"][:, k], Y["val"][:, k]) for k, h in enumerate(hazards)}
    summ = []
    for case in CASES:
        if case["hazard"] not in hazards:
            continue
        k = hazards.index(case["hazard"])
        cells = A.nearest_cells(lat, lon, case["points"])
        ser = A.event_series({n: p[:, k] for n, p in show.items()}, Y["test"][:, k], days, cells, case["start"], case["end"])
        ser.to_csv(os.path.join(args.out_dir, f"case_{case['name']}.csv"), index=False)
        ev = ser["observed"] == 1
        quiet = ser["observed"] == 0
        for src, th in (("ClimaX", thr[case["hazard"]]), ("logistic", thr_l[case["hazard"]])):
            alarm = ser[src] >= th
            summ.append({"case": case["name"], "source": src, "cells": str([(float(lat[i]), float(lon[j])) for i, j in cells]),
                         "event_days": int(ev.sum()), "threshold": th, "hits": int((alarm & ev).sum()), "misses": int((~alarm & ev).sum()),
                         "false_alarms": int((alarm & quiet).sum()), "quiet_days": int(quiet.sum()),
                         "mean_prob_event_days": float(ser.loc[ev, src].mean()) if ev.any() else np.nan,
                         "mean_prob_quiet_days": float(ser.loc[quiet, src].mean()) if quiet.any() else np.nan})

        # figure: time series over the case cells + maps on the day with the most event cells
        win = (days >= np.datetime64(case["start"])) & (days <= np.datetime64(case["end"]))
        idx_win = np.nonzero(win)[0]
        # map day (fixed rule, not chosen by forecast quality): middle of the longest run of event days in the case cells
        ev_flag = ev.values
        best_len, best_mid, run = 0, int(np.argmax(ev_flag)) if ev_flag.any() else len(ev_flag) // 2, 0
        for i_d, f in enumerate(ev_flag):
            run = run + 1 if f else 0
            if run > best_len:
                best_len, best_mid = run, i_d - run // 2
        peak = idx_win[best_mid]
        fig = plt.figure(figsize=(11, 3.6), constrained_layout=True)
        gs = fig.add_gridspec(1, 4, width_ratios=[2.2, 1, 1, 1])
        ax = fig.add_subplot(gs[0])
        x = ser["target_day"].values
        for d in x[ev.values]:
            ax.axvspan(d - np.timedelta64(12, "h"), d + np.timedelta64(12, "h"), color="#e6e5e1", lw=0)
        ax.plot(x, ser["ClimaX"], color=C_CLIMAX, lw=2, label="ClimaX")
        ax.plot(x, ser["logistic"], color=C_LOGISTIC, lw=2, label="logistic")
        ax.plot(x, ser["climatology"], color=C_REF, lw=1.2, ls=(0, (3, 3)), label="climatology")
        ax.axhline(thr[case["hazard"]], color=C_CLIMAX, lw=0.8, ls=":")
        ax.set_ylim(0, 1); ax.set_ylabel("probability (issued 3 days before)")
        ax.set_title(case["title"], loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.text(0.99, 0.97, "grey = observed event day\ndotted = ClimaX alarm threshold", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=MUTED)
        fig.autofmt_xdate()
        pday = str(days[peak])
        for c_i, (title, field, cmap) in enumerate((("observed", Y["test"][peak, k], OBS),
                                                     ("ClimaX", cal["ClimaX"]["test"][peak, k], SEQ),
                                                     ("logistic", cal["logistic"]["test"][peak, k], SEQ))):
            a = fig.add_subplot(gs[c_i + 1])
            fld = np.where(np.isfinite(Y["test"][peak, k]), field, np.nan)
            draw_map(a, fld, lat, lon, cmap, 0, 1, fmt="{:.0f}" if title == "observed" else "{:.2f}")
            for i, j in cells:
                a.add_patch(plt.Rectangle((lon[j] - 2.8125, lat[i] - 2.8125), 5.625, 5.625, fill=False, ec=TEXT, lw=1.4))
            a.set_title(f"{title}, {pday}" if c_i == 0 else title)
        fig.savefig(os.path.join(args.fig_dir, f"case_{case['name']}.png"), dpi=200)
        plt.close(fig)
    cs = pd.DataFrame(summ)
    cs.to_csv(os.path.join(args.out_dir, "cases_summary.csv"), index=False)
    with pd.option_context("display.width", 220):
        print("\ncase studies (threshold = best validation F1):\n" + cs.drop(columns=["cells"]).round(3).to_string(index=False))
    print("\nwrote", args.out_dir, "and", args.fig_dir)


if __name__ == "__main__":
    main()
