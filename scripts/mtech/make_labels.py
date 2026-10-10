"""Generate heatwave and extreme-wind labels, sensitivity tables and plots.

    python scripts/mtech/make_labels.py --data_dir regional_climax_dataset_6x6_corrected \
        --label_dir outputs/labels/corrected6x6 --report_dir docs/mtech/results/labels_corrected6x6

Label arrays (large) go to --label_dir; small tables, plots and metadata go to --report_dir.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from climax_hazards import labels as L  # noqa: E402

SPLITS = ("train", "val", "test")


def sensitivity(data_dir, cache):
    rows = []
    for season, months in (("MAMJ", (3, 4, 5, 6)), ("all_year", None)):
        for q in (90, 95, 99):
            for dur in (1, 2, 3, 5):
                sets = L.heatwave_labels(data_dir, L.HeatwaveConfig(percentile=q, min_duration=dur, season_months=months), cache=cache)
                for s in SPLITS:
                    rows.append({"hazard": f"heatwave_pct_{season}", "percentile": q, "min_duration": dur, "split": s,
                                 **L.prevalence(sets["heatwave_pct"].splits[s]["label"])})
    for dur in (1, 2, 3):
        sets = L.heatwave_labels(data_dir, L.HeatwaveConfig(imd_min_duration=dur), cache=cache)
        for s in SPLITS:
            rows.append({"hazard": "heatwave_imd", "min_duration": dur, "split": s, **L.prevalence(sets["heatwave_imd"].splits[s]["label"])})
    for q in (95, 99, 99.9):
        for monthly in (False, True):
            ls = L.wind_labels(data_dir, L.WindConfig(percentile=q, monthly=monthly), cache=cache)
            for s in SPLITS:
                rows.append({"hazard": "extreme_wind", "percentile": q, "monthly": monthly, "split": s, **L.prevalence(ls.splits[s]["label"])})
    for ms in (13.9, 17.2):  # Beaufort 7 (near gale) and 8 (gale) lower bounds
        ls = L.wind_labels(data_dir, L.WindConfig(absolute_ms=ms), cache=cache)
        for s in SPLITS:
            rows.append({"hazard": "extreme_wind_abs", "absolute_ms": ms, "split": s, **L.prevalence(ls.splits[s]["label"])})
    return pd.DataFrame(rows)


def monthly_table(ls):
    rows = []
    for s, d in ls.splits.items():
        m = d["dates"].astype("datetime64[M]").astype(int) % 12 + 1
        for month in range(1, 13):
            lab = d["label"][m == month]
            rows.append({"split": s, "month": month, **L.prevalence(lab)})
    return pd.DataFrame(rows)


def plots(primary, data_dir, report_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat, lon = np.load(f"{data_dir}/lat.npy"), np.load(f"{data_dir}/lon.npy")
    fig, axes = plt.subplots(2, len(primary), figsize=(5 * len(primary), 8))
    for j, ls in enumerate(primary):
        lab = ls.splits["train"]["label"]
        freq = np.nanmean(lab, axis=0) * 100
        ax = axes[0, j]
        im = ax.pcolormesh(lon, lat, freq, shading="nearest", cmap="viridis")
        ax.set_title(f"{ls.name}: event-day frequency, train (%)")
        ax.set_xlabel("lon (deg E)"), ax.set_ylabel("lat (deg N)")
        fig.colorbar(im, ax=ax)
        for (i, k), v in np.ndenumerate(freq):
            ax.text(lon[k], lat[i], "masked" if np.isnan(v) else f"{v:.1f}", ha="center", va="center", fontsize=7, color="w")
        ax = axes[1, j]
        mt = monthly_table(ls)
        for s in SPLITS:
            sub = mt[mt.split == s]
            ax.plot(sub.month, sub.prevalence * 100, marker="o", label=s)
        ax.set_xticks(range(1, 13)), ax.set_xlabel("month"), ax.set_ylabel("prevalence (% of valid cell-days)")
        ax.set_title(f"{ls.name}: seasonal cycle by split"), ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "label_maps_and_seasonality.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(len(primary), 1, figsize=(12, 3 * len(primary)))
    for ax, ls in zip(np.atleast_1d(axes), primary):
        for s in SPLITS:
            d = ls.splits[s]
            yrs = d["dates"].astype("datetime64[Y]").astype(int) + 1970
            u = np.unique(yrs)
            ax.bar(u, [np.nanmean(d["label"][yrs == y]) * 100 for y in u], label=s)
        ax.set_title(f"{ls.name}: annual prevalence (%)"), ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "label_annual_prevalence.png"), dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--report_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    cache = {}
    hw = L.heatwave_labels(args.data_dir, L.HeatwaveConfig(), cache=cache)
    wind = L.wind_labels(args.data_dir, L.WindConfig(), cache=cache)
    primary = [hw["heatwave_pct"], hw["heatwave_imd"], wind]
    meta = {ls.name: L.save_labelset(ls, args.label_dir) for ls in primary}
    json.dump(meta, open(os.path.join(args.report_dir, "label_metadata.json"), "w"), indent=1)

    sens = sensitivity(args.data_dir, cache)
    sens.to_csv(os.path.join(args.report_dir, "threshold_sensitivity.csv"), index=False)
    pd.concat([monthly_table(ls).assign(hazard=ls.name) for ls in primary]).to_csv(
        os.path.join(args.report_dir, "monthly_prevalence.csv"), index=False)

    # IMD-style diagnostics: how close do grid-box Tmax values get to station criteria?
    d = hw["heatwave_imd"].splits["train"]
    mask = L.land_mask(args.data_dir, L.HeatwaveConfig().land_fraction_min)
    tmax = np.where(mask[None], d["value"], np.nan)
    diag = {"train_land_tmax_c_p99": float(np.nanpercentile(tmax, 99)), "train_land_tmax_c_max": float(np.nanmax(tmax)),
            "train_land_cell_days_tmax_ge_40c": int(np.nansum(tmax >= 40)), "train_land_cell_days_tmax_ge_45c": int(np.nansum(tmax >= 45)),
            "train_land_cell_days_departure_ge_4p5c": int(np.nansum(np.where(mask[None], d["departure"], np.nan) >= 4.5)),
            "n_land_cells": int(mask.sum())}
    json.dump(diag, open(os.path.join(args.report_dir, "imd_criteria_diagnostics.json"), "w"), indent=1)

    plots(primary, args.data_dir, args.report_dir)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(json.dumps({k: v["prevalence"] for k, v in meta.items()}, indent=1))
        print(json.dumps(diag, indent=1))
        keys = ["hazard", "percentile", "min_duration", "monthly", "absolute_ms"]
        table = sens.astype({k: object for k in keys}).fillna({k: "-" for k in keys})
        print(table.pivot_table(index=keys, columns="split", values="prevalence", sort=False)[list(SPLITS)].round(4))


if __name__ == "__main__":
    main()
