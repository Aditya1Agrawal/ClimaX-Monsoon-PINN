"""Evaluate 72 h forecasts against persistence and train climatology on identical samples.

Baselines only (CPU, local or Kaggle):
    python scripts/mtech/evaluate_forecast.py --data_dir regional_climax_dataset --out outputs/eval_baselines.csv

With a checkpoint trained by the internship setup (6x6 padded grid; padding applied in memory):
    python scripts/mtech/evaluate_forecast.py --data_dir <unpadded 6x5 dataset> \
        --checkpoint <epoch_016.ckpt> --config configs/regional_india.yaml --out eval_epoch016.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from climax_hazards import forecast_eval as fe  # noqa: E402
from climax_hazards import inspect_utils as iu  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--config", default="configs/regional_india.yaml", help="variables, lead time and architecture")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--max_files", type=int, default=None, help="limit shards (smoke tests only)")
    p.add_argument("--doy_window", type=int, default=7)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    variables = cfg["data"]["variables"]
    out_variables = cfg["data"]["out_variables"]
    lead = cfg["data"]["predict_range"] * cfg["data"]["hrs_each_step"]
    lat = np.load(os.path.join(args.data_dir, "lat.npy"))
    n_lon = np.load(os.path.join(args.data_dir, "lon.npy")).size
    if iu.detect_lon_padding(args.data_dir)["looks_padded"]:
        sys.exit("data_dir is already padded; pass the original unpadded dataset")

    t0 = time.time()
    clim = fe.train_climatology(args.data_dir, variables, args.doy_window)
    print(f"train climatology {clim.shape} in {time.time() - t0:.0f}s", flush=True)

    models, column_sets = {}, {"real": slice(None)}
    if args.checkpoint:
        import torch

        from climax.arch import ClimaX

        net_args = cfg["model"]["net"]["init_args"]
        model = ClimaX(**net_args)
        ckpt = iu.load_checkpoint(args.checkpoint)
        report = iu.compare_state_dict(model, ckpt["state_dict"])
        if any(report.values()):
            sys.exit(f"checkpoint incompatible with {args.config}: {json.dumps(report, default=str)[:2000]}")
        model.load_state_dict({k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}, strict=True)
        del ckpt
        mean = dict(np.load(os.path.join(args.data_dir, "normalize_mean.npz")))
        std = dict(np.load(os.path.join(args.data_dir, "normalize_std.npz")))
        width = net_args["img_size"][1]
        pad_to = width if width == n_lon + 1 else None
        if width not in (n_lon, n_lon + 1):
            sys.exit(f"model width {width} incompatible with data width {n_lon}")
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        name = os.path.basename(args.checkpoint)
        models[name] = fe.climax_predictor(model, variables, out_variables, mean, std, lead, lat, device=device, pad_to=pad_to)
        if pad_to:
            column_sets = {"real": slice(0, n_lon), "all_incl_dup_col": slice(None)}
        print(f"model {name} on {device}, pad_to={pad_to}", flush=True)

    rows = fe.evaluate(args.data_dir, args.split, variables, out_variables, lead, clim, models, column_sets, args.max_files)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    pivot = df[df.columns.intersection(["source", "columns", "variable", "rmse"])].pivot_table(
        index="variable", columns=["source", "columns"], values="rmse", sort=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"\nlat-weighted RMSE, split={args.split}, lead={lead} h, n={df['n'].iloc[0]}")
        print(pivot.round(3))
    print(f"saved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
