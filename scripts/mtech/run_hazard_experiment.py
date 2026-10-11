"""Option A hazard experiment: frozen ClimaX encoders + hazard heads vs baselines.

All sources are scored on the same samples (00 UTC issue, hazard on day D+3), the same
labels and the same splits. Decision thresholds are chosen on val (max F1) and applied to test.

    python scripts/mtech/run_hazard_experiment.py --data_dir <original 6x5 dataset> --imd_dir <IMD folder> \
        --checkpoint physics=<epoch_016.ckpt> --checkpoint nophysics=<lambda0 best.ckpt> \
        --out_dir /kaggle/working/hazard_optionA

Smoke test (fake or real checkpoint, CPU): add --max_samples 300 --epochs 2 --seeds 0
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from climax_hazards import hazard_data as hd  # noqa: E402
from climax_hazards import hazard_model as hm  # noqa: E402
from climax_hazards import inspect_utils as iu  # noqa: E402

SPLITS = ("train", "val", "test")


def score_rows(source, seed, hazards, probs, Y, extra=None):
    """Threshold from val (max F1), then val and test scores per hazard."""
    rows = []
    for k, h in enumerate(hazards):
        thr = hm.best_f1_threshold(probs["val"][:, k], Y["val"][:, k])
        for s in ("val", "test"):
            rows.append({"source": source, "seed": seed, "hazard": h, "split": s, **(extra or {}),
                         **hm.scores(probs[s][:, k], Y[s][:, k], thr)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="unpadded regional dataset (original 6x5 or corrected 6x6)")
    p.add_argument("--imd_dir", default=None)
    p.add_argument("--config", default="configs/regional_india.yaml")
    p.add_argument("--checkpoint", action="append", default=[], help="name=path, repeatable")
    p.add_argument("--hazards", nargs="+", default=list(hd.HAZARDS))
    p.add_argument("--issue_hours", type=int, nargs="+", default=[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--no_logistic", action="store_true")
    p.add_argument("--compare", action="append", default=[],
                   help="A:B source pair for a paired block-bootstrap of test PR-AUC (repeatable)")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--max_samples", type=int, default=None, help="per split, smoke tests only")
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    if "extreme_precip" in args.hazards and not args.imd_dir:
        print("no --imd_dir: dropping extreme_precip")
        args.hazards = [h for h in args.hazards if h != "extreme_precip"]
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))
    variables = cfg["data"]["variables"]
    lead = cfg["data"]["predict_range"] * cfg["data"]["hrs_each_step"]
    if iu.detect_lon_padding(args.data_dir)["looks_padded"]:
        sys.exit("pass the unpadded dataset; padding is applied in memory when the model needs it")
    W = np.load(os.path.join(args.data_dir, "lon.npy")).size

    # ---- labels, samples, targets ----
    labelsets = hd.build_labelsets(args.data_dir, args.imd_dir, args.hazards)
    hazards = list(labelsets)
    T, X, Y = {}, {}, {}
    for s in SPLITS:
        T[s], X[s] = hd.load_inputs(args.data_dir, s, variables, tuple(args.issue_hours))
        if args.max_samples:
            idx = np.linspace(0, len(T[s]) - 1, min(args.max_samples, len(T[s]))).astype(int)
            T[s], X[s] = T[s][idx], X[s][idx]
        Y[s] = hd.aligned_targets(labelsets, s, T[s], lead)
        print(f"{s}: {len(T[s])} samples {T[s][0]} .. {T[s][-1]}, X {X[s].shape}, Y {Y[s].shape}", flush=True)
    prev = {s: {h: float(np.nanmean(Y[s][:, k])) for k, h in enumerate(hazards)} for s in SPLITS}
    print("target prevalence:", json.dumps(prev), flush=True)

    mean = dict(np.load(os.path.join(args.data_dir, "normalize_mean.npz")))
    std = dict(np.load(os.path.join(args.data_dir, "normalize_std.npz")))
    mu = np.concatenate([mean[v] for v in variables])[None, :, None, None].astype(np.float32)
    sd = np.concatenate([std[v] for v in variables])[None, :, None, None].astype(np.float32)
    Xn = {s: (X[s] - mu) / sd for s in SPLITS}

    rows = []
    test_probs = {}  # source -> (N, K, H, W) test probabilities (mean over seeds)
    os.makedirs(os.path.join(args.out_dir, "preds"), exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "preds", "targets.npz"),
                        **{f"{s}_y": Y[s] for s in ("val", "test")}, **{f"{s}_times": T[s].astype("int64") for s in ("val", "test")})

    def keep(source, probs):
        np.savez_compressed(os.path.join(args.out_dir, "preds", f"{source}.npz"), val=probs["val"], test=probs["test"])
        test_probs[source] = probs["test"]

    # ---- baselines ----
    clim = {s: hd.climatology_probs(labelsets, s, T[s], lead) for s in ("val", "test")}
    rows += score_rows("climatology", 0, hazards, clim, Y)
    keep("climatology", clim)
    pers = {s: hd.persistence_probs(labelsets, s, T[s]) for s in ("val", "test")}
    rows += score_rows("persistence", 0, hazards, pers, Y)
    keep("persistence", pers)
    if not args.no_logistic:
        print("logistic regression baseline ...", flush=True)
        lr = hm.logistic_baseline(Xn["train"], Y["train"], Xn["val"], Y["val"], {"test": Xn["test"]})
        rows += score_rows("logistic", 0, hazards, lr, Y)
        keep("logistic", lr)
    print(f"baselines done ({time.time() - t0:.0f}s)", flush=True)

    # ---- frozen encoders + heads ----
    from climax.arch import ClimaX

    net_args = cfg["model"]["net"]["init_args"]
    histories = []
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        net = ClimaX(**net_args)
        ckpt = iu.load_checkpoint(path)
        report = iu.compare_state_dict(net, ckpt["state_dict"])
        if any(report.values()):
            sys.exit(f"{name}: checkpoint incompatible with {args.config}: {json.dumps(report, default=str)[:1500]}")
        net.load_state_dict({k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}, strict=True)
        hp = ckpt.get("hyper_parameters", {})
        meta = {"physics_lambda": hp.get("physics_lambda"), "ckpt_epoch": ckpt.get("epoch")}
        del ckpt
        mw = net_args["img_size"][1]
        if mw not in (W, W + 1):
            sys.exit(f"model width {mw} incompatible with data width {W}")
        pad = mw == W + 1
        print(f"{name}: lambda={meta['physics_lambda']} epoch={meta['ckpt_epoch']} pad={pad} device={device}", flush=True)

        feats = {}
        for s in SPLITS:
            fpath = os.path.join(args.out_dir, "features", f"{name}_{s}_{len(T[s])}.npy")
            if os.path.exists(fpath):
                feats[s] = np.load(fpath)
            else:
                x = iu.pad_lon_edge(Xn[s], W) if pad else Xn[s]
                feats[s] = hm.extract_features(net, x, variables, lead, device)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                np.save(fpath, feats[s])
        del net
        # targets on the model grid: the duplicated column gets NaN (never trained or scored)
        Ym = {s: (np.concatenate([Y[s], np.full(Y[s].shape[:-1] + (1,), np.nan, np.float32)], -1) if pad else Y[s]) for s in SPLITS}
        seed_probs = []
        for seed in args.seeds:
            heads, hist = hm.train_heads(feats["train"], Ym["train"], feats["val"], Ym["val"], hazards,
                                         net_args["img_size"], net_args["patch_size"], seed=seed, epochs=args.epochs,
                                         device=device, log=None)
            histories += [{"source": name, "seed": seed, **h} for h in hist]
            probs = {s: hm.predict(heads, feats[s], device)[..., :W] for s in ("val", "test")}
            rows += score_rows(name, seed, hazards, probs, Y, meta)
            seed_probs.append(probs)
            os.makedirs(os.path.join(args.out_dir, "heads"), exist_ok=True)
            torch.save(heads.state_dict(), os.path.join(args.out_dir, "heads", f"{name}_seed{seed}.pt"))
            best = max(hist, key=lambda r: r["val_ap_mean"])
            print(f"  seed {seed}: best epoch {best['epoch']} val PR-AUC {best['val_ap_mean']:.4f}", flush=True)
        keep(name, {s: np.mean([sp[s] for sp in seed_probs], axis=0) for s in ("val", "test")})

    # ---- outputs ----
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "hazard_results.csv"), index=False)
    pd.DataFrame(histories).to_csv(os.path.join(args.out_dir, "head_training_history.csv"), index=False)
    test = res[res.split == "test"]
    summ = test.groupby(["hazard", "source"]).agg(
        pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), roc_auc=("roc_auc", "mean"), f1=("f1", "mean"),
        precision=("precision", "mean"), recall=("recall", "mean"), prevalence=("prevalence", "first"), n=("n", "first"),
        seeds=("seed", "nunique")).reset_index()
    summ.to_csv(os.path.join(args.out_dir, "hazard_summary_test.csv"), index=False)
    boot = []
    for pair in args.compare:
        a, b = pair.split(":")
        if a not in test_probs or b not in test_probs:
            print(f"skip comparison {pair}: unknown source")
            continue
        for k, h in enumerate(hazards):
            r = hm.paired_block_bootstrap(test_probs[a][:, k], test_probs[b][:, k], Y["test"][:, k], n_boot=args.n_boot)
            boot.append({"a": a, "b": b, "hazard": h, **r})
    if boot:
        bt = pd.DataFrame(boot)
        bt.to_csv(os.path.join(args.out_dir, "bootstrap_test.csv"), index=False)
        print()
        print("Paired block bootstrap, test PR-AUC(a) - PR-AUC(b) [95% CI] (model sources: mean of seed probabilities):")
        print(bt.round(4).to_string(index=False))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    json.dump({"args": vars(args), "git_commit": commit, "lead_hours": lead, "hazards": hazards,
               "n_samples": {s: int(len(T[s])) for s in SPLITS}, "target_prevalence": prev,
               "elapsed_s": round(time.time() - t0)}, open(os.path.join(args.out_dir, "run_meta.json"), "w"), indent=1)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nTEST (thresholds chosen on val):")
        print(summ.round(4).to_string(index=False))
    print(f"saved to {args.out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
