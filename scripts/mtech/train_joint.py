"""Fine-tune ClimaX jointly for weather, physics and hazards with adaptive loss balancing.

    python scripts/mtech/train_joint.py --data_dir <corrected 6x6 dataset> --imd_dir <IMD folder> \
        --init_ckpt <lambda0 epoch_016.ckpt> --out_dir /kaggle/working/joint_uw

Smoke test (CPU ok):  add --smoke
Ablation with the internship's fixed weights:  --balancing fixed --fixed_physics 0.001
Compare with a frozen-encoder run of run_hazard_experiment.py on the same dataset:
    --compare_dir <its out_dir> --compare_sources nophysics_e016 ...

Outputs: best.pt, history.csv, hazard_results.csv, weather_test.csv, preds/joint.npz,
bootstrap_vs_frozen.csv (if --compare_dir), run_meta.json.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from climax_hazards import forecast_eval as fe  # noqa: E402
from climax_hazards import hazard_data as hd  # noqa: E402
from climax_hazards import hazard_model as hm  # noqa: E402
from climax_hazards import inspect_utils as iu  # noqa: E402
from climax_hazards import joint as J  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def predict(model, data, idx, variables, lead, device, amp, bs=256):
    model.eval()
    ws, hs = [], []
    for i in range(0, len(idx), bs):
        x, _, _ = data.batch(idx[i:i + bs])
        x = x.to(device)
        lt = torch.full((len(x),), lead / 100.0, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            w, h = model(x, lt, variables)
        ws.append(w.float().cpu().numpy())
        if h is not None:
            hs.append(torch.sigmoid(h.float()).cpu().numpy())
    return np.concatenate(ws), (np.concatenate(hs) if hs else None)


def evaluate_val(model, val, variables, lead, device, amp, w_lat_np, std, weather_idx, hz_idx):
    w, _ = predict(model, val, weather_idx, variables, lead, device, amp)
    _, y, _ = val.batch(weather_idx)
    y = y.numpy()
    err = ((w - y) ** 2 * w_lat_np).mean(axis=(0, 2, 3))           # per variable, normalised units
    out = {"val_w_mse": float(err.mean())}
    rmse_phys = np.sqrt(err) * std.reshape(-1)
    for v in ("geopotential_500", "temperature_850", "2m_temperature", "10m_u_component_of_wind"):
        out[f"val_rmse_{v}"] = float(rmse_phys[variables.index(v)])
    if val.haz is not None:
        _, p = predict(model, val, hz_idx, variables, lead, device, amp)
        aps = [hm.average_precision(p[:, k], val.haz[hz_idx, k]) for k in range(p.shape[1])]
        out.update({f"val_ap_{h}": a for h, a in zip(val.hazards, aps)})
        out["val_ap_mean"] = float(np.nanmean(aps))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--imd_dir", default=None)
    p.add_argument("--config", default="configs/regional_india.yaml")
    p.add_argument("--init_ckpt", default=None, help="Lightning checkpoint to start from (strict load)")
    p.add_argument("--from_scratch", action="store_true")
    p.add_argument("--hazards", nargs="+", default=list(hd.HAZARDS))
    p.add_argument("--balancing", choices=["uncertainty", "fixed"], default="uncertainty")
    p.add_argument("--fixed_physics", type=float, default=0.001, help="fixed mode: physics weight (raw m^2 loss scale, as in the internship)")
    p.add_argument("--fixed_hazard", type=float, default=1.0)
    p.add_argument("--no_physics", action="store_true", help="drop the physics task (ablation)")
    p.add_argument("--physics_scale_m", type=float, default=18.0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--samples_per_epoch", type=int, default=65536)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr_encoder", type=float, default=1e-4)
    p.add_argument("--lr_heads", type=float, default=5e-4)
    p.add_argument("--lr_weights", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val_weather_stride", type=int, default=6, help="use every n-th validation pair for weather loss")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny run for pipeline checks")
    p.add_argument("--tiny_model", action="store_true", help="small ClimaX for local CPU tests (not for results)")
    p.add_argument("--weather_eval_max_files", type=int, default=None)
    p.add_argument("--compare_dir", default=None, help="out_dir of run_hazard_experiment.py on the same data_dir")
    p.add_argument("--compare_sources", nargs="*", default=[])
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    if args.smoke:
        args.epochs, args.samples_per_epoch, args.warmup_steps = 2, 512, 10
        args.weather_eval_max_files = args.weather_eval_max_files or 1
    if "extreme_precip" in args.hazards and not args.imd_dir:
        log("no --imd_dir: dropping extreme_precip")
        args.hazards = [h for h in args.hazards if h != "extreme_precip"]
    os.makedirs(os.path.join(args.out_dir, "preds"), exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda" and not args.no_amp
    t0 = time.time()

    # ---- config, data, labels ----
    cfg = yaml.safe_load(open(args.config))
    net_args = dict(cfg["model"]["net"]["init_args"])
    variables = list(cfg["data"]["variables"])
    assert list(net_args["default_vars"]) == variables and list(cfg["data"]["out_variables"]) == variables
    lead = cfg["data"]["predict_range"] * cfg["data"]["hrs_each_step"]
    if iu.detect_lon_padding(args.data_dir)["looks_padded"]:
        sys.exit("pass an unpadded dataset")
    lat = np.load(os.path.join(args.data_dir, "lat.npy"))
    W = np.load(os.path.join(args.data_dir, "lon.npy")).size
    if net_args["img_size"] != [lat.size, W]:
        sys.exit(f"model img_size {net_args['img_size']} != data grid {[lat.size, W]}; use the corrected 6x6 dataset")
    m = dict(np.load(os.path.join(args.data_dir, "normalize_mean.npz")))
    s = dict(np.load(os.path.join(args.data_dir, "normalize_std.npz")))
    mean = np.concatenate([m[v] for v in variables])[None, :, None, None].astype(np.float32)
    std = np.concatenate([s[v] for v in variables])[None, :, None, None].astype(np.float32)

    log("building hazard labels ...")
    labelsets = hd.build_labelsets(args.data_dir, args.imd_dir, args.hazards) if args.hazards else {}
    hazards = list(labelsets)
    log("loading splits into memory ...")
    data = {sp: J.SplitData(args.data_dir, sp, variables, mean, std, lead, labelsets) for sp in ("train", "val", "test")}
    for sp, d in data.items():
        log(f"{sp}: {len(d.times)} hours, {len(d)} forecast pairs, {len(d.issue_hour_indices())} 00 UTC samples")

    # ---- model ----
    if args.tiny_model:
        net_args.update(embed_dim=64, depth=2, decoder_depth=1, num_heads=4)
    from climax.arch import ClimaX

    net = ClimaX(**net_args)
    init_meta = {"init": "scratch"}
    if args.init_ckpt and not args.from_scratch:
        ck = iu.load_checkpoint(args.init_ckpt)
        J.load_climax_state(net, ck["state_dict"])
        init_meta = {"init": os.path.basename(args.init_ckpt), "init_lambda": ck.get("hyper_parameters", {}).get("physics_lambda"),
                     "init_epoch": ck.get("epoch")}
        del ck
        log("initialised from", init_meta)
    elif not args.from_scratch:
        sys.exit("give --init_ckpt or --from_scratch")
    model = J.JointModel(net, hazards).to(device)

    tasks = ["weather"] + ([] if args.no_physics else ["physics"]) + hazards
    if args.balancing == "uncertainty":
        weighting = J.UncertaintyWeighting(tasks).to(device)
    else:
        fw = {"weather": 1.0, "physics": args.fixed_physics * args.physics_scale_m ** 2}  # undo the scale -> raw-m^2 lambda
        fw.update({h: args.fixed_hazard for h in hazards})
        weighting = J.FixedWeighting({t: fw[t] for t in tasks})
    phys = J.HypsometricResidual(variables, mean, std, scale_m=args.physics_scale_m).to(device)
    w_lat = J.lat_weights(lat, device)
    pos_w = torch.tensor(hm.pos_weights(data["train"].haz), device=device) if hazards else None
    log("tasks:", tasks, "| pos_weight:", None if pos_w is None else pos_w.cpu().numpy().round(1).tolist())

    head_params = list(model.heads.parameters())
    enc_params = list(model.net.parameters())
    groups = [{"params": enc_params, "lr": args.lr_encoder, "weight_decay": 1e-5},
              {"params": head_params, "lr": args.lr_heads, "weight_decay": 1e-4}]
    if args.balancing == "uncertainty":
        groups.append({"params": list(weighting.parameters()), "lr": args.lr_weights, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99))
    steps_per_epoch = math.ceil(min(args.samples_per_epoch, len(data["train"])) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: lr_lambda(st, args.warmup_steps, total_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    val = data["val"]
    val_w_idx = np.arange(0, len(val), args.val_weather_stride)
    val_h_idx = val.issue_hour_indices()
    w_lat_np = w_lat.cpu().numpy()

    # ---- train ----
    history, best_score, bad = [], -np.inf, 0
    best_path = os.path.join(args.out_dir, "best.pt")
    step = 0
    for ep in range(args.epochs):
        model.train()
        idx_all = rng.choice(len(data["train"]), size=min(args.samples_per_epoch, len(data["train"])), replace=False)
        sums = {t: 0.0 for t in tasks}
        n = 0
        te = time.time()
        for b in range(0, len(idx_all), args.batch_size):
            x, y, h = data["train"].batch(idx_all[b:b + args.batch_size])
            x, y = x.to(device), y.to(device)
            lt = torch.full((len(x),), lead / 100.0, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                wpred, hlogit = model(x, lt, variables)
            wpred = wpred.float()
            losses = {"weather": J.weather_loss(wpred, y, w_lat)}
            if "physics" in tasks:
                losses["physics"] = phys(wpred, y)
            if hazards:
                for k, lk in zip(hazards, J.hazard_losses(hlogit.float(), h.to(device), pos_w)):
                    losses[k] = lk
            total, wts = weighting(losses)
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(weighting.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            for t in tasks:
                sums[t] += float(losses[t].detach()) * len(x)
            n += len(x)
        tr = {f"train_{t}": sums[t] / n for t in tasks}
        vl = evaluate_val(model, val, variables, lead, device, amp, w_lat_np, std, val_w_idx, val_h_idx)
        rec = {"epoch": ep, "minutes": (time.time() - te) / 60, **tr, **{f"weight_{t}": wts[t] for t in tasks}, **vl}
        history.append(rec)
        score = vl.get("val_ap_mean", -vl["val_w_mse"])
        msg = f"epoch {ep:02d} {rec['minutes']:.1f} min | " + " ".join(f"{t}={tr['train_' + t]:.4f}(w{wts[t]:.2g})" for t in tasks)
        log(msg + f" | val w_mse {vl['val_w_mse']:.4f} z500 {vl['val_rmse_geopotential_500']:.1f}"
            + (f" | val AP {vl['val_ap_mean']:.4f}" if "val_ap_mean" in vl else ""))
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)
        if (np.isfinite(score) and score > best_score) or not os.path.exists(best_path):
            best_score, bad = (score if np.isfinite(score) else best_score), 0
            torch.save({"model": model.state_dict(), "weighting": weighting.state_dict(), "epoch": ep, "tasks": tasks,
                        "hazards": hazards, "net_args": net_args, "args": vars(args)}, best_path)
        else:
            bad += 1
            if bad >= args.patience:
                log("early stopping")
                break

    # ---- test evaluation with the best epoch ----
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    log(f"best epoch {best['epoch']} (selection: {'mean val PR-AUC' if hazards else 'val w_mse'})")
    rows = []
    if hazards:
        T, X, Y, probs = {}, {}, {}, {}
        for sp in ("val", "test"):
            T[sp], X[sp] = hd.load_inputs(args.data_dir, sp, variables, (0,))
            Y[sp] = hd.aligned_targets(labelsets, sp, T[sp], lead)
            xn = torch.from_numpy((X[sp] - mean) / std).float()
            model.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(xn), 256):
                    xb = xn[i:i + 256].to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        _, hl = model(xb, torch.full((len(xb),), lead / 100.0, device=device), variables)
                    out.append(torch.sigmoid(hl.float()).cpu().numpy())
            probs[sp] = np.concatenate(out)
        np.savez_compressed(os.path.join(args.out_dir, "preds", "joint.npz"), val=probs["val"], test=probs["test"])
        np.savez_compressed(os.path.join(args.out_dir, "preds", "targets.npz"),
                            **{f"{sp}_y": Y[sp] for sp in ("val", "test")}, **{f"{sp}_times": T[sp].astype("int64") for sp in ("val", "test")})
        for k, hname in enumerate(hazards):
            thr = hm.best_f1_threshold(probs["val"][:, k], Y["val"][:, k])
            for sp in ("val", "test"):
                rows.append({"source": "joint", "hazard": hname, "split": sp, **hm.scores(probs[sp][:, k], Y[sp][:, k], thr)})
        hz = pd.DataFrame(rows)
        hz.to_csv(os.path.join(args.out_dir, "hazard_results.csv"), index=False)
        log("TEST hazards (threshold from val):\n" + hz[hz.split == "test"][["hazard", "pr_auc", "roc_auc", "f1", "precision", "recall", "prevalence"]].round(4).to_string(index=False))

        if args.compare_dir:
            ref_t = np.load(os.path.join(args.compare_dir, "preds", "targets.npz"))
            same = np.array_equal(ref_t["test_times"], T["test"].astype("int64")) and np.allclose(ref_t["test_y"], Y["test"], equal_nan=True)
            if not same:
                log("compare_dir targets differ from this run (different dataset/labels) - skipping bootstrap")
            else:
                boot = []
                srcs = args.compare_sources or [f[:-4] for f in os.listdir(os.path.join(args.compare_dir, "preds")) if f.endswith(".npz") and f != "targets.npz"]
                for src in srcs:
                    other = np.load(os.path.join(args.compare_dir, "preds", f"{src}.npz"))["test"]
                    for k, hname in enumerate(hazards):
                        r = hm.paired_block_bootstrap(probs["test"][:, k], other[:, k], Y["test"][:, k])
                        boot.append({"a": "joint", "b": src, "hazard": hname, **r})
                bt = pd.DataFrame(boot)
                bt.to_csv(os.path.join(args.out_dir, "bootstrap_vs_frozen.csv"), index=False)
                log("paired 7-day block bootstrap, test PR-AUC(joint) - PR-AUC(b):\n" + bt.round(4).to_string(index=False))

    # weather: same evaluation as evaluate_forecast.py (persistence + climatology on identical pairs)
    clim = fe.train_climatology(args.data_dir, variables)
    pred_fn = fe.climax_predictor(model.net, variables, variables, m, s, lead, lat, device=str(device))
    wrows = fe.evaluate(args.data_dir, "test", variables, variables, lead, clim, {"joint": pred_fn},
                        {"real": slice(None)}, args.weather_eval_max_files)
    wt = pd.DataFrame(wrows)
    wt.to_csv(os.path.join(args.out_dir, "weather_test.csv"), index=False)
    piv = wt.pivot_table(index="variable", columns="source", values="rmse", sort=False)
    log("TEST weather RMSE:\n" + piv.round(3).to_string())

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    json.dump({"args": vars(args), "git_commit": commit, "tasks": tasks, "hazards": hazards, **init_meta,
               "best_epoch": int(best["epoch"]), "final_task_weights": {t: history[best["epoch"]][f"weight_{t}"] for t in tasks},
               "n_pairs": {sp: len(d) for sp, d in data.items()}, "elapsed_min": round((time.time() - t0) / 60, 1)},
              open(os.path.join(args.out_dir, "run_meta.json"), "w"), indent=1, default=str)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
