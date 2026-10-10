"""Build a regional ClimaX npz dataset from the raw WeatherBench 5.625 deg NetCDF files.

Same layout and conventions as the internship ``preprocess_and_crop.py`` (8 shards of
1095 hourly steps per year, years truncated to 8760 h, train-only normalisation,
per-split climatology) so the existing datamodule reads it unchanged. Differences:

* the crop is given by explicit grid-point bounds, so the default lon 67.5-95.625 E
  produces a real 6 x 6 grid (the internship's 68.0 E bound dropped 67.5 E and the
  grid was later edge-padded to 6 columns);
* ``orography`` is read from the correct key (it was all zeros before);
* ``timestamps.npz`` (UTC hours since 1970, one entry per shard) and
  ``build_metadata.json`` are written next to the data.

The output folder must not exist yet; existing datasets are never overwritten.

Example:
    python scripts/mtech/build_regional_dataset.py \
        --raw_dir raw_global_data/5.625deg --out_dir regional_climax_dataset_6x6_corrected
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import xarray as xr

HOURS_PER_YEAR = 8760
NUM_SHARDS = 8
LEVELS = [50, 250, 500, 600, 700, 850, 925, 1000]
# same variable order as the internship dataset
SURFACE = {"10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "2m_temperature": "t2m"}
UPPER = {"geopotential": "z", "temperature": "t"}
CONSTANTS = {"land_sea_mask": "lsm", "orography": "orography", "lattitude": "lat2d"}


def crop(ds, lat_bounds, lon_bounds):
    lat = ds["lat"].values
    lat_sel = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_sel = (ds["lon"].values >= lon_bounds[0]) & (ds["lon"].values <= lon_bounds[1])
    out = ds.isel(lat=np.where(lat_sel)[0], lon=np.where(lon_sel)[0])
    if out["lat"].values[0] > out["lat"].values[-1]:
        out = out.isel(lat=slice(None, None, -1))
    return out


def load_year(raw_dir, year, lat_bounds, lon_bounds):
    arrays = {}
    for name, code in {**SURFACE, **UPPER}.items():
        path = os.path.join(raw_dir, name, f"{name}_{year}_5.625deg.nc")
        with xr.open_dataset(path) as ds:
            times = ds["time"].values
            expected = np.datetime64(f"{year}-01-01T00:00") + np.arange(len(times)).astype("timedelta64[h]")
            if not np.array_equal(times.astype("datetime64[h]"), expected):
                raise ValueError(f"{path}: time axis is not contiguous hourly from Jan 1 00 UTC")
            ds = crop(ds, lat_bounds, lon_bounds)
            if name in SURFACE:
                arrays[name] = ds[code].values[:HOURS_PER_YEAR, None].astype(np.float32)
            else:
                for lev in LEVELS:
                    arrays[f"{name}_{lev}"] = ds[code].sel(level=lev).values[:HOURS_PER_YEAR, None].astype(np.float32)
            lat, lon = ds["lat"].values, ds["lon"].values
    return arrays, lat, lon


def process_split(raw_dir, out_dir, split, years, const, lat_bounds, lon_bounds, stats, timestamps):
    os.makedirs(os.path.join(out_dir, split))
    clim = {}
    for year in years:
        t0 = time.time()
        arrays, lat, lon = load_year(raw_dir, year, lat_bounds, lon_bounds)
        for k, a in arrays.items():
            if not np.isfinite(a).all():
                raise ValueError(f"non-finite values in {k} {year}")
            clim.setdefault(k, []).append(a.mean(axis=0))
            if split == "train":
                stats.setdefault(k, []).append((a.mean(axis=(0, 2, 3)), a.std(axis=(0, 2, 3))))
        data = {k: np.repeat(v[None, None], HOURS_PER_YEAR, axis=0) for k, v in const.items()}
        data.update(arrays)
        per = HOURS_PER_YEAR // NUM_SHARDS
        start = np.datetime64(f"{year}-01-01T00:00", "h")
        for s in range(NUM_SHARDS):
            sl = slice(s * per, (s + 1) * per)
            np.savez(os.path.join(out_dir, split, f"{year}_{s}.npz"), **{k: v[sl] for k, v in data.items()})
            timestamps[f"{split}/{year}_{s}"] = (start + np.arange(sl.start, sl.stop)).astype("int64")
        print(f"  {split} {year} done in {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(out_dir, split, "climatology.npz"), **{k: np.stack(v).mean(axis=0) for k, v in clim.items()})
    return lat, lon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lat_bounds", type=float, nargs=2, default=[8.0, 37.0])
    p.add_argument("--lon_bounds", type=float, nargs=2, default=[67.0, 96.0])
    p.add_argument("--train_years", type=int, nargs=2, default=[1979, 2010], help="inclusive")
    p.add_argument("--val_years", type=int, nargs=2, default=[2011, 2014])
    p.add_argument("--test_years", type=int, nargs=2, default=[2015, 2018])
    args = p.parse_args()

    if os.path.exists(args.out_dir):
        sys.exit(f"{args.out_dir} already exists; refusing to overwrite")
    os.makedirs(args.out_dir)

    with xr.open_dataset(os.path.join(args.raw_dir, "constants.nc")) as c:
        c = crop(c, args.lat_bounds, args.lon_bounds)
        const = {name: c[code].values.astype(np.float32) for name, code in CONSTANTS.items()}
    const_stats = {k: (np.array([v.mean()]), np.array([v.std()])) for k, v in const.items()}

    stats, timestamps = {}, {}
    lat = lon = None
    for split, (y0, y1) in (("train", args.train_years), ("val", args.val_years), ("test", args.test_years)):
        lat, lon = process_split(args.raw_dir, args.out_dir, split, range(y0, y1 + 1), const,
                                 args.lat_bounds, args.lon_bounds, stats, timestamps)

    mean, std = {}, {}
    for k, (m, s) in const_stats.items():
        mean[k], std[k] = m, s
    for k, v in stats.items():  # same pooling formula as preprocess_and_crop.py
        m = np.stack([x[0] for x in v])
        s = np.stack([x[1] for x in v])
        mean[k] = m.mean(axis=0)
        std[k] = np.sqrt((s**2).mean(axis=0) + (m**2).mean(axis=0) - m.mean(axis=0) ** 2)
    np.savez(os.path.join(args.out_dir, "normalize_mean.npz"), **mean)
    np.savez(os.path.join(args.out_dir, "normalize_std.npz"), **std)
    np.save(os.path.join(args.out_dir, "lat.npy"), lat)
    np.save(os.path.join(args.out_dir, "lon.npy"), lon)
    np.savez(os.path.join(args.out_dir, "timestamps.npz"), **{k.replace("/", "__"): v for k, v in timestamps.items()})

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {
        "script": "scripts/mtech/build_regional_dataset.py", "git_commit": commit, "args": vars(args),
        "lat": lat.tolist(), "lon": lon.tolist(), "levels": LEVELS, "hours_per_year": HOURS_PER_YEAR,
        "num_shards": NUM_SHARDS, "time_convention": "hourly UTC; shard k starts at Jan 1 00 UTC + k*1095 h; leap years lose Dec 31",
        "timestamps_npz": "keys '<split>__<year>_<shard>', int64 hours since 1970-01-01T00 UTC",
        "zero_std_variables": [k for k, v in std.items() if np.any(v == 0)],
    }
    json.dump(meta, open(os.path.join(args.out_dir, "build_metadata.json"), "w"), indent=1)
    print("done:", args.out_dir, "grid", lat.size, "x", lon.size)


if __name__ == "__main__":
    main()
