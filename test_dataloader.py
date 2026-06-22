# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Build a minimal fake ERA5-style dataset and run it through GlobalForecastDataModule.

ERA5 on disk (WeatherBench layout, before preprocessing):
  <root_dir>/
    constants.nc          # lsm, orography, lat2d  (no time dimension)
    2m_temperature/<year>_*.nc
    10m_u_component_of_wind/...
    geopotential/...      # 4D: time x level x lat x lon  (variable name z, u, t, ...)
    ...

After nc2np_equally_era5.py (what GlobalForecastDataModule reads):
  <npz_dir>/
    train|val|test/<year>_<shard>.npz   # arrays per var, shape (T, 1, 32, 64)
    train|val|test/climatology.npz
    normalize_mean.npz, normalize_std.npz, lat.npy, lon.npy
"""

import os
import shutil
import types
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from climax.global_forecast.datamodule import GlobalForecastDataModule
from climax.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR

from test_model import DEFAULT_VARS

# Subset of pressure levels used at 5.625° in configs/global_forecast_climax.yaml
PRESSURE_LEVELS = [50, 250, 500, 600, 700, 850, 925]

OUT_VARIABLES = [
    "geopotential_500",
    "temperature_850",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

SINGLE_LEVEL_DIRS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

PRESSURE_LEVEL_DIRS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]

H, W = 32, 64
N_TIMESTEPS = 4
YEAR = 2019
DATA_ROOT = Path(__file__).resolve().parent / "fake_era5_demo"
RAW_DIR = DATA_ROOT / "raw_5.625deg"
NPZ_DIR = DATA_ROOT / "5.625deg_npz"


def _lat_lon():
    lat = np.linspace(90.0, -90.0, H, dtype=np.float32)
    lon = np.arange(0.0, 360.0, 360.0 / W, dtype=np.float32)
    return lat, lon


def create_fake_era5_netcdf(raw_dir: Path, year: int = YEAR, n_timesteps: int = N_TIMESTEPS) -> None:
    """Write WeatherBench-style NetCDF files (xarray)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = _lat_lon()
    times = np.arange(n_timesteps, dtype=np.int64)

    constants = xr.Dataset(
        {
            "lsm": (["lat", "lon"], np.random.rand(H, W).astype(np.float32)),
            "orography": (["lat", "lon"], np.random.rand(H, W).astype(np.float32)),
            "lat2d": (["lat", "lon"], np.broadcast_to(lat[:, None], (H, W)).astype(np.float32)),
        },
        coords={"lat": lat, "lon": lon},
    )
    constants.to_netcdf(raw_dir / "constants.nc")

    rng = np.random.default_rng(0)
    surface_data = rng.standard_normal((n_timesteps, H, W), dtype=np.float32)

    for folder in SINGLE_LEVEL_DIRS:
        var_name = NAME_TO_VAR[folder]
        out_dir = raw_dir / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {var_name: (["time", "lat", "lon"], surface_data)},
            coords={"time": times, "lat": lat, "lon": lon},
        )
        ds.to_netcdf(out_dir / f"{year}_5.625deg.nc")

    levels = [lvl for lvl in DEFAULT_PRESSURE_LEVELS if lvl in PRESSURE_LEVELS]
    level_data = rng.standard_normal((n_timesteps, len(levels), H, W), dtype=np.float32)

    for folder in PRESSURE_LEVEL_DIRS:
        var_name = NAME_TO_VAR[folder]
        out_dir = raw_dir / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {var_name: (["time", "level", "lat", "lon"], level_data)},
            coords={"time": times, "lat": lat, "lon": lon, "level": levels},
        )
        ds.to_netcdf(out_dir / f"{year}_5.625deg.nc")


def create_preprocessed_npz(
    npz_dir: Path,
    variables,
    n_timesteps: int = N_TIMESTEPS,
    year: int = YEAR,
) -> None:
    """Write minimal .npz shards + stats matching nc2np_equally_era5.py output."""
    rng = np.random.default_rng(1)
    lat, lon = _lat_lon()

    shard = {}
    for var in variables:
        shard[var] = rng.standard_normal((n_timesteps, 1, H, W), dtype=np.float32)

    for partition in ("train", "val", "test"):
        part_dir = npz_dir / partition
        part_dir.mkdir(parents=True, exist_ok=True)
        np.savez(part_dir / f"{year}_0.npz", **shard)
        clim = {var: shard[var].mean(axis=0).astype(np.float32) for var in variables}
        np.savez(part_dir / "climatology.npz", **clim)

    normalize_mean = {var: np.array([float(shard[var].mean())], dtype=np.float32) for var in variables}
    normalize_std = {var: np.array([float(shard[var].std()) + 1e-6], dtype=np.float32) for var in variables}
    np.savez(npz_dir / "normalize_mean.npz", **normalize_mean)
    np.savez(npz_dir / "normalize_std.npz", **normalize_std)
    np.save(npz_dir / "lat.npy", lat)
    np.save(npz_dir / "lon.npy", lon)


def build_climax_model():
    from climax.arch import ClimaX

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    model = ClimaX(
        default_vars=DEFAULT_VARS,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    )
    model.eval()
    model.cpu()

    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars], dtype=np.int64)
        return torch.from_numpy(ids).to(device)

    model.get_var_ids = types.MethodType(get_var_ids, model)
    return model


def main():
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    create_fake_era5_netcdf(RAW_DIR)
    create_preprocessed_npz(NPZ_DIR, DEFAULT_VARS)

    datamodule = GlobalForecastDataModule(
        root_dir=str(NPZ_DIR),
        variables=DEFAULT_VARS,
        buffer_size=4,
        out_variables=OUT_VARIABLES,
        predict_range=1,
        hrs_each_step=1,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))
    x, y, lead_times, variables, out_variables = batch

    model = build_climax_model()
    with torch.no_grad():
        _, output = model.forward(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            metric=None,
            lat=None,
        )

    print("Fake raw ERA5 NetCDF root:", RAW_DIR)
    print("Preprocessed NPZ root:", NPZ_DIR)
    print("Batch x shape:", tuple(x.shape))
    print("Batch y shape:", tuple(y.shape))
    print("Model output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
