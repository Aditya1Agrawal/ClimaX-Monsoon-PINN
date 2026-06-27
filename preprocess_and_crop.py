import os
import glob
import click
import numpy as np
import xarray as xr
from tqdm import tqdm

# Bounding box for the Indian region
LAT_REGS = [38.0, 7.0]   # North to South (ERA5 latitude is typically descending)
LON_REGS = [68.0, 98.0]  # West to East

HOURS_PER_YEAR = 8760
DEFAULT_PRESSURE_LEVELS = [50, 250, 500, 600, 700, 850, 925, 1000]
NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "geopotential": "z",
    "temperature": "t",
    "land_sea_mask": "lsm",
    "orography": "orog",
    "lattitude": "lat2d"
}

def crop_dataset(ds):
    """Crops an xarray dataset to the Indian region safely, checking coordinate order."""
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    
    # Handle descending vs ascending latitude order safely
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        lat_slice = slice(max(LAT_REGS), min(LAT_REGS))
    else:
        lat_slice = slice(min(LAT_REGS), max(LAT_REGS))
        
    return ds.sel({lat_name: lat_slice, lon_name: slice(min(LON_REGS), max(LON_REGS))})

def nc2np_cropped(path, variables, years, save_dir, partition, num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    normalize_mean = {}
    normalize_std = {}
    climatology = {}

    # 1. Load and Crop Constants
    constants_path = os.path.join(path, "constants.nc")
    if not os.path.exists(constants_path):
        # Fallback if constants.nc is inside the folder structure
        constants_path = glob.glob(os.path.join(path, "**/constants.nc"), recursive=True)[0]
        
    constants = xr.open_dataset(constants_path)
    constants = crop_dataset(constants)
    
    constant_fields = ["land_sea_mask", "orography", "lattitude"]
    constant_values = {}
    for f in constant_fields:
        code = NAME_TO_VAR[f]
        if code in constants:
            arr = constants[code].to_numpy()
        else:
            # Fallback if coordinate generation is needed dynamically
            arr = np.zeros((len(constants['lat']), len(constants['lon'])))
            
        constant_values[f] = np.expand_dims(arr, axis=(0, 1)).repeat(HOURS_PER_YEAR, axis=0)
        if partition == "train":
            normalize_mean[f] = constant_values[f].mean(axis=(0, 2, 3))
            normalize_std[f] = constant_values[f].std(axis=(0, 2, 3))

    # 2. Process Climate Variables Year by Year
    for year in tqdm(years, desc=f"Processing {partition}"):
        np_vars = {f: constant_values[f] for f in constant_fields}

        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            if not ps:
                continue
            
            ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)
            ds = crop_dataset(ds) # Crop immediately to save space and RAM
            code = NAME_TO_VAR[var]

            if len(ds[code].shape) == 3:  # Surface variables
                ds[code] = ds[code].expand_dims("val", axis=1)
                np_vars[var] = ds[code].to_numpy()[:HOURS_PER_YEAR]

                if partition == "train":
                    var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                    var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                    normalize_mean.setdefault(var, []).append(var_mean_yearly)
                    normalize_std.setdefault(var, []).append(var_std_yearly)

                climatology.setdefault(var, []).append(np_vars[var].mean(axis=0))

            else:  # Multi-level pressure variables
                all_levels = np.intersect1d(ds["level"][:].to_numpy(), DEFAULT_PRESSURE_LEVELS)
                for level in all_levels:
                    ds_level = ds.sel(level=[level])
                    v_key = f"{var}_{int(level)}"
                    np_vars[v_key] = ds_level[code].to_numpy()[:HOURS_PER_YEAR]

                    if partition == "train":
                        var_mean_yearly = np_vars[v_key].mean(axis=(0, 2, 3))
                        var_std_yearly = np_vars[v_key].std(axis=(0, 2, 3))
                        normalize_mean.setdefault(v_key, []).append(var_mean_yearly)
                        normalize_std.setdefault(v_key, []).append(var_std_yearly)

                    climatology.setdefault(v_key, []).append(np_vars[v_key].mean(axis=0))

        # Shard and save the cropped time blocks
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"), **sharded_data)

    # 3. Finalize Train Statistics
    if partition == "train":
        for var in normalize_mean.keys():
            if var not in constant_fields:
                mean = np.stack(normalize_mean[var], axis=0)
                std = np.stack(normalize_std[var], axis=0)
                variance = (std**2).mean(axis=0) + (mean**2).mean(axis=0) - mean.mean(axis=0) ** 2
                normalize_mean[var] = mean.mean(axis=0)
                normalize_std[var] = np.sqrt(variance)

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    # 4. Save Climatology
    climatology = {k: np.stack(v, axis=0).mean(axis=0) for k, v in climatology.items()}
    np.savez(os.path.join(save_dir, partition, "climatology.npz"), **climatology)


@click.command()
@click.option("--root_dir", type=click.Path(exists=True), required=True)
@click.option("--save_dir", type=str, required=True)
@click.option("--start_train_year", type=int, default=1979)
@click.option("--start_val_year", type=int, default=2011) # Updated to follow your custom 80/10/10 split
@click.option("--start_test_year", type=int, default=2015)
@click.option("--end_year", type=int, default=2019)
@click.option("--num_shards", type=int, default=8)
def main(root_dir, save_dir, start_train_year, start_val_year, start_test_year, end_year, num_shards):
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    # Extract clean list of variables present in directories
    variables = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Detected variables for automated processing: {variables}")

    os.makedirs(save_dir, exist_ok=True)

    # Process all splits consecutively
    nc2np_cropped(root_dir, variables, train_years, save_dir, "train", num_shards)
    nc2np_cropped(root_dir, variables, val_years, save_dir, "val", num_shards)
    nc2np_cropped(root_dir, variables, test_years, save_dir, "test", num_shards)

    # Save the exact cropped lat/lon coordinate matrices required by ClimaX
    sample_file = glob.glob(os.path.join(root_dir, variables[0], "*.nc"))[0]
    ds_sample = crop_dataset(xr.open_dataset(sample_file))
    np.save(os.path.join(save_dir, "lat.npy"), ds_sample["lat"].to_numpy())
    np.save(os.path.join(save_dir, "lon.npy"), ds_sample["lon"].to_numpy())
    print(f"Successfully constructed clean Indian regional ClimaX dataset at: {save_dir}")

if __name__ == "__main__":
    main()