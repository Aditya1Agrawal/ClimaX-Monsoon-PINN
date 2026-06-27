import os
import glob
import xarray as xr
import numpy as np

# ==========================================
# 1. PATH CONFIGURATIONS
# ==========================================
INPUT_PATH = "./raw_global_data/5.625deg"
OUTPUT_PATH = "./cropped_india_data"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Define the Indian subcontinent bounding box
LAT_MIN, LAT_MAX = 5.0, 38.0
LON_MIN, LON_MAX = 65.0, 98.0

def crop_dataset(dataset_path, output_dir):
    print(f"\nProcessing directory: {dataset_path}")
    
    # Find all NetCDF files (.nc) in the variable folder
    nc_files = sorted(glob.glob(os.path.join(dataset_path, "*.nc")))
    if not nc_files:
        print(f"No NetCDF files found in {dataset_path}. Skipping.")
        return

    for file_path in nc_files:
        file_name = os.path.basename(file_path)
        print(f"  Slicing file: {file_name}...")
        
        # Open the global NetCDF dataset
        with xr.open_dataset(file_path) as ds:
            # WeatherBench datasets often use 0-360 for longitude or -180 to 180. 
            # Also check if latitude is inverted (90 to -90 vs -90 to 90).
            
            # Handle latitude slice direction
            if ds.lat[0] > ds.lat[-1]:
                lat_slice = slice(LAT_MAX, LAT_MIN)
            else:
                lat_slice = slice(LAT_MIN, LAT_MAX)
                
            # Handle longitude format (convert 65-98 to 0-360 if necessary)
            if (ds.lon_min if hasattr(ds, 'lon_min') else ds.lon.max()) > 180:
                lon_slice = slice(LON_MIN % 360, LON_MAX % 360)
            else:
                lon_slice = slice(LON_MIN, LON_MAX)

            # Perform the slice
            cropped_ds = ds.sel(lat=lat_slice, lon=lon_slice)
            
            # Convert to NumPy / NPZ format for ClimaX training efficiency
            # Extract the variable name from the file or dataset
            var_name = list(cropped_ds.data_vars)[0]
            data_array = cropped_ds[var_name].values  # Shape: [time, lat, lon] or [time, level, lat, lon]
            
            # Save the cropped data as a lightweight compressed .npz array
            var_output_dir = os.path.join(output_dir, os.path.basename(dataset_path))
            os.makedirs(var_output_dir, exist_ok=True)
            
            npz_filename = file_name.replace(".nc", ".npz")
            npz_path = os.path.join(var_output_dir, npz_filename)
            
            np.savez_compressed(
                npz_path, 
                data=data_array, 
                lat=cropped_ds.lat.values, 
                lon=cropped_ds.lon.values
            )
            print(f"  Successfully saved regional array to: {npz_path}")

def main():
    # Scan the raw_global_data/5.625deg directory for variable folders
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input path {INPUT_PATH} does not exist.")
        
    subdirs = [os.path.join(INPUT_PATH, d) for d in os.listdir(INPUT_PATH) 
               if os.path.isdir(os.path.join(INPUT_PATH, d))]
    
    if not subdirs:
        print("No variable directories found. Make sure files are unzipped inside subfolders!")
        return

    for subdir in subdirs:
        crop_dataset(subdir, OUTPUT_PATH)
    
    print("\nProcessing complete for current variables!")

if __name__ == "__main__":
    main()