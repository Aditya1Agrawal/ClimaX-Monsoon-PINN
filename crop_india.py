# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Crop a NetCDF climate file to the Indian subcontinent bounding box.

Default input: fake ERA5-style file from test_dataloader.py
  fake_era5_demo/raw_5.625deg/2m_temperature/2019_5.625deg.nc
"""

from pathlib import Path

import xarray as xr

# ---------------------------------------------------------------------------
# Bounding box (edit these to change the crop region)
# ---------------------------------------------------------------------------
# Latitude: north and south edges in degrees north (positive = N, negative = S).
LAT_NORTH = 38.0   # northern boundary (e.g. Himalayas / northern Pakistan)
LAT_SOUTH = 8.0    # southern boundary (e.g. southern tip of India / Sri Lanka margin)

# Longitude: west and east edges in degrees east (0–360 convention, as in ERA5/WeatherBench).
# For files that use -180..180 longitude, set LON_WEST/LON_EAST in that range instead
# (e.g. LON_WEST = 68, LON_EAST = 98 still works for India).
LON_WEST = 68.0    # western boundary (Arabian Sea / Pakistan)
LON_EAST = 98.0    # eastern boundary (Bay of Bengal / eastern Bangladesh)

# Paths (change INPUT_PATH to any NetCDF with lat/lon coordinates)
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "fake_era5_demo" / "raw_5.625deg" / "2m_temperature" / "2019_5.625deg.nc"
OUTPUT_PATH = SCRIPT_DIR / "fake_era5_demo" / "india_crop" / "2m_temperature_india.nc"


def _lat_slice(lat_coord, lat_north: float, lat_south: float):
    """
    Return a slice for latitude that works whether lat increases or decreases.

    WeatherBench / ERA5 grids often run 90°N -> 90°S (decreasing). In that case
    xarray expects slice(high_lat, low_lat). For ascending lat, use slice(low, high).
    """
    if float(lat_coord[0]) > float(lat_coord[-1]):
        # Decreasing latitude (north at index 0)
        return slice(lat_north, lat_south)
    return slice(lat_south, lat_north)


def _lon_slice(lon_coord, lon_west: float, lon_east: float):
    """
    Return a slice for longitude.

    Assumes 0–360° longitude (standard for ERA5). If your file uses -180..180,
    use negative values for west longitudes (e.g. LON_WEST = 68, LON_EAST = 98
    are unchanged for India).
    """
    lon_min = float(lon_coord.min())
    lon_max = float(lon_coord.max())

    if lon_min < 0 and lon_west > 180:
        # Convert 0–360 request to -180..180 file coordinates
        lon_west = lon_west - 360
        lon_east = lon_east - 360

    if lon_max > 180 and lon_west < 0:
        lon_west += 360
        lon_east += 360

    return slice(lon_west, lon_east)


def crop_to_bbox(
    ds: xr.Dataset,
    lat_north: float,
    lat_south: float,
    lon_west: float,
    lon_east: float,
) -> xr.Dataset:
    """Subset dataset to the given lat/lon bounding box."""
    lat_dim = "lat" if "lat" in ds.dims else "latitude"
    lon_dim = "lon" if "lon" in ds.dims else "longitude"

    lat_sl = _lat_slice(ds[lat_dim], lat_north, lat_south)
    lon_sl = _lon_slice(ds[lon_dim], lon_west, lon_east)

    return ds.sel({lat_dim: lat_sl, lon_dim: lon_sl})


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run test_dataloader.py first to generate fake ERA5 NetCDF files, "
            "or set INPUT_PATH to your own NetCDF path."
        )

    ds = xr.open_dataset(INPUT_PATH)
    cropped = crop_to_bbox(ds, LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cropped.to_netcdf(OUTPUT_PATH)
    ds.close()

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Bounding box: lat {LAT_SOUTH}°N–{LAT_NORTH}°N, lon {LON_WEST}°E–{LON_EAST}°E")
    print(f"Cropped shape: {dict(cropped.dims)}")


if __name__ == "__main__":
    main()
