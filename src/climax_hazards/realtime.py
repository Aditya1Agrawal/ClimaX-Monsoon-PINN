"""Real-time inputs for the demo: today's atmosphere from NOAA GFS via the Open-Meteo API.

The model was trained on WeatherBench ERA5 at 5.625 degrees. WeatherBench produced that grid by
bilinear interpolation of 0.25-degree ERA5, so the value of a coarse cell equals the field at its
centre; here the same 17 variables are read from GFS at the 36 cell centres (Open-Meteo returns the
nearest 0.25-degree GFS point). Units are converted to the training units:

    temperature  degC -> K
    geopotential geopotential height (gpm) * g0 -> m^2/s^2
    10 m wind    speed + meteorological direction (blowing from) -> u, v components

Caveats (shown in the app): GFS is not ERA5 (small distribution shift), only the trained 72 h lead
is supported, and real-time forecasts are not part of the validated 2015-2018 results.
Only grid coordinates and dates are sent to the service; no user data.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

API = "https://api.open-meteo.com/v1/gfs"
G0 = 9.80665
LEVELS = (50, 250, 500, 600, 700, 850, 925)


def openmeteo_hourly_names() -> List[str]:
    names = ["temperature_2m", "wind_speed_10m", "wind_direction_10m"]
    for p in LEVELS:
        names += [f"temperature_{p}hPa", f"geopotential_height_{p}hPa"]
    return names


def build_url(lat: np.ndarray, lon: np.ndarray, start: date, end: date) -> str:
    """One request for all cell centres (lat-major order), hourly, UTC, wind in m/s."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    q = {"latitude": ",".join(f"{v:.4f}" for v in la.ravel()), "longitude": ",".join(f"{v:.4f}" for v in lo.ravel()),
         "hourly": ",".join(openmeteo_hourly_names()), "start_date": start.isoformat(), "end_date": end.isoformat(),
         "timezone": "UTC", "wind_speed_unit": "ms",
         # elevation=nan: no statistical downscaling to a high-resolution terrain model; return the GFS
         # grid value, as ERA5/WeatherBench (model-grid values, interpolated) does. One value per point.
         "elevation": ",".join(["nan"] * la.size)}
    return API + "?" + urllib.parse.urlencode(q)


def _ssl_context():
    """Verify certificates with the operating system's trust store (as browsers and curl do) when
    ``truststore`` is installed; otherwise Python's default. Verification is never disabled."""
    import ssl

    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


def fetch(lat: np.ndarray, lon: np.ndarray, start: date, end: date, timeout: float = 60.0) -> list:
    """Raw Open-Meteo response: a list with one object per cell centre."""
    url = build_url(lat, lon, start, end)
    req = urllib.request.Request(url, headers={"User-Agent": "climax-hazard-demo"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        data = json.loads(r.read().decode())
    if isinstance(data, dict):
        if data.get("error"):
            raise RuntimeError(f"Open-Meteo error: {data.get('reason')}")
        data = [data]
    if len(data) != lat.size * lon.size:
        raise RuntimeError(f"expected {lat.size * lon.size} points, got {len(data)}")
    return data


def wind_uv(speed, direction_deg) -> Tuple[np.ndarray, np.ndarray]:
    """Meteorological direction (where the wind blows FROM, degrees clockwise from north) -> (u east, v north)."""
    r = np.deg2rad(np.asarray(direction_deg, dtype=np.float64))
    s = np.asarray(speed, dtype=np.float64)
    return -s * np.sin(r), -s * np.cos(r)


def fields_at(data: list, when: str, variables: Sequence[str], shape: Tuple[int, int]) -> np.ndarray:
    """Physical (V, H, W) array in training units for the hour ``when`` ('YYYY-MM-DDTHH:MM')."""
    H, W = shape
    out = np.full((len(variables), H * W), np.nan, dtype=np.float64)
    for n, pt in enumerate(data):
        h = pt["hourly"]
        if when not in h["time"]:
            raise KeyError(f"{when} not in the Open-Meteo response")
        i = h["time"].index(when)
        val = {k: (np.nan if h[k][i] is None else float(h[k][i])) for k in h if k != "time"}
        u, v = wind_uv(val["wind_speed_10m"], val["wind_direction_10m"])
        for c, var in enumerate(variables):
            if var == "2m_temperature":
                out[c, n] = val["temperature_2m"] + 273.15
            elif var == "10m_u_component_of_wind":
                out[c, n] = u
            elif var == "10m_v_component_of_wind":
                out[c, n] = v
            elif var.startswith("temperature_"):
                out[c, n] = val[f"temperature_{var.split('_')[-1]}hPa"] + 273.15
            elif var.startswith("geopotential_"):
                out[c, n] = val[f"geopotential_height_{var.split('_')[-1]}hPa"] * G0
            else:
                raise KeyError(f"no real-time source for {var}")
    return out.reshape(len(variables), H, W).astype(np.float32)


def standardised_anomaly(x_phys: np.ndarray, mu: np.ndarray, sd: np.ndarray, variables: Sequence[str]) -> Dict[str, float]:
    """Max |z| per variable against the training normalisation: a quick check that the inputs look like ERA5."""
    z = (x_phys[None] - mu) / sd
    return {v: float(np.nanmax(np.abs(z[0, i]))) for i, v in enumerate(variables)}


def issue_window(issue: date, lead_days: int) -> Tuple[date, date]:
    return issue, issue + timedelta(days=lead_days)
