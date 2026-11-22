"""Real-time input conversion (no network: the Open-Meteo response is faked)."""

from datetime import date
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from climax_hazards import realtime as R

VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential_500", "temperature_850"]


def test_wind_uv_meteorological_convention():
    u, v = R.wind_uv([10, 10, 10, 10], [0, 90, 180, 270])     # from N, E, S, W
    assert np.allclose(u, [0, -10, 0, 10], atol=1e-9)
    assert np.allclose(v, [-10, 0, 10, 0], atol=1e-9)


def _fake(n_points, t="2026-11-23T00:00"):
    hourly = {"time": ["2026-11-22T23:00", t]}
    for k in R.openmeteo_hourly_names():
        hourly[k] = [0.0, 1.0]
    hourly["temperature_2m"] = [0.0, 25.0]
    hourly["temperature_850hPa"] = [0.0, 15.0]
    hourly["geopotential_height_500hPa"] = [0.0, 5880.0]
    hourly["wind_speed_10m"] = [0.0, 5.0]
    hourly["wind_direction_10m"] = [0.0, 270.0]                 # westerly
    return [{"hourly": hourly} for _ in range(n_points)]


def test_fields_units_and_shape():
    x = R.fields_at(_fake(4), "2026-11-23T00:00", VARS, (2, 2))
    assert x.shape == (5, 2, 2)
    assert np.allclose(x[0], 298.15) and np.allclose(x[4], 288.15)       # degC -> K
    assert np.allclose(x[3], 5880.0 * R.G0, rtol=1e-6)                  # gpm -> m^2/s^2
    assert np.allclose(x[1], 5.0) and np.allclose(x[2], 0.0, atol=1e-6)  # westerly -> u > 0
    with pytest.raises(KeyError):
        R.fields_at(_fake(4), "2026-11-24T00:00", VARS, (2, 2))


def test_url_contains_all_cells_and_variables():
    lat, lon = np.array([8.4375, 14.0625]), np.array([67.5, 73.125, 78.75])
    q = parse_qs(urlparse(R.build_url(lat, lon, date(2026, 11, 23), date(2026, 11, 26))).query)
    assert len(q["latitude"][0].split(",")) == 6 == len(q["longitude"][0].split(","))
    assert set(q["hourly"][0].split(",")) == set(R.openmeteo_hourly_names())
    assert q["timezone"] == ["UTC"] and q["wind_speed_unit"] == ["ms"]
    assert q["elevation"][0].split(",") == ["nan"] * 6        # model-grid values, no terrain downscaling


def test_standardised_anomaly():
    mu = np.zeros((1, 2, 1, 1), np.float32)
    sd = np.ones((1, 2, 1, 1), np.float32)
    x = np.zeros((2, 2, 2), np.float32)
    x[1, 0, 0] = 5
    z = R.standardised_anomaly(x, mu, sd, ["a", "b"])
    assert z == {"a": 0.0, "b": 5.0}
