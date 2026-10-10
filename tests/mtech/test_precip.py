import numpy as np
import pytest

from climax_hazards import precip as P


def _write_year(root, year, fill):
    n = 366 if year % 4 == 0 else 365
    a = np.full((n, 129, 135), fill, dtype="<f4")
    a[:, :, :20] = P.IMD_MISSING  # western strip missing (like sea)
    (root / "rain").mkdir(exist_ok=True)
    a.tofile(root / "rain" / f"{year}.grd")
    return a


def test_reader_masks_missing_and_checks_size(tmp_path):
    _write_year(tmp_path, 2000, 3.0)
    dates, rain = P.read_imd_year(tmp_path, 2000)
    assert rain.shape == (366, 129, 135) and str(dates[-1]) == "2000-12-31"
    assert np.isnan(rain[:, :, :20]).all() and (rain[:, :, 20:] == 3).all()
    (tmp_path / "rain" / "2001.grd").write_bytes(b"\0" * 100)
    with pytest.raises(ValueError):
        P.read_imd_year(tmp_path, 2001)


def test_regrid_conserves_and_reports_coverage():
    lat, lon = np.array([19.6875]), np.array([73.125, 84.375])
    w1, w2, area = P.conservative_weights(lat, lon)
    rain = np.ones((2, 129, 135), dtype=np.float32)
    rain[1] = 10.0
    lon_grid = P.IMD_LON
    rain[:, :, lon_grid < 73.125] = np.nan  # west half of the first box missing
    mean, cov = P.regrid_to_boxes(rain, w1, w2, area)
    assert np.allclose(mean[:, 0, :], [[1, 1], [10, 10]])
    assert abs(cov[0, 0, 0] - 0.5) < 0.03 and abs(cov[0, 0, 1] - 1.0) < 0.01
