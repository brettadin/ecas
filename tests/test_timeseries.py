"""Spectral time series: loaders, spectrum-at-time, transmission-spectrum extraction."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import archives, mock_data
from app.core.config import ConfigManager
from app.core.engine import build_engine
from app.core.timeseries import SpectralTimeSeries, TimeSeriesError, load_matrix_csv, load_x1dints
from app.core.transit import (
    DepthSpectrum, TransitError, TransitParams, bin_depth_spectrum, depth_spectrum_to_observation, transit_model,
)

TRUTH = {"H2O": 3.0, "CO2": 2.0, "CH4": 0.8, "O3": 1.5, "N2O": 1.0}


@pytest.fixture(scope="module")
def engine():
    return build_engine(ConfigManager.load())[0]


@pytest.fixture(scope="module")
def series_and_truth(engine):
    _, truth = mock_data.make_mock_lightcurve()
    mult = engine.multipliers_from_dict(TRUTH)
    return mock_data.make_mock_spectral_timeseries(engine, truth, multipliers=mult), truth, mult


def test_spectrum_at_time_is_flat_out_of_transit_and_dips_in_transit(series_and_truth):
    ts, truth, _ = series_and_truth
    assert ts.n_times == 360 and ts.n_wavelengths == 400 and np.all(np.diff(ts.wavelength_um) > 0)
    cadence = float(np.median(np.diff(ts.time)))
    wl, ratio_out, n_out = ts.relative_spectrum_at(truth.t0 - 2.5 * truth.duration_days, truth, 0.0)
    wl, ratio_in, n_in = ts.relative_spectrum_at(truth.t0, truth, 3 * cadence)
    assert n_out == 1 and n_in >= 3
    assert np.nanstd(ratio_out) < 0.002 and abs(np.nanmedian(ratio_out) - 1) < 0.002          # flat, ~1
    dip = 1 - np.nanmedian(ratio_in)
    assert 0.5 * truth.rp_rs ** 2 < dip < 2.5 * truth.rp_rs ** 2                              # ~ the transit depth
    assert 1 - np.nanmin(ratio_in) > 1.05 * dip                                                # bands add extra depth


def test_white_light_curve_and_transmission_spectrum_close_the_loop(engine, series_and_truth):
    ts, truth, mult = series_and_truth
    wl_curve = ts.white_light()
    assert wl_curve.time.size == ts.n_times and wl_curve.flux.min() < 0.99 * np.median(wl_curve.flux)
    spec = ts.transmission_spectrum(truth)
    base = truth.rp_rs ** 2
    assert abs(np.median(spec.depth) - base) / base < 0.15 and spec.depth.max() > 1.05 * np.median(spec.depth)
    obs = depth_spectrum_to_observation(spec)
    from app.core.identify import decompose
    from app.core.library import ReferenceLibrary
    res = decompose(obs, ReferenceLibrary.from_engine(engine))
    assert res.detected()                                                                      # something identified


def test_depth_spectrum_binning_propagates_uncertainties():
    spectrum = DepthSpectrum(
        wavelength_um=np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        depth=np.array([100.0, 200.0, 300.0, 400.0, 500.0]) * 1e-6,
        depth_err=np.array([10.0, 20.0, 10.0, 10.0, 10.0]) * 1e-6,
    )
    binned = bin_depth_spectrum(spectrum, 2)
    assert binned.wavelength_um.size == 3
    assert binned.depth[0] == pytest.approx(
        (100.0 / 10.0**2 + 200.0 / 20.0**2) / (1.0 / 10.0**2 + 1.0 / 20.0**2) * 1e-6)
    assert binned.depth_err[0] == pytest.approx(np.sqrt(1.0 / (1.0 / (10e-6)**2 + 1.0 / (20e-6)**2)))
    assert binned.depth[-1] == pytest.approx(500e-6)                           # partial final bin is retained
    with pytest.raises(TransitError, match="positive integer"):
        bin_depth_spectrum(spectrum, 0)


def test_transmission_extraction_uses_supplied_flux_errors(series_and_truth):
    ts, truth, _ = series_and_truth
    errors = np.full_like(ts.flux, 1e-4)
    with_errors = SpectralTimeSeries(ts.time, ts.wavelength_um, ts.flux, errors, dict(ts.meta))
    measured = with_errors.transmission_spectrum(truth)
    assert measured.depth_err.shape == ts.wavelength_um.shape
    assert np.isfinite(measured.depth_err).all() and np.all(measured.depth_err > 0.0)


def test_correlated_noise_diagnostics_compare_residual_scatter_without_changing_depth_errors(series_and_truth):
    ts, truth, _ = series_and_truth
    depth_err_before = ts.transmission_spectrum(truth).depth_err.copy()
    white = ts.characterize_correlated_noise(truth, max_bin_size=8)
    rng = np.random.default_rng(41)
    block_noise = np.repeat(rng.normal(0.0, 5e-4, (ts.n_times // 8 + 1, 1)), 8, axis=0)[:ts.n_times]
    correlated = SpectralTimeSeries(
        ts.time, ts.wavelength_um, ts.flux * (1.0 + block_noise),
        meta=dict(ts.meta))
    correlated_diagnostics = correlated.characterize_correlated_noise(truth, max_bin_size=8)

    assert white.bin_sizes[0] == 1
    assert white.beta_median[0] == pytest.approx(1.0)
    assert correlated_diagnostics.beta_median[-1] > 1.5
    assert correlated_diagnostics.white_noise_ppm_median > 0.0
    assert correlated_diagnostics.n_channels == ts.n_wavelengths
    np.testing.assert_array_equal(ts.transmission_spectrum(truth).depth_err, depth_err_before)
    with pytest.raises(TimeSeriesError, match="positive integer"):
        ts.characterize_correlated_noise(truth, max_bin_size=0)


def test_matrix_csv_roundtrip_units_and_errors(tmp_path, series_and_truth):
    ts, truth, _ = series_and_truth
    p = tmp_path / "m.csv"
    nm = ts.wavelength_um[::10] * 1000.0
    rows = ["# demo", "time," + ",".join(f"{w:.3f}" for w in nm)]
    rows += [f"{t:.6f}," + ",".join(f"{f:.7f}" for f in ts.flux[i, ::10]) for i, t in enumerate(ts.time)]
    p.write_text("\n".join(rows))
    back = load_matrix_csv(p)                                                                  # nm detected -> um
    assert back.n_wavelengths == nm.size and back.wavelength_um.min() == pytest.approx(ts.wavelength_um.min(), rel=1e-3)
    assert back.n_times == ts.n_times
    (tmp_path / "short.csv").write_text("time,1,2,3\n0,1,1,1\n")
    with pytest.raises(TimeSeriesError):
        load_matrix_csv(tmp_path / "short.csv")
    with pytest.raises(TimeSeriesError):
        SpectralTimeSeries(np.arange(30.0), np.linspace(1, 5, 10), np.ones((5, 10)))          # wrong shape


def _write_x1dints(path, ts, n_nan=3, with_times=True, time_col="int_mid_BJD_TDB", bjd0=2460000.25):
    from astropy.io import fits
    hdus = [fits.PrimaryHDU()]
    wl = ts.wavelength_um.copy(); wl_nan = np.concatenate([wl, np.full(n_nan, np.nan)])
    for i in range(ts.n_times):
        flux = np.concatenate([ts.flux[i], np.full(n_nan, np.nan)])
        cols = [fits.Column(name="WAVELENGTH", format="D", unit="um", array=wl_nan),
                fits.Column(name="FLUX", format="D", unit="Jy", array=flux),
                fits.Column(name="FLUX_ERROR", format="D", unit="Jy", array=np.full(flux.size, 1e-4))]
        hdus.append(fits.BinTableHDU.from_columns(cols, name="EXTRACT1D"))
    if with_times:
        cols = [fits.Column(name="integration_number", format="J", array=np.arange(1, ts.n_times + 1)),
                fits.Column(name=time_col, format="D", array=bjd0 + ts.time)]
        hdus.append(fits.BinTableHDU.from_columns(cols, name="INT_TIMES"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def test_x1dints_loader_on_a_file_built_to_the_documented_layout(tmp_path, series_and_truth):
    ts, truth, _ = series_and_truth
    p = tmp_path / "jw_x1dints.fits"
    _write_x1dints(p, ts)
    back = load_x1dints(p)
    assert back.n_times == ts.n_times and back.n_wavelengths == ts.n_wavelengths         # NaN padding dropped
    np.testing.assert_allclose(back.flux, ts.flux, rtol=1e-12)
    assert back.meta["time_kind"] == "BJD_TDB" and back.meta["time_offset"] == pytest.approx(2460000.25 + ts.time[0])
    assert back.time[0] == 0.0 and np.allclose(np.diff(back.time), np.diff(ts.time))
    mjd = tmp_path / "mjd.fits"; _write_x1dints(mjd, ts, time_col="int_mid_MJD_UTC", bjd0=60000.0)
    assert load_x1dints(mjd).meta["time_kind"] == "MJD_UTC"
    nt = tmp_path / "nt.fits"; _write_x1dints(nt, ts, with_times=False)
    assert "integration number" in load_x1dints(nt).meta["time_kind"]                     # honest about missing times
    from astropy.io import fits
    fits.HDUList([fits.PrimaryHDU()]).writeto(tmp_path / "empty.fits")
    with pytest.raises(TimeSeriesError, match="does not look like an x1dints"):
        load_x1dints(tmp_path / "empty.fits")


def test_x1dints_loader_reads_jwst_vector_columns_per_integration(tmp_path):
    from astropy.io import fits

    n_times, n_wavelengths = 60, 12
    times = 2460000.5 + np.arange(n_times) / 1440.0
    wavelength = np.linspace(1.0, 2.0, n_wavelengths)
    flux = np.ones((n_times, n_wavelengths)) + np.arange(n_times)[:, None] * 1e-5
    error = np.full_like(flux, 1e-4)
    spectra = fits.BinTableHDU.from_columns([
        fits.Column(name="INT_NUM", format="J", array=np.arange(n_times)),
        fits.Column(name="WAVELENGTH", format=f"{n_wavelengths}D", unit="um",
                    array=np.tile(wavelength, (n_times, 1))),
        fits.Column(name="FLUX", format=f"{n_wavelengths}D", unit="Jy", array=flux),
        fits.Column(name="FLUX_ERROR", format=f"{n_wavelengths}D", unit="Jy", array=error),
    ], name="EXTRACT1D")
    int_times = fits.BinTableHDU.from_columns([
        fits.Column(name="int_mid_BJD_TDB", format="D", array=times),
    ], name="INT_TIMES")
    path = tmp_path / "native_x1dints.fits"
    fits.HDUList([fits.PrimaryHDU(), int_times, spectra]).writeto(path)

    series = load_x1dints(path)
    assert series.n_times == n_times and series.n_wavelengths == n_wavelengths
    np.testing.assert_allclose(series.flux, flux)
    np.testing.assert_allclose(series.flux_err, error)
    assert series.meta["time_kind"] == "BJD_TDB"
    assert series.meta["time_offset"] == pytest.approx(times[0])


def test_archive_ephemeris_gives_transit_params_for_single_transit_data():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_archives import _csv
    rec = archives.fetch_planet("TEST-1 b", fetch=lambda u: _csv(pl_orbper="3.2", pl_tranmid="2459000.5",
                                                                   pl_ratror="0.1", pl_ratdor="9.4", pl_imppar="0.3"))
    # data observed ~ 7 orbits later, time axis = BJD - 2459022.0
    p = archives.transit_params_from_archive(rec, t_mid_bjd=2459022.9, time_offset_bjd=2459022.0)
    assert p.period == 3.2 and p.rp_rs == 0.1 and p.a_rs == 9.4 and p.b == 0.3
    assert p.t0 == pytest.approx(2459000.5 + 7 * 3.2 - 2459022.0)                          # nearest epoch, offset removed
    bare = archives.fetch_planet("TEST-1 b", fetch=lambda u: _csv(pl_ratdor=""))
    with pytest.raises(archives.ArchiveError, match="no a/R"):
        archives.transit_params_from_archive(bare, 2459022.9, 2459022.0)


def _x1dints_with_grid_offsets(path, offsets_pix, n=60, seed=0):
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    base = np.linspace(2.4, 5.2, 400)
    pixel = float(np.median(np.diff(base)))
    hdus = [fits.PrimaryHDU()]
    for i in range(n):
        w = (base + offsets_pix[i] * pixel).astype(np.float32).astype(np.float64)   # float32 storage
        f = 1.0 + rng.normal(0.0, 1e-3, base.size)
        cols = [fits.Column(name="WAVELENGTH", format="D", unit="um", array=w),
                fits.Column(name="FLUX", format="D", array=f),
                fits.Column(name="FLUX_ERROR", format="D", array=np.full(f.size, 1e-3))]
        hdus.append(fits.BinTableHDU.from_columns(cols, name="EXTRACT1D"))
    times = 2460000.5 + np.linspace(0.0, 0.1, n)
    hdus.append(fits.BinTableHDU.from_columns([fits.Column(name="int_mid_BJD_TDB", format="D", array=times)],
                                              name="INT_TIMES"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_sub_pixel_grid_differences_are_placed_on_one_grid(tmp_path):
    """Integrations differing by fractions of a pixel (float32 storage, recalibration) must load."""
    rng = np.random.default_rng(5)
    path = _x1dints_with_grid_offsets(tmp_path / "subpixel_x1dints.fits", rng.uniform(-0.3, 0.3, 60))
    series = load_x1dints(path)
    assert series.n_times == 60 and series.n_wavelengths >= 390          # edge channels may fall off
    check = series.meta["wavelength_grid_check"]
    assert check["integrations_interpolated"] > 0 and check["largest_shift_pixels"] <= 0.5


def test_multi_pixel_grid_differences_are_refused_with_the_offending_integration(tmp_path):
    offsets = np.zeros(60); offsets[30] = 2.0
    path = _x1dints_with_grid_offsets(tmp_path / "segments_x1dints.fits", offsets)
    with pytest.raises(TimeSeriesError, match="integration 30 is shifted by 2.00 pixels"):
        load_x1dints(path)
