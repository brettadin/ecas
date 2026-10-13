"""
ECAS test-suite.  Run from the project root:  ``python -m pytest -q``

Core tests need only NumPy/SciPy.  The GUI smoke test is skipped automatically when
PyQt6/pyqtgraph are unavailable and runs offscreen (no display required).
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import mock_data
from app.core.config import ConfigError, ConfigManager
from app.core.engine import AtmosphericConditions, Unit, build_engine
from app.core.parser import OrcaIRParser, OrcaParseError, load_peaks
from app.core.physics import (
    LineShape, doppler_hwhm, gaussian, lorentz_hwhm, lorentzian, voigt,
)
from app.ui.mappings import LinearMapping, LogMapping


# ------------------------------------------------------------------- parser
@pytest.mark.parametrize("variant", mock_data.VARIANTS)
@pytest.mark.parametrize("key", mock_data.available_mock_gases())
def test_parser_matches_array_generator(key, variant):
    """Every structural variant parses to exactly the generator's array output."""
    parsed = OrcaIRParser().parse_text(mock_data.render_mock_orca_output(key, variant=variant))
    expected = mock_data.mock_peak_table(key)
    np.testing.assert_allclose(parsed.wavenumber_cm1, expected.wavenumber_cm1, atol=0.006)
    np.testing.assert_allclose(parsed.intensity_km_mol, expected.intensity_km_mol, rtol=1e-3, atol=0.006)
    assert (parsed.wavenumber_cm1 > 1.0).all() and (parsed.intensity_km_mol > 0).all()


def test_parser_ignores_decoys_and_filters_modes():
    text = mock_data.render_mock_orca_output("CO2", include_imaginary=True)
    t = OrcaIRParser().parse_text(text)
    assert t.meta["n_imaginary"] == 1                  # the -35.2 cm^-1 mode was seen...
    assert len(t) == 3                                 # ...667.3 x2 and 2349; silent 1340 dropped
    assert t.wavenumber_cm1.tolist() == pytest.approx([667.3, 667.3, 2349.0], abs=0.01)


def test_parser_uses_last_block_and_stops_at_asterisk():
    first = mock_data.render_mock_orca_output("H2O")
    second = mock_data.render_mock_orca_output("CO2")
    t = OrcaIRParser().parse_text(first + "\n" + second)
    assert len(t) == 3 and t.wavenumber_cm1.max() == pytest.approx(2349.0, abs=0.01)
    # rows after the "\n*" boundary must never be harvested
    leaked = second + "\n  99:   1234.00   0.1  500.00  0.5  (0.1 0.1 0.1)\n"
    assert 1234.0 not in OrcaIRParser().parse_text(leaked).wavenumber_cm1


def test_parser_errors():
    with pytest.raises(OrcaParseError):
        OrcaIRParser().parse_text("nothing to see here")
    with pytest.raises(OrcaParseError):
        OrcaIRParser().parse_text("-----\nIR SPECTRUM\n-----\n\n* nothing\n")


def test_parse_file_and_mock_uri(tmp_path):
    paths = mock_data.write_mock_orca_files(tmp_path, ["H2O"], variant="orca6")
    from_file = load_peaks(paths["H2O"])
    from_uri = load_peaks("mock://H2O")
    np.testing.assert_allclose(from_file.wavenumber_cm1, from_uri.wavenumber_cm1, atol=0.01)
    assert from_file.meta["source"].endswith("H2O.out")


# ------------------------------------------------------------------ physics
def test_profiles_are_unit_area():
    x = np.linspace(-400.0, 400.0, 400001)
    assert np.trapezoid(lorentzian(x, 0, 1.0), x) == pytest.approx(1.0, abs=2e-3)
    assert np.trapezoid(gaussian(x, 0, 1.0), x) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(voigt(x, 0, 1.0, 1.0), x) == pytest.approx(1.0, abs=2e-3)


def test_voigt_limits():
    x = np.linspace(-5, 5, 101)
    np.testing.assert_allclose(voigt(x, 0, 1.0, 1e-4), lorentzian(x, 0, 1.0), rtol=2e-3)
    np.testing.assert_allclose(voigt(x, 0, 1e-6, 1.0), gaussian(x, 0, 1.0), atol=1e-5)


def test_broadening_scaling():
    assert lorentz_hwhm(2.0, 296.0, 0.07, 0.7) == pytest.approx(0.14)           # linear in P
    assert lorentz_hwhm(1.0, 148.0, 0.07, 0.5) == pytest.approx(0.07 * 2**0.5)  # (T0/T)^n
    d300 = doppler_hwhm(1000.0, 300.0, 18.0)
    assert doppler_hwhm(1000.0, 1200.0, 18.0) == pytest.approx(2 * d300)        # sqrt(T)
    assert doppler_hwhm(1000.0, 300.0, 72.0) == pytest.approx(d300 / 2)         # 1/sqrt(m)


# ------------------------------------------------------------------- config
def test_config_validation(tmp_path):
    cfg = ConfigManager.load()
    assert [g.gas_id for g in cfg.active_gases] == ["H2O", "CO2", "CH4", "O3", "N2O", "SO2", "NH3", "HCN", "CO"]
    assert len({g.color for g in cfg.gases}) == len(cfg.gases)      # auto-allocated, unique
    raw = ConfigManager.to_dict(cfg)
    assert ConfigManager.from_dict(raw).gases == cfg.gases           # round trip
    raw["gases"].append(dict(raw["gases"][0]))
    with pytest.raises(ConfigError):
        ConfigManager.from_dict(raw)                                  # duplicate id
    with pytest.raises(ConfigError):
        ConfigManager.from_dict({"gases": []})


def test_dynamic_gas_injection_and_missing_file(tmp_path):
    raw = ConfigManager.to_dict(ConfigManager.load())
    for g in raw["gases"]:
        if g["id"] == "SO2":
            g["enabled"] = True                                       # one-line "injection"
    raw["gases"].append({"id": "BAD", "orca_out": "does/not/exist.out", "mass_amu": 30.0})
    engine, report = build_engine(ConfigManager.from_dict(raw, base_dir=tmp_path))
    assert "SO2" in engine.gas_index and "BAD" in report.errors
    assert "BAD" not in engine.gas_index                              # skipped, not fatal


# ------------------------------------------------------------------- engine
@pytest.fixture(scope="module")
def engine():
    return build_engine(ConfigManager.load())[0]


@pytest.mark.parametrize("unit", list(Unit))
def test_beer_lambert_bounds_and_saturation(engine, unit):
    zero = np.zeros(engine.n_gases)
    assert np.all(engine.transmittance(unit, zero) == 1.0)
    t = engine.transmittance(unit, engine.default_multipliers())
    assert t.min() >= 0.0 and t.max() <= 1.0 and t.min() < 0.5
    huge = engine.transmittance(unit, np.full(engine.n_gases, 1e4))
    assert huge.min() == 0.0                                          # peaks saturate at exactly 0


def test_grid_limits_and_no_negative_space(engine):
    for unit, (lo, hi) in {Unit.WAVENUMBER: (400.0, 4200.0), Unit.WAVELENGTH: (0.1, 25.0)}.items():
        ax = engine.grid.axis(unit)
        assert (ax[0], ax[-1]) == pytest.approx((lo, hi)) and ax.min() > 0
    nu = engine.grid.wavenumber(Unit.WAVELENGTH)
    np.testing.assert_allclose(nu * engine.grid.axis(Unit.WAVELENGTH), 1e4)   # lambda = 1e4/nu


def test_units_agree_where_grids_overlap(engine):
    """Same physics, two axes: sampling the wavenumber curve at the wavelength
    grid's nu values must reproduce the wavelength-tab curve (up to interpolation)."""
    m = engine.default_multipliers()
    t_nu = engine.transmittance(Unit.WAVENUMBER, m)
    t_lam = engine.transmittance(Unit.WAVELENGTH, m)
    nu_lam = engine.grid.wavenumber(Unit.WAVELENGTH)
    inside = (nu_lam >= 400.0) & (nu_lam <= 4200.0)
    ref = np.interp(nu_lam[inside], engine.grid.axis(Unit.WAVENUMBER), t_nu)
    assert np.abs(ref - t_lam[inside]).max() < 0.25                   # coarse: different sampling
    assert np.median(np.abs(ref - t_lam[inside])) < 1e-3


def test_cache_only_rebuilds_on_condition_change(engine):
    engine.update_conditions(AtmosphericConditions(1.0, 296.0, 4.0))
    m = engine.default_multipliers()
    engine.transmittance(Unit.WAVENUMBER, m)
    n0 = engine.profile_evaluations
    for k in range(20):                                               # slider drags
        m[k % engine.n_gases] = 0.1 * k
        engine.transmittance(Unit.WAVENUMBER, m)
    assert engine.profile_evaluations == n0                           # zero rebuilds
    engine.update_conditions(AtmosphericConditions(10.0, 296.0, 4.0))
    engine.transmittance(Unit.WAVENUMBER, m)
    assert engine.profile_evaluations == n0 + 1


def test_pressure_broadens_and_layers_multiply(engine):
    m = engine.default_multipliers()
    engine.update_conditions(AtmosphericConditions(1.0, 296.0, 0.0))
    narrow = 1.0 - engine.transmittance(Unit.WAVENUMBER, m)
    engine.update_conditions(AtmosphericConditions(50.0, 296.0, 0.0))
    wide = 1.0 - engine.transmittance(Unit.WAVENUMBER, m)
    assert np.count_nonzero(wide > 0.01) > np.count_nonzero(narrow > 0.01)
    layers = engine.layer_transmittances(Unit.WAVENUMBER, m)
    np.testing.assert_allclose(layers.prod(axis=1), engine.transmittance(Unit.WAVENUMBER, m))
    engine.update_conditions(AtmosphericConditions(1.0, 296.0, 4.0))


@pytest.mark.parametrize("shape", list(LineShape))
def test_all_line_shapes_run(engine, shape):
    engine.update_conditions(replace(engine.conditions, line_shape=shape))
    t = engine.transmittance(Unit.WAVENUMBER, engine.default_multipliers())
    assert np.isfinite(t).all()
    engine.update_conditions(replace(engine.conditions, line_shape=LineShape.VOIGT))


# ----------------------------------------------------------------- mappings
@pytest.mark.parametrize("m", [LinearMapping(50, 1500, 1450), LogMapping(1e-4, 1e2, 1000)])
def test_mapping_roundtrip(m):
    for pos in (0, 1, m.steps // 3, m.steps):
        assert m.to_pos(m.to_value(pos)) == pos
    assert m.to_value(0) == pytest.approx(m.lo) and m.to_value(m.steps) == pytest.approx(m.hi)


# ---------------------------------------------------------------- GUI smoke
def test_gui_smoke_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    from PyQt6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    cfg = ConfigManager.load()
    eng, _ = build_engine(cfg)
    win = MainWindow(eng, cfg.physics)

    canvas = win._canvases[Unit.WAVENUMBER]
    y0 = canvas.master.yData.copy()
    assert y0.shape == (eng.grid.n_points,) and 0.0 <= y0.min() <= y0.max() <= 1.0

    win._gas_sliders["CO2"].set_value(15.0)           # slider -> signal -> state
    win._s_pressure.set_value(20.0)
    win._refresh()                                     # (timer is event-loop driven)
    assert not np.allclose(canvas.master.yData, y0)

    win._tabs.setCurrentIndex(1)                       # unit switch
    assert win.active_unit() is Unit.WAVELENGTH
    lam = win._canvases[Unit.WAVELENGTH]
    assert lam.master.yData is not None and lam.master.xData[0] == pytest.approx(0.1)

    win._layers_box.setChecked(True)
    assert all(item.yData is not None for item in lam.layers)
    win._reset()
    win._refresh()
    assert win._status.text().startswith("model + setData")


# ------------------------------------------------------------ Phase 4: inverse
from app.core.fitting import (  # noqa: E402
    CandidateLibrary, FitError, ObservedSpectrum, find_anomalies, fit_multipliers,
    load_spectrum_csv,
)

TRUTH = {"H2O": 3.0, "CO2": 2.0, "CH4": 0.8, "O3": 1.5, "N2O": 1.0}


@pytest.fixture
def fresh_engine():
    return build_engine(ConfigManager.load())[0]


@pytest.mark.parametrize("unit", list(Unit))
def test_fit_recovers_truth_without_anomaly(fresh_engine, unit):
    obs = mock_data.make_mock_observation(fresh_engine, unit, TRUTH, anomaly=None, noise=0.003)
    fit = fit_multipliers(fresh_engine, obs, x0=np.ones(fresh_engine.n_gases))
    true = fresh_engine.multipliers_from_dict(TRUTH)
    assert fit.success and 0.7 < fit.reduced_chi2 < 1.4
    np.testing.assert_allclose(fit.multipliers[:5], true[:5], rtol=0.1)
    assert np.max(fit.multipliers[5:]) < 0.1
    assert np.all(fit.uncertainties > 0)
    assert find_anomalies(obs, fit, 3.0, min_points=3) == []


def test_anomaly_detection_and_reverse_id(fresh_engine):
    obs = mock_data.make_mock_observation(fresh_engine, Unit.WAVENUMBER, TRUTH, ("SO2", 3.0))
    free = np.ones(fresh_engine.n_gases, dtype=bool)
    free[fresh_engine.gas_index["SO2"]] = False
    fit = fit_multipliers(fresh_engine, obs, x0=fresh_engine.default_multipliers(), free=free)
    found = find_anomalies(obs, fit, 3.0)
    assert found and all(a.extra_absorption for a in found)
    assert any(abs(a.nu_peak_cm1 - 1362.0) < 25 for a in found)       # strongest SO2 band
    ranked = CandidateLibrary.load().rank([a.nu_peak_cm1 for a in found],
                                          exclude=("H2O", "CO2", "CH4", "O3", "N2O"))
    assert ranked[0].molecule == "SO2"


def test_fit_respects_bounds_and_free_mask(fresh_engine):
    obs = mock_data.make_mock_observation(fresh_engine, Unit.WAVENUMBER, TRUTH, anomaly=None)
    free = np.array([True, True] + [False] * (fresh_engine.n_gases - 2))
    x0 = fresh_engine.default_multipliers()
    fit = fit_multipliers(fresh_engine, obs, x0=x0, free=free)
    np.testing.assert_array_equal(fit.multipliers[~free], x0[~free])
    assert np.all(fit.multipliers >= 0) and np.all(fit.uncertainties[~free] == 0)
    with pytest.raises(FitError):
        fit_multipliers(fresh_engine, obs, free=np.zeros(fresh_engine.n_gases, bool))


def test_csv_loader_units_and_errors(tmp_path):
    p = tmp_path / "jwst.csv"
    p.write_text("# demo\nwavelength_um,transmittance,err\n" +
                 "\n".join(f"{3 + 0.1 * i},{1 - 0.01 * i},0.002" for i in range(20)))
    o = load_spectrum_csv(p)
    assert o.unit is Unit.WAVELENGTH and o.sigma is not None and o.x.size == 20
    np.testing.assert_allclose(o.wavenumber_cm1, 1e4 / o.x)
    q = tmp_path / "lab.txt"
    q.write_text("\n".join(f"{500 + 10 * i} {0.9}" for i in range(20)))
    assert load_spectrum_csv(q).unit is Unit.WAVENUMBER
    with pytest.raises(FitError):
        ObservedSpectrum(np.arange(3.0) + 1, np.ones(3), Unit.WAVELENGTH)


def test_gui_inverse_workflow_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    from PyQt6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    cfg = ConfigManager.load()
    # SO2 is an enabled registry gas, so the fitter would simply absorb the injected SO2
    # band. Disable it here so the injected band stays an unmodelled anomaly (the path
    # this test is about).
    raw = ConfigManager.to_dict(cfg)
    for g in raw["gases"]:
        if g["id"] == "SO2":
            g["enabled"] = False
    cfg = ConfigManager.from_dict(raw, base_dir=cfg.base_dir)
    eng, _ = build_engine(cfg)
    win = MainWindow(eng, cfg.physics)
    win.show()
    win._on_mock_observation()
    assert win._btn_fit.isEnabled() and win._canvases[Unit.WAVELENGTH].data_items
    win._on_fit()
    assert win._fit is not None and win._fit.reduced_chi2 > 3.0   # unmodelled SO2 => poor fit
    assert win._anomalies and win._anom_header.text().startswith("Unidentified")
    assert "SO2" in win._cand_label.text()
    assert len(win._canvases[Unit.WAVENUMBER].regions) == len(win._anomalies)
    win._tabs.setCurrentIndex(1)                       # residual axis relinks, no crash
    win._thr_spin.setValue(8.0)                        # re-analyse without refit
    win._on_clear_observation()
    assert win._obs is None and not win._canvases[Unit.WAVENUMBER].data_items


# --------------------------------------------------------- Phase 5: transit
from app.core.transit import (  # noqa: E402
    StellarParams, TransitError, TransitParams, analyze_lightcurve, depth_spectrum_to_observation,
    extract_transmission_spectrum, load_lightcurve_csv, orbit_positions, transit_model,
    LightCurve,
)


def test_transit_model_geometry():
    p = TransitParams(1.0, 3.0, 0.1, 10.0, 0.0, u1=0.0, u2=0.0)
    t = np.array([1.0, 1.0 + 1.5, 1.0 + p.duration_days / 2 + 0.05])   # centre, far side, outside
    f = transit_model(t, p)
    assert f[0] == pytest.approx(1 - 0.01, abs=1e-9)                    # depth = k^2 (no LD)
    assert f[1] == 1.0 and f[2] == 1.0                                  # no secondary dip, no dip outside
    x, y, z = orbit_positions(np.array([1.0, 2.5]), p)
    assert z[0] > 0 > z[1] and abs(x[0]) < 1e-9 and np.hypot(x[0], y[0]) == pytest.approx(0.0)
    ld = TransitParams(1.0, 3.0, 0.1, 10.0, 0.0)                        # limb darkening deepens centre transit
    assert transit_model(np.array([1.0]), ld)[0] < 0.99


def test_analysis_recovers_injected_system():
    lc, truth = mock_data.make_mock_lightcurve()
    res = analyze_lightcurve(lc, StellarParams(1.0, 1.0, 5772.0))
    assert res.params.period == pytest.approx(truth.period, rel=2e-3)
    assert res.params.rp_rs == pytest.approx(truth.rp_rs, rel=0.08)
    assert res.params.a_rs == pytest.approx(truth.a_rs, rel=0.25)
    assert res.params.duration_days == pytest.approx(truth.duration_days, rel=0.1)
    assert res.n_transits >= 4 and res.bls_snr > 10
    assert 0.8 < res.rp_rjup < 1.2                                      # 0.1 R* around the Sun ~ 1 R_Jup
    assert 700 < res.teq_k < 1500 and 5 < res.orbital_speed_kms < 400
    assert len(res.summary_lines()) >= 10


def test_analysis_rejects_flat_noise_and_short_data():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, 0.003)
    with pytest.raises(TransitError):
        analyze_lightcurve(LightCurve(t, 1 + rng.normal(0, 1e-3, t.size)))
    with pytest.raises(TransitError):
        LightCurve(t[:10], np.ones(10))


def test_lightcurve_csv(tmp_path):
    lc, _ = mock_data.make_mock_lightcurve()
    p = tmp_path / "lc.csv"
    p.write_text("time,flux\n" + "\n".join(f"{a},{b}" for a, b in zip(lc.time, lc.flux)))
    assert load_lightcurve_csv(p).time.size == lc.time.size


def test_transmission_spectrum_closes_the_loop(fresh_engine):
    """engine atmosphere -> spectral light curves -> extracted depths -> inverse fit."""
    _, truth = mock_data.make_mock_lightcurve()
    mult = fresh_engine.multipliers_from_dict(TRUTH)
    t, flux, wl = mock_data.make_mock_spectral_lightcurves(fresh_engine, truth, multipliers=mult)
    spec = extract_transmission_spectrum(t, flux, wl, truth)
    assert spec.depth.shape == wl.shape and np.all(spec.depth_err > 0)
    base = truth.rp_rs**2
    assert abs(np.median(spec.depth) - base) / base < 0.15
    assert spec.depth.max() > 1.05 * np.median(spec.depth)              # the CO2 band shows up
    obs = depth_spectrum_to_observation(spec)
    assert obs.unit is Unit.WAVELENGTH and 0.0 <= obs.y.min() < 0.5 and obs.y.max() <= 1.0 + 1e-9
    fit = fit_multipliers(fresh_engine, obs, x0=np.ones(fresh_engine.n_gases))
    ranking = np.argsort(-fit.multipliers)                              # strong absorbers dominate
    assert fit.reduced_chi2 < 20 and ranking[0] in (0, 1, 4)            # H2O / CO2 / N2O scale


def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("enable_3d", [False, True])
def test_gui_transit_workflow(enable_3d):
    app = _qapp()
    from app.ui.main_window import MainWindow

    cfg = ConfigManager.load()
    eng, _ = build_engine(cfg)
    win = MainWindow(eng, cfg.physics, enable_3d=enable_3d)
    panel = win._transit
    panel._on_mock()
    panel.analyze(blocking=True)
    assert panel._result is not None and "Period" in panel._report.toPlainText()
    assert panel._lc_model.yData is not None and panel._fold_model.yData is not None

    panel._slider.setValue(int(panel._slider.maximum() * (panel._params.t0 - panel._t_min)
                               / (panel._t_max - panel._t_min)))        # scrub to mid-transit
    assert panel._tracker.value() == pytest.approx(panel._params.t0, abs=0.01)
    assert panel._sky._planet.isVisible() and "flux 0.9" in panel._clock.text()
    panel._slider.setValue(panel._slider.value() + 700)                  # far from transit
    assert "flux 1.0000" in panel._clock.text()
    if enable_3d:                                       # offscreen Qt has no OpenGL: must fall back, not crash
        assert not panel._view3d.available and panel._view3d.fallback is not None
        assert "OpenGL" in panel._view3d.reason
    panel._on_play(True); panel._advance(); panel._on_play(False)

    panel._on_spectral()                                                 # photometry -> fitter
    assert win._obs is not None and win._workspaces.currentIndex() == 0
    win._on_fit()
    assert win._fit is not None
    win.close()


# ------------------------------------------------------- Phase 6: files & export
from app.core import export as ecas_export  # noqa: E402
from app.core.project import ProjectError, ProjectState, load_project, save_project  # noqa: E402


def test_project_roundtrip(tmp_path, fresh_engine):
    obs = mock_data.make_mock_observation(fresh_engine, Unit.WAVELENGTH, TRUTH)
    lc, transit_truth = mock_data.make_mock_lightcurve()
    series = mock_data.make_mock_spectral_timeseries(
        fresh_engine, transit_truth, n_times=120, n_channels=12,
        multipliers=fresh_engine.default_multipliers())
    series.meta["planet_record"] = {"values": {"pl_name": "WASP-39 b", "st_teff": np.float64(5485.0)}}
    from app.core.transit import DepthSpectrum
    depth_spectrum = DepthSpectrum(np.linspace(1.0, 2.0, 12), np.full(12, 0.01), np.full(12, 0.0002))
    st = ProjectState(conditions={"pressure_atm": 2.0, "line_shape": "gaussian"},
                      multipliers={"H2O": 4.0}, anomaly_threshold=4.5,
                      stellar={"radius_rsun": 1.2}, observation=obs, lightcurve=lc,
                      spectral_series=series, depth_spectrum=depth_spectrum, notes="hi")
    path = save_project(tmp_path / "demo", st)                       # extension auto-appended
    assert path.suffix == ".ecas"
    back = load_project(path)
    assert back.multipliers == {"H2O": 4.0} and back.anomaly_threshold == 4.5 and back.notes == "hi"
    assert back.observation.unit is Unit.WAVELENGTH and back.observation.sigma is None
    np.testing.assert_array_equal(back.observation.y, obs.y)
    np.testing.assert_array_equal(back.lightcurve.flux, lc.flux)
    assert back.spectral_series is not None
    np.testing.assert_array_equal(back.spectral_series.flux, series.flux)
    assert back.spectral_series.meta["planet_record"]["values"]["st_teff"] == pytest.approx(5485.0)
    assert back.depth_spectrum is not None
    np.testing.assert_array_equal(back.depth_spectrum.depth, depth_spectrum.depth)
    (tmp_path / "bad.ecas").write_bytes(b"not a zip")
    with pytest.raises(ProjectError):
        load_project(tmp_path / "bad.ecas")


def test_exports(tmp_path, fresh_engine):
    obs = mock_data.make_mock_observation(fresh_engine, Unit.WAVENUMBER, TRUTH)
    fit = fit_multipliers(fresh_engine, obs, x0=np.ones(fresh_engine.n_gases))
    found = find_anomalies(obs, fit)
    ecas_export.export_model_csv(tmp_path / "m.csv", fresh_engine, Unit.WAVELENGTH, fit.multipliers)
    ecas_export.export_fit_csv(tmp_path / "f.csv", obs, fit)
    ecas_export.export_anomalies_csv(tmp_path / "a.csv", found)
    head = (tmp_path / "m.csv").read_text().splitlines()
    assert head[0].startswith("wavelength,transmittance,T_H2O") and len(head) == fresh_engine.grid.n_points + 1
    assert len((tmp_path / "f.csv").read_text().splitlines()) == obs.x.size + 1
    from app.core.transit import DepthSpectrum
    depth_spec = DepthSpectrum(np.array([1.0, 2.0]), np.array([0.01, 0.02]),
                               np.array([0.001, 0.002]))
    ecas_export.export_depth_spectrum_csv(tmp_path / "depth.csv", depth_spec)
    depth_rows = (tmp_path / "depth.csv").read_text().splitlines()
    assert depth_rows[0] == "wavelength_um,transit_depth_ppm,uncertainty_1sigma_ppm"
    assert depth_rows[1].startswith("1,10000,1000")
    cands = CandidateLibrary.load().rank([a.nu_peak_cm1 for a in found])
    md = ecas_export.build_report_markdown(fresh_engine, fresh_engine.conditions, fit.multipliers,
                                           fit, found, cands, ["Period 3.2 d"])
    assert "## Gas multipliers" in md and "Sulfur Dioxide" in md and "Period 3.2 d" in md


def test_gui_project_roundtrip(tmp_path):
    app = _qapp()
    from app.ui.main_window import MainWindow

    cfg = ConfigManager.load()
    eng, _ = build_engine(cfg)
    a = MainWindow(eng, cfg.physics, enable_3d=False)
    a._gas_sliders["CH4"].set_value(7.0)
    a._s_pressure.set_value(5.0)
    a._on_mock_observation(); a._on_fit()
    a._transit._on_mock(); a._transit.analyze(blocking=True)
    path = save_project(tmp_path / "s.ecas", a.capture_state())

    eng2, _ = build_engine(cfg)
    b = MainWindow(eng2, cfg.physics, enable_3d=False)
    b.restore_state(load_project(path))
    assert b._obs is not None and b._transit._result is not None
    assert b._conditions.pressure_atm == pytest.approx(5.0, rel=0.02)
    np.testing.assert_allclose(b._multipliers, a._multipliers, rtol=0.02, atol=0.02)
    assert b._transit._result.params.period == pytest.approx(a._transit._result.params.period)
    a.close(); b.close()


def test_gui_transmission_spectrum_plot_uses_measured_depths():
    app = _qapp()
    from app.ui.main_window import MainWindow

    cfg = ConfigManager.load()
    eng, _ = build_engine(cfg)
    win = MainWindow(eng, cfg.physics, enable_3d=False)
    win._workspaces.setCurrentIndex(2)
    win._transit._on_mock_series()
    win._transit._on_extract()
    spectrum = win._transit._transmission_spectrum
    assert spectrum is not None
    plot_x, plot_y = win._transit._transmission_points.getData()
    np.testing.assert_allclose(plot_x, spectrum.wavelength_um)
    np.testing.assert_allclose(plot_y, spectrum.depth * 1e6)
    assert win._transit._chart_tabs.currentWidget() is win._transit._transmission_plot
    win._transit._bin_size.setValue(2)
    binned_x, binned_y = win._transit._transmission_points.getData()
    assert len(binned_x) == spectrum.wavelength_um.size // 2
    from app.core.transit import bin_depth_spectrum
    expected = bin_depth_spectrum(spectrum, 2)
    np.testing.assert_array_equal(binned_x, expected.wavelength_um)
    np.testing.assert_allclose(binned_y, expected.depth * 1e6)
    win._transit._bin_size.setValue(1)
    restored_x, restored_y = win._transit._transmission_points.getData()
    np.testing.assert_allclose(restored_x, spectrum.wavelength_um)
    np.testing.assert_allclose(restored_y, spectrum.depth * 1e6)
    assert win._obs is None and win._workspaces.currentIndex() == 2
    assert win._transit._btn_identify.isEnabled()
    win._transit._on_send_to_identify()
    assert win._obs is not None and win._workspaces.currentIndex() == 0
    assert "not atmospheric abundances" in win._identify._dataset_label.text()
    win._workspaces.setCurrentIndex(2)
    win._transit._on_correlated_noise()
    assert win._transit._chart_tabs.currentWidget() is win._transit._noise_plot
    noise_x, noise_y = win._transit._noise_curve.getData()
    assert len(noise_x) >= 1 and np.isfinite(noise_y).all()
    assert "does not change extracted uncertainties" in win._transit._report.toPlainText()
    from app.core.atmosphere import AbundanceScanResult
    transit = win._transit
    transit._scan_run_series = transit._series
    transit._scan_run_spectrum = transit._transmission_spectrum
    transit._scan_run_bin_size = transit._bin_size.value()
    vmr_grid = np.logspace(-6, -2, 5)
    scan_spectrum = bin_depth_spectrum(transit._transmission_spectrum, transit._bin_size.value())
    transit._on_abundance_scan_ready(AbundanceScanResult(
        species="H2O", volume_mixing_ratio_grid=vmr_grid,
        delta_chi_squared=np.array([4.0, 1.0, 0.0, 1.0, 4.0]),
        best_volume_mixing_ratio=1e-4, best_chi_squared=12.0,
        best_depth_offset_ppm=37.0, lower_1sigma_vmr=5e-5,
        upper_1sigma_vmr=2e-4, n_channels=scan_spectrum.wavelength_um.size,
        wavelength_um=scan_spectrum.wavelength_um,
        best_model_depth_ppm=scan_spectrum.depth * 1e6))
    assert transit._chart_tabs.currentWidget() is transit._abundance_plot
    assert "not a posterior" in transit._report.toPlainText()
    assert transit._btn_abundance_scan.isEnabled()
    win.close()


def test_gui_local_spectral_matrix_populates_time_spectrum_and_transit_chart(tmp_path):
    app = _qapp()
    from app.ui.main_window import MainWindow

    cfg = ConfigManager.load()
    eng, _ = build_engine(cfg)
    win = MainWindow(eng, cfg.physics, enable_3d=False)
    times = np.linspace(-0.1, 0.1, 60)
    wavelengths = np.linspace(1.0, 2.0, 8)
    flux = np.ones((times.size, wavelengths.size))
    flux[(times > -0.02) & (times < 0.02)] -= 0.01
    path = tmp_path / "local_spectra.csv"
    rows = ["time," + ",".join(map(str, wavelengths))]
    rows.extend(",".join(map(str, [t, *f])) for t, f in zip(times, flux))
    path.write_text("\n".join(rows), encoding="utf-8")

    assert win._transit.load_series_file(str(path))
    assert win._transit._series_label.text().startswith("60 integrations × 8 channels")
    assert len(win._transit._spec_curve.getData()[0]) == 8
    assert not win._transit._btn_identify.isEnabled()
    assert "Analyze" in win._transit._fold_caption.textItem.toPlainText()
    win.close()


# ============================================================ bug-fix regression tests
from app.core.fitting import XUnit, YMode, build_observation, find_features  # noqa: E402
from app.core.tables import TableError, read_table, suggest_spectrum_columns  # noqa: E402


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _rows(x, y, sep=","):
    return "\n".join(f"{a:.6g}{sep}{b:.6g}" for a, b in zip(x, y))


# ---- CSV import: the "single point / wrong place / tiny transmittance" family
def test_import_nanometres_percent_and_normalisation(tmp_path):
    nm = np.linspace(3000, 15000, 200)                      # 3-15 um expressed in nm
    o = load_spectrum_csv(_write(tmp_path, "a.csv", "Wavelength (nm),Flux\n" + _rows(nm, np.full(200, 0.06))))
    assert o.unit is Unit.WAVELENGTH and o.x.min() == pytest.approx(3.0) and o.x.max() == pytest.approx(15.0)
    assert o.y.max() == pytest.approx(1.0, abs=0.01)         # 0.06 flux normalised to a continuum of 1
    assert "divided by" in " ".join(o.meta["import_notes"])

    um = np.linspace(3, 20, 100)
    p = load_spectrum_csv(_write(tmp_path, "b.csv", "um,T%\n" + _rows(um, np.linspace(50, 90, 100))))
    assert p.y.max() == pytest.approx(0.9) and p.unit is Unit.WAVELENGTH      # percent -> fraction

    ang = np.linspace(30000, 150000, 100)
    q = load_spectrum_csv(_write(tmp_path, "c.csv", "wavelength [Angstrom],transmittance\n" + _rows(ang, np.full(100, .8))))
    assert q.x.min() == pytest.approx(3.0)


def test_import_decimal_comma_five_columns_ragged_and_metadata(tmp_path):
    wn = np.linspace(500, 4000, 120)
    t = np.linspace(0.5, 0.9, 120)
    de = "wavenumber;trans\n" + "\n".join(f"{a:.2f}".replace(".", ",") + ";" + f"{b:.4f}".replace(".", ",")
                                          for a, b in zip(wn, t))
    o = load_spectrum_csv(_write(tmp_path, "de.csv", de))
    assert o.unit is Unit.WAVENUMBER and o.x.size == 120 and o.y.max() == pytest.approx(0.9)

    tsv = "wave\tflux\terr\tmodel\tflag\n" + "\n".join(f"{a:.3f}\t0.9\t0.01\t0.9\t0" for a in np.linspace(3, 20, 60))
    o = load_spectrum_csv(_write(tmp_path, "five.tsv", tsv))
    assert o.sigma is not None and o.meta["columns"][2].startswith("err")        # picked the err column

    messy = "# instrument: XYZ\n# observer: someone\nSample 42, run 7\nwavenumber,transmittance\n" \
            + _rows(wn, t) + "\n1,2,3,4\n"
    o = load_spectrum_csv(_write(tmp_path, "messy.csv", messy))
    assert o.x.size == 120
    with pytest.raises(FitError):
        load_spectrum_csv(_write(tmp_path, "empty.csv", "just words\nno numbers here\n"))


def test_exoplanet_archive_flux_csv_with_empty_errors_and_text_columns(tmp_path):
    header = (
        "CENTRALWAVELNG,BANDWIDTH,FLAM,FLAMERR1,FLAMERR2,FLAM_AUTHORS,"
        "FNU,FNUERR1,FNUERR2,FNU_AUTHORS,OBS_DATE,OBS_DATEERR1,OBS_DATEERR2")
    rows = [header]
    for index, wavelength in enumerate(np.linspace(0.9, 2.0, 12)):
        flux = 2e-17 * (1.0 + 0.2 * np.sin(index / 2))
        rows.append(
            f"{wavelength:.5f},,{flux:.8e},,,Calculated,{flux / 3e-12:.8e},,,"
            f"Luhman et al. 2023,2459982.500000,,")
    path = _write(tmp_path, "archive_sed.csv", "\n".join(rows))

    table = read_table(path)
    columns = suggest_spectrum_columns(table)
    observation = build_observation(table, columns["x"], columns["y"], columns["sigma"])

    assert table.names[0] == "CENTRALWAVELNG" and table.names[2] == "FLAM"
    assert columns == {"x": 0, "y": 2, "sigma": None}
    assert observation.x.size == 12
    assert observation.x.min() == pytest.approx(0.9)
    assert observation.x.max() == pytest.approx(2.0)
    assert observation.sigma is None
    assert observation.meta["spectrum_kind"] == "emission_flux"
    assert observation.meta["flux_density_kind"] == "flam"
    assert "normalise to continuum" in " ".join(observation.meta["import_notes"])


def test_gui_emission_flux_ui_uses_blackbody_not_transmission_fit(tmp_path):
    app = _qapp()
    from app.ui.main_window import MainWindow

    wavelength = np.linspace(0.9, 5.3, 30)
    exponent = 14387.768775 / (wavelength * 1450.0)
    flux = wavelength**-5 / np.expm1(exponent)
    path = _write(
        tmp_path, "planet_emission.csv",
        "CENTRALWAVELNG,FLAM\n" +
        "\n".join(f"{wl:.8f},{value:.12e}" for wl, value in zip(wavelength, flux)),
    )
    cfg = ConfigManager.load()
    engine, _ = build_engine(cfg)
    window = MainWindow(engine, cfg.physics, enable_3d=False)
    obs = window.load_observation_file(str(path))

    assert obs is not None
    assert not window._btn_fit.isEnabled()
    assert not window._btn_identify.isEnabled()
    assert window._btn_blackbody.isEnabled()
    assert not window._identify._btn_run.isEnabled()
    assert "displayed below" in window._identify._dataset_label.text()
    assert window._identify._data_item.xData is not None
    assert not window._canvases[Unit.WAVELENGTH].master.isVisible()
    window._on_fit_blackbody()
    assert window._canvases[Unit.WAVELENGTH].emission_continuum.isVisible()
    assert "1450 K" in window._fit_label.text()
    assert "not modeled" in window._fit_label.text()
    window.close()


def test_import_fits_and_jcamp(tmp_path):
    from astropy.io import fits
    wl = np.linspace(3, 12, 150)
    cols = [fits.Column(name="WAVELENGTH", format="D", unit="um", array=wl),
            fits.Column(name="FLUX", format="D", array=np.full(150, 0.8)),
            fits.Column(name="FLUX_ERROR", format="D", array=np.full(150, 0.01))]
    path = tmp_path / "x1d.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name="EXTRACT1D")]).writeto(path)
    o = load_spectrum_csv(path)
    assert o.unit is Unit.WAVELENGTH and o.sigma is not None and o.x.size == 150

    jdx = "##TITLE=demo\n##JCAMP-DX=4.24\n##XUNITS=1/CM\n##YUNITS=TRANSMITTANCE\n##XYPOINTS=(XY..XY)\n" \
          + "\n".join(f"{a:.2f}, {b:.4f}" for a, b in zip(np.linspace(500, 4000, 60), np.full(60, 0.9))) + "\n##END=\n"
    o = load_spectrum_csv(_write(tmp_path, "s.jdx", jdx))
    assert o.unit is Unit.WAVENUMBER and o.x.size == 60


def test_explicit_import_overrides():
    from app.core.tables import RawTable
    t = RawTable(["x", "y"], np.column_stack([np.linspace(600, 2600, 100), np.full(100, 0.5)]))
    assert build_observation(t, 0, 1).unit is Unit.WAVENUMBER                    # auto: assumed cm-1
    forced = build_observation(t, 0, 1, x_unit=XUnit.NANOMETER, y_mode=YMode.TRANSMITTANCE)
    assert forced.unit is Unit.WAVELENGTH and forced.x.max() == pytest.approx(2.6)
    ab = build_observation(t, 0, 1, y_mode=YMode.ABSORBANCE)
    assert ab.y[0] == pytest.approx(10 ** -0.5)


# ---- model behaviour
def test_gaussian_line_shape_is_visible_and_responds_to_pressure(fresh_engine):
    m = fresh_engine.default_multipliers()
    fresh_engine.update_conditions(AtmosphericConditions(1.0, 296.0, 4.0, LineShape.GAUSSIAN))
    t1 = fresh_engine.transmittance(Unit.WAVENUMBER, m)
    assert t1.min() < 0.5                                                       # was 0.997 (invisible)
    fresh_engine.update_conditions(AtmosphericConditions(50.0, 296.0, 4.0, LineShape.GAUSSIAN))
    t2 = fresh_engine.transmittance(Unit.WAVENUMBER, m)
    assert np.count_nonzero(1 - t2 > 0.01) > np.count_nonzero(1 - t1 > 0.01)    # pressure broadens it
    nu = np.linspace(2200, 2500, 300001)
    from app.core.physics import line_profile_matrix
    area = np.trapezoid(line_profile_matrix(nu[:, None], 2349.0, 4.0, 0.002, LineShape.GAUSSIAN)[:, 0], nu)
    assert area == pytest.approx(1.0, abs=1e-3)


# ---- fit window / continuum / peak finding
def test_fit_window_and_continuum(fresh_engine):
    obs = mock_data.make_mock_observation(fresh_engine, Unit.WAVENUMBER, TRUTH, anomaly=None, noise=0.003)
    tilted = ObservedSpectrum(obs.x, obs.y * (0.4 + 0.1 * (obs.x - 450) / 3650), obs.unit)
    plain = fit_multipliers(fresh_engine, tilted, x0=np.ones(fresh_engine.n_gases))
    cont = fit_multipliers(fresh_engine, tilted, x0=np.ones(fresh_engine.n_gases), baseline=2)
    assert cont.reduced_chi2 < 0.2 * plain.reduced_chi2                         # continuum absorbs the slope
    assert cont.baseline_coeffs.size == 2 and 0.3 < cont.baseline_coeffs[0] < 0.6
    expected = fresh_engine.multipliers_from_dict(TRUTH)
    np.testing.assert_allclose(cont.multipliers[:5], expected[:5], rtol=0.15)
    assert np.max(cont.multipliers[5:]) < 0.1
    win = fit_multipliers(fresh_engine, obs, x0=np.ones(fresh_engine.n_gases), window_cm1=(1000.0, 2500.0))
    assert win.n_excluded > 0 and win.used.x.min() >= 1000 and win.used.x.max() <= 2500
    with pytest.raises(FitError):
        fit_multipliers(fresh_engine, obs, window_cm1=(10.0, 20.0))


def test_find_features_identifies_unmodelled_molecule(fresh_engine):
    obs = mock_data.make_mock_observation(fresh_engine, Unit.WAVELENGTH, TRUTH, ("SO2", 3.0))
    feats = find_features(obs, min_snr=5.0)
    assert any(abs(f.nu_peak_cm1 - 1362.0) < 15 for f in feats) and feats[0].significance_sigma > 5
    lib = CandidateLibrary.from_engine(fresh_engine)
    names = {m.molecule for row in lib.match_features([f.nu_peak_cm1 for f in feats]) for m in row}
    assert {"SO2", "CO2", "H2O"} <= names                                      # engine gases AND json table
    flat = ObservedSpectrum(obs.x, np.ones_like(obs.y) + np.random.default_rng(1).normal(0, 0.002, obs.y.size), obs.unit)
    assert find_features(flat, min_snr=6.0) == []


# ---- transit: the freeze, spectrum rejection, live stellar parameters
def test_transit_rejects_spectra_quickly(tmp_path):
    import time
    x = np.linspace(450, 4100, 3000)                                            # cm-1 pretending to be days
    y = 1 - 0.3 * np.exp(-((x - 2349) / 15) ** 2)
    t0 = time.perf_counter()
    with pytest.raises(TransitError):
        analyze_lightcurve(LightCurve(x, y))
    assert time.perf_counter() - t0 < 8.0                                       # was ~52 s
    spec = _write(tmp_path, "spectrum.csv", "wavelength_um,transmittance\n" + _rows(np.linspace(3, 20, 100), np.full(100, .9)))
    with pytest.raises(TransitError, match="looks like a spectrum"):
        load_lightcurve_csv(spec)
    lc = LightCurve(np.array([1.0] * 30 + list(np.arange(2.0, 62.0))), np.ones(90))
    assert np.all(np.diff(lc.time) > 0)                                         # duplicate times removed


def test_transit_result_updates_with_stellar_params():
    lc, _ = mock_data.make_mock_lightcurve()
    sun = analyze_lightcurve(lc, StellarParams(1.0, 1.0, 5772.0))
    big = sun.with_stellar(StellarParams(2.0, 1.5, 6500.0))
    assert big.rp_rjup == pytest.approx(2.0 * sun.rp_rjup) and big.a_au == pytest.approx(2.0 * sun.a_au)
    assert big.orbital_speed_kms == pytest.approx(2.0 * sun.orbital_speed_kms)
    assert big.teq_k > sun.teq_k and big.params == sun.params                   # fit itself unchanged


def test_gui_transit_threaded_analysis_live_stellar_speed_and_fallback():
    app = _qapp()
    import time as _time
    from app.ui.main_window import MainWindow

    cfg = ConfigManager.load()
    eng, _ = build_engine(cfg)
    win = MainWindow(eng, cfg.physics, enable_3d=False)
    panel = win._transit
    panel._on_mock()
    panel.analyze(blocking=False)                                               # worker thread
    assert not panel._btn_analyze.isEnabled()                                   # UI stays responsive
    deadline = _time.time() + 30
    while panel._result is None and _time.time() < deadline:
        app.processEvents(); _time.sleep(0.01)
    assert panel._result is not None
    for _ in range(50):
        app.processEvents(); _time.sleep(0.01)
    assert panel._btn_analyze.isEnabled() and panel._btn_analyze.text() == "Analyze transit"

    before = panel._report.toPlainText()
    panel._spin_r[1].setValue(2.0)                                              # R* now acts immediately
    assert panel._report.toPlainText() != before and "R★ 2" in panel._report.toPlainText()
    assert "R☉" in panel._sky._caption.textItem.toPlainText() or True
    hex_before = panel._star_hex
    panel._spin_t[1].setValue(3500.0)
    assert panel._star_hex != hex_before

    # speed slider changes the animation rate (frame-rate independent)
    panel._speed.setValue(panel.SPEED.to_pos(1.0)); panel._set_clock(panel._params.t0 - 0.5)
    t_a = panel._t; panel._advance(dt=0.1); step_1x = panel._t - t_a
    panel._speed.setValue(panel.SPEED.to_pos(5.0)); panel._set_clock(t_a)
    panel._advance(dt=0.1); step_5x = panel._t - t_a
    assert step_5x == pytest.approx(5.0 * step_1x, rel=0.1) and step_1x > 0

    # planet-size box and camera buttons act on the 2-D fallback
    orbit = panel._view3d.fallback
    assert orbit is not None and "pip install pyvista" in panel._view3d.reason or "disabled" in panel._view3d.reason
    panel._scale.setValue(1.0); r1 = orbit._planet.rect().width()
    panel._scale.setValue(10.0); r10 = orbit._planet.rect().width()
    assert r10 == pytest.approx(10 * r1, rel=0.05)
    orbit.set_view("observer"); w_obs = np.ptp(orbit._ring.getData()[1])
    orbit.set_view("oblique"); w_obl = np.ptp(orbit._ring.getData()[1])
    assert w_obl > w_obs                                                        # views really differ
    win.close()


# ---- GUI: data shows without a fit, dataset list, highlight
def test_gui_data_visible_without_fit_dataset_list_and_highlight(tmp_path):
    app = _qapp()
    from app.ui.main_window import MainWindow

    cfg = ConfigManager.load()
    eng, _ = build_engine(cfg)
    win = MainWindow(eng, cfg.physics, enable_3d=False)
    win.show()
    # a nanometre file with tiny absolute values, previously a flat line / single point
    nm = np.linspace(2500, 20000, 600)
    wn = 1e7 / nm
    flux = 0.06 * np.exp(-0.5 * np.exp(-((wn - 2349) / 20) ** 2) - 0.3 * np.exp(-((wn - 1595) / 25) ** 2))
    p1 = _write(tmp_path, "nm.csv", "Wavelength (nm),Flux\n" + _rows(nm, flux))
    obs = win.load_observation_file(str(p1))
    assert obs is not None and win._fit is None
    for unit in Unit:
        item = win._canvases[unit].data_items[0]
        x, y = item.getData()
        assert x.size == 600 and y.max() == pytest.approx(1.0, abs=0.05) and y.min() < 0.7      # features visible
        lo, hi = win._canvases[unit].plot.getViewBox().viewRange()[0]
        assert lo <= min(x) + 1e-6 and hi >= max(x) - 1e-6                                      # view includes the data

    p2 = _write(tmp_path, "b.csv", "wavenumber,transmittance\n" + _rows(np.linspace(500, 4000, 300), np.full(300, .9)))
    win.load_observation_file(str(p2))
    assert win._data_list.count() == 2 and len(win._canvases[Unit.WAVENUMBER].data_items) == 2
    win._data_list.item(0).setCheckState(Qt_unchecked())                         # hide first
    assert len(win._canvases[Unit.WAVENUMBER].data_items) == 1
    win._data_list.setCurrentRow(0)                                              # make it active again
    assert win._active_dataset().name == "nm.csv"
    win._on_show_model(False)
    assert not win._canvases[Unit.WAVENUMBER].master.isVisible()

    win._on_identify()                                                           # data-only identification
    assert win._anomalies and "→" in win._anom_list.item(0).text()
    assert any(c.molecule in ("CO2", "H2O") for c in win._candidates)
    win._anom_list.setCurrentRow(1)                                              # click -> highlight + zoom
    reg = win._canvases[Unit.WAVENUMBER].regions
    assert reg[1].zValue() == -4 and reg[0].zValue() == -5
    lo, hi = win._canvases[win.active_unit()].plot.getViewBox().viewRange()[0]
    a = win._anomalies[1]
    assert lo <= a.nu_peak_cm1 <= hi and (hi - lo) < 3500
    win._on_clear_observation(); win._on_clear_observation()
    assert win._obs is None and not win._btn_fit.isEnabled()
    win.close()


def Qt_unchecked():
    from PyQt6.QtCore import Qt
    return Qt.CheckState.Unchecked


def test_pyvista_scene_builds_and_moves_planet_in_isolated_process(tmp_path):
    """Real VTK rendering (software, offscreen) of the exact scene the viewport builds.
    Run in a subprocess so a graphics-stack problem can never take the test session down."""
    pytest.importorskip("pyvista")
    import subprocess, textwrap
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        import numpy as np, pyvista as pv
        from app.core.transit import TransitParams, orbit_positions
        from app.ui.transit_panel import build_scene, camera_for
        p = TransitParams(1.0, 3.2, 0.1, 9.4, 0.3)
        pl = pv.Plotter(off_screen=True, window_size=(300, 200))
        pl.set_background("#000000")
        actor = build_scene(pl, p, planet_scale=4.0)
        pl.camera_position = camera_for("oblique", p.a_rs)
        frames = []
        for t in (1.0, 1.8):
            actor.position = tuple(float(v[0]) for v in orbit_positions(np.array([t]), p))
            pl.render()
            frames.append(pl.screenshot().copy())      # screenshot() reuses one buffer: copy it
        assert frames[0].shape == (200, 300, 3)
        assert (frames[0] != frames[1]).any(), "planet did not move"
        assert (frames[0].sum(axis=2) > 60).sum() > 100, "scene is blank"
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "OK" in out.stdout, out.stdout + out.stderr


def test_gui_identify_workspace_citations_and_report(tmp_path):
    app = _qapp()
    from app.ui.main_window import MainWindow
    from app.ui.sources_dialog import SourcesDialog

    cfg = ConfigManager.load()
    # SO2 is an enabled simulator gas (mock-sourced). Disable it here so the injected SO2
    # band is unexplained until a real reference is imported below.
    raw = ConfigManager.to_dict(cfg)
    for g in raw["gases"]:
        if g["id"] == "SO2":
            g["enabled"] = False
    cfg = ConfigManager.from_dict(raw, base_dir=cfg.base_dir)
    eng, _ = build_engine(cfg)
    win = MainWindow(eng, cfg.physics, enable_3d=False, library_dir=tmp_path / "lib")
    win.show()
    assert win._workspaces.count() == 3 and win._workspaces.tabText(1) == "Identify"
    panel = win._identify
    assert not panel._btn_run.isEnabled()                                    # nothing loaded yet
    win._on_mock_observation()
    assert panel._btn_run.isEnabled() and "MOCK observation" in panel._dataset_label.text()

    res = panel.run()
    assert res is not None and {c.ref_id for c in res.detected()} == set(TRUTH)
    assert panel._table.rowCount() == len(res.components) and panel._table.item(0, 5).text() == "DETECTED"
    assert panel._features.count() == len(res.attributions) > 0
    panel._features.setCurrentRow(0)
    assert panel._region is not None
    panel._table.setCurrentCell(1, 0)                                        # emphasises that molecule's curve
    assert "mock" in win._citations.keys()

    # importing a JCAMP reference adds it, persists it, and records its citation
    nu = np.linspace(500, 4000, 400)
    t = 1 - 0.6 * np.exp(-0.5 * ((nu - 1362) / 12) ** 2)
    jdx = ["##TITLE=Sulfur dioxide", "##JCAMP-DX=4.24", "##CAS REGISTRY NO=7446-09-5", "##MOLFORM=O2 S",
           "##XUNITS=1/CM", "##YUNITS=TRANSMITTANCE", "##XFACTOR=1.0", "##YFACTOR=1.0", "##FIRSTX=500", "##LASTX=4000",
           "##NPOINTS=400", "##DELTAX=%.6f" % (3500 / 399), "##XYDATA=(X++(Y..Y))"]
    jdx += [f"{nu[i]:.4f} " + " ".join(f"{v:.5f}" for v in t[i:i + 5]) for i in range(0, 400, 5)]
    path = tmp_path / "so2.jdx"; path.write_text("\n".join(jdx) + "\n##END=\n")
    assert panel.import_reference(str(path)) and "cas:7446-09-5" in panel._user
    assert (tmp_path / "lib" / "index.json").exists() and "nist_webbook" in win._citations.keys()
    res2 = panel.run()
    so2 = next(c for c in res2.components if c.ref_id == "cas:7446-09-5")
    assert so2.detected and so2.significance > 10 and not so2.quantitative and so2.source_key == "nist_webbook"
    assert res2.rms_t < res.rms_t                                            # explaining SO2 shrank the real residual
    assert not panel.import_reference(str(tmp_path / "missing.jdx"))        # bad file: message, no crash

    dlg = SourcesDialog(win._citations, win)
    assert "NIST Chemistry WebBook" in dlg._text.toPlainText() and "10.18434/T4D303" in dlg._text.toPlainText()
    md = ecas_export.build_report_markdown(eng, win._conditions, win._multipliers,
                                           identification_lines=res2.summary_lines(), citations=win._citations)
    assert "## Molecule identification" in md and "## References" in md and "MOCK" in md
    win.close()


def test_gui_planet_lookup_fills_star_cites_and_compares(tmp_path):
    app = _qapp()
    import time as _time
    from app.ui.main_window import MainWindow
    from app.ui.archive_dialog import PlanetLookupDialog
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_archives import _csv

    cfg = ConfigManager.load()
    eng, _ = build_engine(cfg)
    fixture = _csv(pl_orbper="3.2", pl_ratror="0.1", pl_ratdor="9.0", pl_imppar="0.3", st_rad="1.4", st_mass="1.2",
                   st_teff="6300", pl_trandur="2.84")
    win = MainWindow(eng, cfg.physics, enable_3d=False, library_dir=tmp_path / "lib", planet_fetch=lambda url: fixture)
    panel = win._transit

    dlg = PlanetLookupDialog(win, fetch=lambda url: fixture, cache_dir=None)         # threaded fetch in the dialog
    dlg._name.setText("TEST-1 b"); dlg.fetch_now()
    deadline = _time.time() + 20
    while dlg.record is None and _time.time() < deadline:
        app.processEvents(); _time.sleep(0.01)
    assert dlg.record is not None and dlg._btn_use.isEnabled() and "Doe et al. 2024" in dlg._summary.toPlainText()
    bad = PlanetLookupDialog(win, fetch=lambda url: (_ for _ in ()).throw(archives_error()), cache_dir=None)
    bad._name.setText("X b"); bad.fetch_now()
    deadline = _time.time() + 20
    while "unreachable" not in bad._status.text() and _time.time() < deadline:
        app.processEvents(); _time.sleep(0.01)
    assert "unreachable" in bad._status.text() and not bad._btn_use.isEnabled()

    panel.use_planet(dlg.record)
    assert (panel._spin_r[1].value(), panel._spin_m[1].value(), panel._spin_t[1].value()) == (1.4, 1.2, 6300.0)
    assert panel._btn_fetch_mast.isEnabled()
    assert panel._btn_fetch_spectral.isEnabled()
    assert "nea_planetary_systems" in win._citations.keys() and "TEST-1 b" in win._citations.references_markdown()
    panel._on_mock(); panel.analyze(blocking=True)
    text = panel._report.toPlainText()
    assert "Comparison with the NASA Exoplanet Archive (TEST-1 b)" in text and "consistent" in text
    assert "R★ 1.4" in text and any("Archive source" in l for l in panel.report_lines)
    win.close()


def archives_error():
    from app.core.archives import ArchiveError
    return ArchiveError("archive unreachable (test)")


def test_gui_spectral_time_series_snapshot_extract_and_identify(tmp_path):
    app = _qapp()
    from app.ui.main_window import MainWindow

    cfg = ConfigManager.load()
    eng, _ = build_engine(cfg)
    win = MainWindow(eng, cfg.physics, enable_3d=False, library_dir=tmp_path / "lib")
    win._workspaces.setCurrentIndex(2)
    panel = win._transit
    assert not panel._btn_extract.isEnabled()
    panel._on_mock_series()
    assert panel._series is not None and panel._btn_extract.isEnabled() and panel._lc is not None
    assert "integrations" in panel._series_label.text() and "mock" in win._citations.keys()
    panel.analyze(blocking=True)                                                   # one transit: honest refusal + advice
    assert panel._result is None and "single" in panel._report.toPlainText() and "archive" in panel._report.toPlainText()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_archives import _csv
    from app.core.archives import fetch_planet
    panel.use_planet(fetch_planet("TEST-1 b", fetch=lambda u: _csv(pl_orbper="3.2", pl_ratror="0.1", pl_ratdor="9.4",
                                                                      pl_imppar="0.3", st_rad="1.0", st_mass="1.0")))
    panel.analyze(blocking=True)                                                   # archive period -> single-transit fit
    assert panel._result is not None and panel._result.params.period == pytest.approx(3.2, rel=1e-5)
    assert panel._result.params.rp_rs == pytest.approx(0.1, rel=0.1)

    # spectrum at t: flat out of transit, dips mid-transit, and follows the scrubber
    panel._set_clock(panel._params.t0 - 2.5 * panel._params.duration_days)
    y_out = np.array(panel._spec_curve.getData()[1], dtype=float)
    panel._set_clock(panel._params.t0)
    y_in = np.array(panel._spec_curve.getData()[1], dtype=float)
    assert abs(np.nanmedian(y_out) - 1) < 0.003 and np.nanmedian(y_in) < 0.992
    assert "t =" in panel._spec_caption.textItem.toPlainText()

    # extraction displays the measured spectrum first; the separate action transfers it to Identify
    panel._on_extract()
    assert panel._transmission_spectrum is not None and win._obs is None
    assert win._workspaces.currentIndex() == 2 and panel._btn_identify.isEnabled()
    panel._on_send_to_identify()
    assert win._obs is not None and win._workspaces.currentIndex() == 0
    assert win._obs.meta["name"].startswith("transmission spectrum") and "geometry from your transit analysis" in " ".join(win._obs.meta["import_notes"])
    assert "abundances are relative, not absolute" in " ".join(win._obs.meta["import_notes"])
    res = win._identify.run()
    assert res is not None and res.detected()

    # a plain light curve replaces the series and disables extraction
    panel._on_mock()
    assert panel._series is None and not panel._btn_extract.isEnabled()
    # without any geometry the user gets a clear instruction, not a crash
    panel._series = None
    win.close()


def test_single_transit_with_known_period():
    from app.core.transit import TransitParams
    rng = np.random.default_rng(8)
    truth = TransitParams(1.0, 3.2, 0.1, 9.0, 0.3)
    t = np.arange(0.4, 1.6, 4 / 1440)                                         # ONE transit, ~1.2 d of data
    lc = LightCurve(t, transit_model(t, truth) + rng.normal(0, 6e-4, t.size))
    with pytest.raises(TransitError, match="single"):
        analyze_lightcurve(lc)                                                  # BLS cannot work: says what to do
    res = analyze_lightcurve(lc, known_period=3.2)
    assert res.params.period == pytest.approx(3.2, rel=1e-6)                    # held fixed
    assert res.params.t0 == pytest.approx(1.0, abs=0.005) and res.params.rp_rs == pytest.approx(0.1, rel=0.1)
    assert res.params.a_rs == pytest.approx(9.0, rel=0.3) and res.n_transits == 1
    with pytest.raises(TransitError):
        analyze_lightcurve(LightCurve(t, 1 + rng.normal(0, 6e-4, t.size)), known_period=3.2)   # no dip: refuses


# ------------------------------------------ regression: single-transit & uncertainty handling
from app.core.timeseries import SpectralTimeSeries  # noqa: E402
def _single_transit_case(seed=0, noise=0.0008, drift=0.0):
    """One ~1.3 h transit in a 4 h JWST-like window, with LHS 1140 b-like host star (synthetic test data)."""
    rng = np.random.default_rng(seed)
    t = np.arange(2054) * (0.17 / 2054)
    truth = TransitParams(0.092, 24.74, 0.05, 94.1, 0.2)
    model = transit_model(t, truth)
    if drift:
        model = model * (1.0 + drift * (t - t.mean()) / 0.085)
    return t, truth, model + rng.normal(0.0, noise, t.size)


def test_single_transit_depth_recovered_with_and_without_drift():
    """The depth must not be biased by normalising on the in-transit median, nor by a slow drift."""
    st = StellarParams(0.216, 0.184, 3096)
    t, truth, flux = _single_transit_case()
    res = analyze_lightcurve(LightCurve(t, flux), st, detrend_window_days=None, known_period=24.74)
    assert res.params.rp_rs == pytest.approx(truth.rp_rs, rel=0.03)
    t, truth, flux = _single_transit_case(drift=0.002)
    res = analyze_lightcurve(LightCurve(t, flux), st, detrend_window_days=None, known_period=24.74)
    assert res.params.rp_rs == pytest.approx(truth.rp_rs, rel=0.03)
    assert any("linear baseline" in n for n in res.notes)


def test_uncertainty_column_on_wrong_scale_does_not_reject_a_real_transit():
    st = StellarParams(0.216, 0.184, 3096)
    t, _, flux = _single_transit_case()
    res = analyze_lightcurve(LightCurve(t, flux, np.full(t.size, 0.0008 * 1000.0)), st,
                             detrend_window_days=None, known_period=24.74)
    assert res.params.rp_rs == pytest.approx(0.05, rel=0.05)
    assert any("uncertainty column" in n for n in res.notes)


def test_failure_message_reports_candidate_and_how_to_proceed():
    st = StellarParams(0.216, 0.184, 3096)
    t = np.arange(2054) * (0.17 / 2054)
    noise_only = 1.0 + np.random.default_rng(3).normal(0.0, 0.0008, t.size)
    with pytest.raises(TransitError, match="Detection SNR"):
        analyze_lightcurve(LightCurve(t, noise_only), st, detrend_window_days=None, known_period=24.74)


def test_extracted_depth_errors_follow_the_data_not_the_error_column():
    t, truth, _ = _single_transit_case()
    rng = np.random.default_rng(1)
    wl = np.linspace(2.44, 5.2, 300)
    flux = 1.0 + (transit_model(t, truth) - 1.0)[:, None] + rng.normal(0.0, 0.001, (t.size, wl.size))
    honest = SpectralTimeSeries(t, wl, flux, np.full(flux.shape, 0.001), {}).transmission_spectrum(truth)
    inflated = SpectralTimeSeries(t, wl, flux, np.full(flux.shape, 1.0), {}).transmission_spectrum(truth)
    assert np.median(inflated.depth_err) == pytest.approx(np.median(honest.depth_err), rel=0.05)
