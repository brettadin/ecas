"""Tests for optional atmospheric forward-model inputs and pRT adapter behavior."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import atmosphere
from app.core.atmosphere import (
    AtmosphereModelError,
    PetitRadtransParameters,
    TransmissionAtmosphereModel,
    compare_model_to_depth_spectrum,
    compute_petit_radtrans_transmission,
    depth_spectrum_channel_edges,
    scan_single_gas_abundance,
    vmr_to_mass_fractions,
)
from app.core.retrieval_validation import (
    make_synthetic_depth_spectrum, run_profile_validation, summarize_profile_scan,
)
from app.core.transit import DepthSpectrum


def test_vmr_conversion_adds_h2_he_background_and_normalizes_mass():
    mass_fractions, mean_molar_mass = vmr_to_mass_fractions({"H2O": 1e-3, "CO2": 2e-4})
    assert set(mass_fractions) == {"H2O", "CO2", "H2", "He"}
    assert sum(mass_fractions.values()) == pytest.approx(1.0)
    assert mass_fractions["H2"] > mass_fractions["He"] > 0.0
    assert mean_molar_mass == pytest.approx(sum(
        fraction * {"H2O": 18.01528, "CO2": 44.0095, "H2": 2.01588, "He": 4.002602}[species]
        for species, fraction in {
            "H2O": 1e-3, "CO2": 2e-4,
            "H2": (1.0 - 1.2e-3) / 1.17,
            "He": (1.0 - 1.2e-3) * 0.17 / 1.17,
        }.items()
    ))


def test_petit_radtrans_adapter_returns_physical_transit_depth():
    calls = {}

    class FakeRadtrans:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def calculate_transit_radii(self, **kwargs):
            calls["calculate"] = kwargs
            wavelength_cm = np.linspace(1e-4, 2e-4, 12)
            radius_cm = np.full(12, 1.1 * 7.1492e9)
            return wavelength_cm, radius_cm, {}

    parameters = PetitRadtransParameters(
        temperature_k=1100.0, gravity_cgs=800.0, planet_radius_rjup=1.1,
        stellar_radius_rsun=0.9, mixing_ratios={"H2O": 1e-3, "CO2": 1e-4},
    )
    result = compute_petit_radtrans_transmission(parameters, (1.0, 2.0), FakeRadtrans)

    assert calls["init"]["line_species"] == ["H2O", "CO2"]
    np.testing.assert_allclose(calls["init"]["wavelength_boundaries"], [0.995, 2.01])
    assert calls["calculate"]["reference_gravity"] == 800.0
    assert calls["calculate"]["temperatures"].shape == (100,)
    np.testing.assert_allclose(result.wavelength_um, np.linspace(1.0, 2.0, 12))
    np.testing.assert_allclose(result.depth_ppm, (1.1 * 7.1492e9 / (0.9 * 6.957e10))**2 * 1e6)
    assert result.meta["volume_mixing_ratios"] == {"H2O": 1e-3, "CO2": 1e-4}
    assert any("no instrumental" in assumption for assumption in result.meta["assumptions"])


@pytest.mark.parametrize("ratios", [{"UnknownGas": 1e-4}, {"H2O": 1.0}, {"H2O": 0.0}])
def test_invalid_atmosphere_mixing_ratios_are_rejected(ratios):
    with pytest.raises(AtmosphereModelError):
        PetitRadtransParameters(mixing_ratios=ratios)


def test_petit_radtrans_rejects_invalid_wavelength_range():
    with pytest.raises(AtmosphereModelError, match="wavelength range"):
        compute_petit_radtrans_transmission(PetitRadtransParameters(), (2.0, 1.0), lambda **_: None)


def test_petit_radtrans_padding_keeps_inferred_channel_edges_in_model_coverage():
    class BinCenteredRadtrans:
        def __init__(self, **kwargs):
            self.wavelength_boundaries = kwargs["wavelength_boundaries"]

        def calculate_transit_radii(self, **kwargs):
            lo, hi = self.wavelength_boundaries
            wavelength_cm = np.linspace(lo * 1.0005, hi * 0.9995, 12) * 1e-4
            radius_cm = np.full(wavelength_cm.shape, 1.0 * 7.1492e9)
            return wavelength_cm, radius_cm, {}

    parameters = PetitRadtransParameters(mixing_ratios={"H2O": 1e-3})
    model = compute_petit_radtrans_transmission(parameters, (1.0, 2.0), BinCenteredRadtrans)
    centers = np.linspace(1.1, 1.9, 5)
    measured = DepthSpectrum(centers, np.full(centers.size, 0.01), np.full(centers.size, 1e-4))

    comparison = compare_model_to_depth_spectrum(model, measured)

    assert comparison.model_depth_ppm.shape == centers.shape


def test_windows_fortran_dll_search_registers_runtime_once(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "ucrt64" / "bin"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "libgfortran-5.dll").touch()
    registered = []

    monkeypatch.setattr(atmosphere.sys, "platform", "win32")
    monkeypatch.setattr(atmosphere.shutil, "which", lambda _: None)
    monkeypatch.setattr(atmosphere.os, "add_dll_directory",
                        lambda path: registered.append(path) or object(), raising=False)
    monkeypatch.setenv("MSYS2_ROOT", str(tmp_path))
    monkeypatch.delenv("MSYS2_HOME", raising=False)
    monkeypatch.setenv("PATH", "")

    atmosphere._prepare_windows_fortran_dll_search()
    atmosphere._prepare_windows_fortran_dll_search()

    assert registered.count(str(runtime_dir)) == 1


def test_model_comparison_integrates_linear_model_over_irregular_channels():
    measured_wavelength = np.array([1.0, 1.4, 2.1, 3.0])
    model_wavelength = np.linspace(0.5, 3.5, 61)
    model_depth_ppm = 1000.0 + 500.0 * model_wavelength
    model = TransmissionAtmosphereModel(
        model_wavelength, model_depth_ppm, np.full(model_wavelength.shape, 0.1))
    edges = depth_spectrum_channel_edges(measured_wavelength)
    expected_model = 1000.0 + 500.0 * (edges[:-1] + edges[1:]) / 2.0
    measured_depth_ppm = expected_model + np.array([0.0, 2.0, -3.0, 4.0])
    measured = DepthSpectrum(measured_wavelength, measured_depth_ppm * 1e-6, np.full(4, 2e-6))

    comparison = compare_model_to_depth_spectrum(model, measured)

    np.testing.assert_allclose(comparison.model_depth_ppm, expected_model)
    np.testing.assert_allclose(comparison.residual_ppm, [0.0, 2.0, -3.0, 4.0], atol=1e-10)
    assert comparison.chi_squared == pytest.approx((0.0**2 + 1.0**2 + 1.5**2 + 2.0**2))
    assert comparison.chi_squared_per_point == pytest.approx(comparison.chi_squared / 4)
    assert depth_spectrum_channel_edges(measured_wavelength).shape == (5,)


def test_model_comparison_omits_invalid_uncertainties_and_rejects_incomplete_coverage():
    wavelength = np.array([1.0, 1.5, 2.0])
    model_wavelength = np.linspace(0.5, 2.5, 16)
    model = TransmissionAtmosphereModel(
        model_wavelength, np.full(16, 1000.0), np.full(16, 0.1))
    measured = DepthSpectrum(wavelength, np.array([0.001, 0.002, 0.001]),
                             np.array([0.0001, 0.0, 0.0001]))

    comparison = compare_model_to_depth_spectrum(model, measured)

    assert np.isfinite(comparison.residual_sigma[[0, 2]]).all()
    assert np.isnan(comparison.residual_sigma[1])
    with pytest.raises(AtmosphereModelError, match="coverage"):
        compare_model_to_depth_spectrum(
            TransmissionAtmosphereModel(np.linspace(0.8, 2.2, 16), np.full(16, 1000.0),
                                        np.full(16, 0.1)), measured)


@pytest.mark.parametrize("wavelength", [
    np.array([1.0]),
    np.array([1.0, 1.0]),
    np.array([1.5, 1.0]),
    np.array([np.nan, 1.0]),
])
def test_model_channel_edges_reject_invalid_wavelength_centers(wavelength):
    with pytest.raises(AtmosphereModelError):
        depth_spectrum_channel_edges(wavelength)


def test_single_gas_scan_recovers_injected_vmr_and_fits_depth_offset():
    known_vmr = 1e-4

    class FeatureRadtrans:
        def __init__(self, **kwargs):
            pass

        def calculate_transit_radii(self, *, mass_fractions, mean_molar_masses, **kwargs):
            wavelength_um = np.linspace(0.8, 2.2, 101)
            h2o_vmr = (mass_fractions["H2O"][0] * mean_molar_masses[0] / 18.01528)
            feature = np.sin(np.linspace(0.0, 4.0 * np.pi, wavelength_um.size))
            radius = 7.1492e9 + 1.2e7 * np.log10(h2o_vmr / known_vmr) * feature
            radius += 1e6
            return wavelength_um * 1e-4, radius, {}

    parameters = PetitRadtransParameters(
        planet_radius_rjup=1.0, stellar_radius_rsun=1.0, mixing_ratios={"H2O": 1e-4})
    centers = np.linspace(1.0, 2.0, 21)
    model = compute_petit_radtrans_transmission(
        parameters, (float(depth_spectrum_channel_edges(centers)[0]),
                     float(depth_spectrum_channel_edges(centers)[-1])), FeatureRadtrans)
    measured_depth = np.interp(centers, model.wavelength_um, model.depth_ppm) + 37.0
    measured = DepthSpectrum(centers, measured_depth * 1e-6, np.full(centers.size, 5e-6))
    scan_grid = np.logspace(-6, -2, 17)

    result = scan_single_gas_abundance(parameters, measured, "H2O", scan_grid, FeatureRadtrans)

    assert result.best_volume_mixing_ratio == pytest.approx(known_vmr)
    assert result.best_depth_offset_ppm == pytest.approx(37.0, abs=1.0)
    assert result.best_chi_squared < 1.0
    assert result.lower_1sigma_vmr is not None and result.upper_1sigma_vmr is not None
    assert result.lower_1sigma_vmr < known_vmr < result.upper_1sigma_vmr
    assert result.delta_chi_squared.min() == pytest.approx(0.0)
    assert result.n_channels == centers.size
    assert result.wavelength_um.shape == result.best_model_depth_ppm.shape == centers.shape


def test_single_gas_scan_rejects_unconfigured_species_and_invalid_grid():
    parameters = PetitRadtransParameters(mixing_ratios={"H2O": 1e-3})
    measured = DepthSpectrum(np.linspace(1.0, 2.0, 8), np.full(8, 0.01), np.full(8, 1e-4))
    with pytest.raises(AtmosphereModelError, match="not included"):
        scan_single_gas_abundance(parameters, measured, "CO2", np.logspace(-6, -2, 5), lambda **_: None)
    with pytest.raises(AtmosphereModelError, match="at least three"):
        scan_single_gas_abundance(parameters, measured, "H2O", np.array([1e-5, 1e-4]), lambda **_: None)


def test_single_gas_scan_reports_unbounded_profile_and_boundary_best_fit():
    class FlatRadtrans:
        def __init__(self, **kwargs):
            pass

        def calculate_transit_radii(self, **kwargs):
            wavelength_um = np.linspace(0.8, 2.2, 101)
            radius = np.full(wavelength_um.size, 7.1492e9)
            return wavelength_um * 1e-4, radius, {}

    parameters = PetitRadtransParameters(mixing_ratios={"H2O": 1e-4})
    centers = np.linspace(1.0, 2.0, 9)
    measured = DepthSpectrum(centers, np.full(centers.size, 0.001), np.full(centers.size, 1e-6))
    result = scan_single_gas_abundance(
        parameters, measured, "H2O", np.logspace(-6, -3, 5), FlatRadtrans)

    assert result.best_volume_mixing_ratio == pytest.approx(1e-6)
    assert result.lower_1sigma_vmr is None
    assert result.upper_1sigma_vmr is None
    assert result.boundary_status == "lower_scan_edge"
    assert result.conditional_limit_kind is None
    assert result.conditional_limit_vmr is None
    assert any("boundary" in warning for warning in result.warnings)
    assert sum("bound" in warning for warning in result.warnings) == 3


@pytest.mark.parametrize(
    ("known_vmr", "expected_kind", "expected_limit"),
    [(1e-6, "upper", "upper_1sigma_vmr"), (1e-2, "lower", "lower_1sigma_vmr")],
)
def test_single_gas_scan_reports_one_sided_limit_at_scan_edges(
    known_vmr, expected_kind, expected_limit,
):
    class MonotonicRadtrans:
        def __init__(self, **kwargs):
            pass

        def calculate_transit_radii(self, *, mass_fractions, mean_molar_masses, **kwargs):
            wavelength_um = np.linspace(0.8, 2.2, 101)
            h2o_vmr = mass_fractions["H2O"][0] * mean_molar_masses[0] / 18.01528
            feature = np.sin(np.linspace(0.0, 4.0 * np.pi, wavelength_um.size))
            radius = 7.1492e9 + 1.2e7 * np.log10(h2o_vmr / known_vmr) * feature
            return wavelength_um * 1e-4, radius, {}

    parameters = PetitRadtransParameters(mixing_ratios={"H2O": known_vmr})
    centers = np.linspace(1.0, 2.0, 21)
    model = compute_petit_radtrans_transmission(
        parameters, (float(depth_spectrum_channel_edges(centers)[0]),
                     float(depth_spectrum_channel_edges(centers)[-1])), MonotonicRadtrans)
    measured_template = DepthSpectrum(centers, np.zeros(centers.size), np.full(centers.size, 5e-6))
    noiseless = compare_model_to_depth_spectrum(model, measured_template)
    measured = DepthSpectrum(centers, noiseless.model_depth_ppm * 1e-6, np.full(centers.size, 5e-6))
    result = scan_single_gas_abundance(
        parameters, measured, "H2O", np.logspace(-6, -2, 17), MonotonicRadtrans)

    assert result.boundary_status in {"lower_scan_edge", "upper_scan_edge"}
    assert result.conditional_limit_kind == expected_kind
    assert result.conditional_limit_vmr == getattr(result, expected_limit)
    assert result.conditional_limit_vmr is not None


def test_synthetic_validation_helpers_measure_bias_and_interval_coverage():
    model = TransmissionAtmosphereModel(
        np.linspace(0.9, 2.1, 121),
        1000.0 + 100.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, 121)),
        np.full(121, 1.0),
    )
    centers = np.linspace(1.05, 1.95, 9)
    first = make_synthetic_depth_spectrum(model, centers, 2.0, depth_offset_ppm=12.0, seed=11)
    second = make_synthetic_depth_spectrum(model, centers, np.full(centers.size, 2.0), seed=12)
    assert first.depth.shape == second.depth.shape == centers.shape
    assert np.array_equal(first.depth_err, second.depth_err)
    assert not np.array_equal(first.depth, second.depth)

    class Result:
        pass

    result = Result()
    result.best_volume_mixing_ratio = 1.1e-4
    result.lower_1sigma_vmr = 8e-5
    result.upper_1sigma_vmr = 1.3e-4
    result.best_chi_squared = 1.2
    result.best_depth_offset_ppm = 12.0
    result.warnings = ()
    summary = summarize_profile_scan([(result, 1e-4, 11)])
    assert summary.interval_coverage_fraction == pytest.approx(1.0)
    assert summary.bounded_interval_fraction == pytest.approx(1.0)
    assert summary.boundary_limited_fraction == pytest.approx(0.0)
    assert summary.median_absolute_log10_bias == pytest.approx(np.log10(1.1))


def test_synthetic_validation_can_apply_line_spread_and_reject_invalid_sigma():
    model = TransmissionAtmosphereModel(
        np.linspace(0.9, 2.1, 121),
        1000.0 + 100.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, 121)),
        np.full(121, 1.0),
    )
    centers = np.linspace(1.05, 1.95, 9)
    sharp = make_synthetic_depth_spectrum(model, centers, 2.0, seed=5)
    blurred = make_synthetic_depth_spectrum(model, centers, 2.0, seed=5,
                                            line_spread_sigma_um=0.03)
    assert not np.array_equal(sharp.depth, blurred.depth)
    with pytest.raises(ValueError, match="line-spread sigma"):
        make_synthetic_depth_spectrum(model, centers, 2.0, line_spread_sigma_um=0.0)
    with pytest.raises(ValueError, match="correlation length"):
        make_synthetic_depth_spectrum(model, centers, 2.0, correlation_length_channels=0.0)


def test_profile_validation_runner_repeats_each_injection_and_seed():
    model = TransmissionAtmosphereModel(
        np.linspace(1.0, 2.0, 21), np.full(21, 1000.0), np.full(21, 1.0))
    calls = []

    def model_for_vmr(vmr):
        calls.append(("model", vmr))
        return model

    def scan_observation(observation):
        calls.append(("scan", int(round(float(observation.depth[0] * 1e6)))))
        result = type("Result", (), {})()
        result.best_volume_mixing_ratio = 1e-4
        result.lower_1sigma_vmr = 5e-5
        result.upper_1sigma_vmr = 2e-4
        result.best_chi_squared = 1.0
        result.best_depth_offset_ppm = 7.0
        result.warnings = ()
        return result

    summary = run_profile_validation(
        model_for_vmr, scan_observation, np.linspace(1.1, 1.9, 5), 2.0,
        [1e-4, 2e-4], [3, 4], depth_offset_ppm=7.0)

    assert len(summary.records) == 4
    assert len([call for call in calls if call[0] == "model"]) == 2
    assert len([call for call in calls if call[0] == "scan"]) == 4
    assert summary.bounded_interval_fraction == pytest.approx(1.0)
