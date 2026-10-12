"""Tests for normalized planetary-emission continuum analysis."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.emission import EmissionFitError, fit_blackbody_continuum
from app.core.fitting import build_observation
from app.core.tables import read_table, suggest_spectrum_columns


def _planck_shape(wavelength_um, temperature_k, density_kind):
    exponent = 14387.768775 / (wavelength_um * temperature_k)
    denominator = np.expm1(exponent)
    power = 5 if density_kind == "flam" else 3
    shape = wavelength_um ** -power / denominator
    return shape / np.max(shape)


@pytest.mark.parametrize("density_kind", ["flam", "fnu"])
def test_blackbody_fit_recovers_color_temperature_and_arbitrary_scale(density_kind):
    wavelength = np.linspace(0.9, 5.3, 300)
    expected_temperature = 1450.0
    flux = 2.7 * _planck_shape(wavelength, expected_temperature, density_kind)
    fit = fit_blackbody_continuum(wavelength, flux, density_kind=density_kind)

    assert fit.temperature_k == pytest.approx(expected_temperature, rel=1e-4)
    assert fit.scale > 0.0
    assert fit.rms_residual < 1e-5
    assert fit.density_kind == density_kind
    assert fit.wavelength_um.shape == fit.model.shape == fit.residual.shape


def test_blackbody_fit_rejects_invalid_spectrum_inputs():
    wavelength = np.linspace(1.0, 2.0, 10)
    with pytest.raises(EmissionFitError, match="positive flux"):
        fit_blackbody_continuum(wavelength, np.zeros(10))
    with pytest.raises(EmissionFitError, match="uncertainties"):
        fit_blackbody_continuum(wavelength, np.ones(10), np.zeros(10))


def test_archive_flux_import_is_identified_as_emission_not_transmission(tmp_path):
    wavelength = np.linspace(0.9, 5.3, 30)
    flux = 2e-17 * _planck_shape(wavelength, 1450.0, "flam")
    path = tmp_path / "archive.csv"
    rows = ["CENTRALWAVELNG,FLAM,FLAMERR1,FLAM_AUTHORS"]
    rows.extend(f"{wl:.8f},{value:.12e},,Calculated" for wl, value in zip(wavelength, flux))
    path.write_text("\n".join(rows), encoding="utf-8")
    table = read_table(path)
    columns = suggest_spectrum_columns(table)
    observation = build_observation(table, columns["x"], columns["y"], columns["sigma"])

    assert observation.meta["spectrum_kind"] == "emission_flux"
    assert observation.meta["flux_density_kind"] == "flam"
    assert "not a transmission spectrum" in " ".join(observation.meta["import_notes"])
    assert observation.x.size == wavelength.size
