"""Exploratory continuum fitting for normalized planetary emission spectra."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

FloatArray = NDArray[np.float64]
_SECOND_RADIATION_CONSTANT_UM_K = 14387.768775


class EmissionFitError(ValueError):
    """Raised when an emission-continuum fit cannot be evaluated."""


@dataclass
class BlackbodyContinuumFit:
    wavelength_um: FloatArray
    temperature_k: float
    scale: float
    model: FloatArray
    residual: FloatArray
    rms_residual: float
    density_kind: Literal["flam", "fnu"]


def _planck_shape(wavelength_um: FloatArray, temperature_k: float, density_kind: str) -> FloatArray:
    exponent = _SECOND_RADIATION_CONSTANT_UM_K / (wavelength_um * temperature_k)
    log_denominator = exponent + np.log1p(-np.exp(-exponent))
    power = 5.0 if density_kind == "flam" else 3.0
    log_shape = -power * np.log(wavelength_um) - log_denominator
    shape = np.exp(log_shape - np.max(log_shape))
    return shape


def fit_blackbody_continuum(
    wavelength_um: FloatArray,
    normalized_flux: FloatArray,
    uncertainty: Optional[FloatArray] = None,
    density_kind: Literal["flam", "fnu"] = "flam",
) -> BlackbodyContinuumFit:
    """Fit Planck color temperature and arbitrary scale to a normalized Fλ or Fν spectrum.

    This is a continuum diagnostic. It does not retrieve atmospheric composition, physical radius,
    or luminosity; normalization removes the absolute flux calibration.
    """
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    flux = np.asarray(normalized_flux, dtype=np.float64)
    if density_kind not in ("flam", "fnu"):
        raise EmissionFitError("flux-density convention must be 'flam' or 'fnu'")
    if wavelength.ndim != 1 or flux.shape != wavelength.shape:
        raise EmissionFitError("wavelength and flux must be matching 1-D arrays")
    if wavelength.size < 8 or not np.isfinite(wavelength).all() or np.any(wavelength <= 0.0):
        raise EmissionFitError("blackbody fit needs at least 8 finite positive wavelengths")
    if not np.isfinite(flux).all() or np.any(flux <= 0.0):
        raise EmissionFitError("blackbody fit needs finite positive flux values")
    if np.any(np.diff(wavelength) <= 0.0):
        raise EmissionFitError("blackbody wavelengths must be strictly increasing")
    if uncertainty is None:
        weights = np.ones_like(flux)
    else:
        sigma = np.asarray(uncertainty, dtype=np.float64)
        if sigma.shape != flux.shape:
            raise EmissionFitError("flux uncertainty must match the spectrum")
        valid = np.isfinite(sigma) & (sigma > 0.0)
        if valid.sum() < 8:
            raise EmissionFitError("blackbody fit needs at least 8 finite positive uncertainties")
        wavelength, flux, sigma = wavelength[valid], flux[valid], sigma[valid]
        weights = 1.0 / sigma**2

    def evaluate(log_temperature: float) -> tuple[float, float, FloatArray]:
        temperature = float(np.exp(log_temperature))
        shape = _planck_shape(wavelength, temperature, density_kind)
        scale = float(np.sum(weights * flux * shape) / np.sum(weights * shape**2))
        residual = flux - scale * shape
        return float(np.sum(weights * residual**2)), scale, scale * shape

    solution = minimize_scalar(
        lambda log_temperature: evaluate(log_temperature)[0],
        bounds=(np.log(100.0), np.log(10000.0)), method="bounded",
        options={"xatol": 1e-10},
    )
    if not solution.success or not np.isfinite(solution.fun):
        raise EmissionFitError(f"blackbody temperature fit failed: {solution.message}")
    temperature = float(np.exp(solution.x))
    _, scale, model = evaluate(solution.x)
    residual = flux - model
    rms = float(np.sqrt(np.mean(residual**2)))
    return BlackbodyContinuumFit(wavelength.copy(), temperature, scale, model, residual, rms, density_kind)
