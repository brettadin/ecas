"""
ECAS core physics: line broadening and line-shape kernels.

Everything here is pure NumPy/SciPy and fully vectorized.  There is no Qt import
and no Python-level looping: every function accepts broadcastable arrays, so the
caller can build an ``(N_grid, M_lines)`` profile matrix in a single call by
passing ``x`` with shape ``(N, 1)`` and line parameters with shape ``(1, M)``.

Conventions
-----------
* Spectral positions and widths are in wavenumbers (cm^-1).
* All widths are HWHM (half width at half maximum).
* Every profile integrates to 1 over the wavenumber axis, so multiplying by a
  line intensity (km/mol) yields an integrated band strength.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import wofz

# --- physical constants (SI) -------------------------------------------------
K_B = 1.380649e-23          # Boltzmann constant, J/K
C_LIGHT = 2.99792458e8      # speed of light, m/s
AMU = 1.66053906660e-27     # atomic mass unit, kg

# --- HITRAN reference state --------------------------------------------------
P_REF_ATM = 1.0             # reference pressure, atm
T_REF_K = 296.0             # reference temperature, K

_LN2 = float(np.log(2.0))
_TINY = 1.0e-12             # width floor: avoids 0/0 in the kernels

FloatArray = NDArray[np.float64]


class LineShape(str, Enum):
    """Selectable line-shape model."""

    LORENTZIAN = "lorentzian"
    GAUSSIAN = "gaussian"
    VOIGT = "voigt"


# --- unit conversion ---------------------------------------------------------
def wavenumber_to_wavelength_um(nu_cm1: ArrayLike) -> FloatArray:
    """Convert wavenumber (cm^-1) to wavelength (micrometres): lambda = 1e4 / nu."""
    return 1.0e4 / np.asarray(nu_cm1, dtype=np.float64)


def wavelength_um_to_wavenumber(lam_um: ArrayLike) -> FloatArray:
    """Convert wavelength (micrometres) to wavenumber (cm^-1): nu = 1e4 / lambda."""
    return 1.0e4 / np.asarray(lam_um, dtype=np.float64)


# --- broadening widths -------------------------------------------------------
def lorentz_hwhm(
    pressure_atm: ArrayLike,
    temperature_k: ArrayLike,
    gamma0_cm1_atm: ArrayLike,
    n_exponent: ArrayLike,
    p_ref_atm: float = P_REF_ATM,
    t_ref_k: float = T_REF_K,
) -> FloatArray:
    """
    Collisional (pressure) broadening, HITRAN convention.

        gamma_L(P, T) = gamma_0 * (P / P_0) * (T_0 / T) ** n

    ``gamma0_cm1_atm`` and ``n_exponent`` may be per-line arrays; the result
    broadcasts against them.
    """
    return (
        np.asarray(gamma0_cm1_atm, dtype=np.float64)
        * (np.asarray(pressure_atm, dtype=np.float64) / p_ref_atm)
        * (t_ref_k / np.asarray(temperature_k, dtype=np.float64))
        ** np.asarray(n_exponent, dtype=np.float64)
    )


def doppler_hwhm(
    nu0_cm1: ArrayLike, temperature_k: ArrayLike, mass_amu: ArrayLike
) -> FloatArray:
    """
    Thermal Doppler broadening (Gaussian HWHM).

        alpha_D(T) = nu_0 * sqrt(2 k_B T ln2 / (m c^2))
    """
    mass_kg = np.asarray(mass_amu, dtype=np.float64) * AMU
    return np.asarray(nu0_cm1, dtype=np.float64) * np.sqrt(
        2.0 * K_B * np.asarray(temperature_k, dtype=np.float64) * _LN2
        / (mass_kg * C_LIGHT**2)
    )


# --- line-shape kernels ------------------------------------------------------
def lorentzian(x: ArrayLike, x0: ArrayLike, gamma: ArrayLike) -> FloatArray:
    """Unit-area Lorentzian with HWHM ``gamma``: (1/pi) * gamma / ((x-x0)^2 + gamma^2)."""
    d = np.asarray(x, dtype=np.float64) - x0
    g = np.maximum(np.asarray(gamma, dtype=np.float64), _TINY)
    return g / (np.pi * (d * d + g * g))


def gaussian(x: ArrayLike, x0: ArrayLike, alpha: ArrayLike) -> FloatArray:
    """Unit-area Gaussian with HWHM ``alpha``."""
    d = np.asarray(x, dtype=np.float64) - x0
    a = np.maximum(np.asarray(alpha, dtype=np.float64), _TINY)
    return np.sqrt(_LN2 / np.pi) / a * np.exp(-_LN2 * (d / a) ** 2)


def voigt(
    x: ArrayLike, x0: ArrayLike, gamma: ArrayLike, alpha: ArrayLike
) -> FloatArray:
    """
    Unit-area Voigt profile (Lorentzian convolved with Gaussian), computed exactly
    through the Faddeeva function:  V = Re[w(z)] / (sigma * sqrt(2 pi)),
    with z = ((x - x0) + i*gamma) / (sigma * sqrt(2)) and sigma = alpha / sqrt(2 ln2).

    Parameters
    ----------
    gamma : Lorentzian HWHM (pressure broadening).
    alpha : Gaussian HWHM (Doppler broadening).
    """
    d = np.asarray(x, dtype=np.float64) - x0
    g = np.maximum(np.asarray(gamma, dtype=np.float64), _TINY)
    a = np.maximum(np.asarray(alpha, dtype=np.float64), _TINY)
    sigma = a / np.sqrt(2.0 * _LN2)
    z = (d + 1j * g) / (sigma * np.sqrt(2.0))
    return wofz(z).real / (sigma * np.sqrt(2.0 * np.pi))


def line_profile_matrix(
    x: ArrayLike,
    centers: ArrayLike,
    gamma_l: ArrayLike,
    alpha_d: ArrayLike,
    shape: LineShape = LineShape.VOIGT,
) -> FloatArray:
    """
    Evaluate every line on every grid point with pure broadcasting.

    Pass ``x`` as ``(N, 1)`` and the per-line parameters as ``(1, M)`` to obtain an
    ``(N, M)`` matrix of unit-area profiles.  No loops are involved.  The Gaussian option
    uses ``hypot(alpha_d, gamma_l)`` as its HWHM so that it responds to pressure/envelope.
    """
    shape = LineShape(shape)
    if shape is LineShape.LORENTZIAN:
        return lorentzian(x, centers, gamma_l)
    if shape is LineShape.GAUSSIAN:
        # A pure-Doppler Gaussian is ~0.002 cm^-1 wide - invisible on any display grid and it
        # would silently ignore pressure and band-envelope broadening.  Use the quadrature sum
        # of the Doppler and total Lorentzian widths as the Gaussian HWHM instead.
        return gaussian(x, centers, np.hypot(alpha_d, gamma_l))
    return voigt(x, centers, gamma_l, alpha_d)


def beer_lambert(optical_depth: ArrayLike) -> FloatArray:
    """Transmittance = exp(-tau).  Saturates smoothly towards 0.0 as tau grows."""
    return np.exp(-np.asarray(optical_depth, dtype=np.float64))
