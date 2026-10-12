"""
ECAS spectrum engine - the Qt-free heart of the dashboard.

Design (why slider drags are cheap)
-----------------------------------
All lines of all gases are flattened into one set of ``(M,)`` arrays.  The optical
depth then factorises as::

    tau_total(x) = tau_scale * sum_g  m_g * sum_{i in g} I_i * phi_i(x; P, T)
                 = ( tau_scale * Phi(x; P, T) @ S ) @ m           # (N,M)@(M,G)@(G,)

* ``Phi``  (N, M) - unit-area profiles; depends ONLY on P, T, envelope, shape.
* ``S``    (M, G) - intensity matrix; ``S[i, g] = I_i`` if line i belongs to gas g.
* ``m``    (G,)   - per-gas multipliers (the sliders).

So ``Phi @ S`` -> ``tau_gas`` (N, G) is cached per spectral unit and only rebuilt
when the *atmospheric conditions* change.  Dragging a gas slider is a single
(N, G) @ (G,) mat-vec plus ``exp`` - no Python loops, no profile re-evaluation.

The grid is a fixed tracking vector ``t`` in [0, 1] mapped onto either the
wavenumber range (400 -> 4200 cm^-1) or the wavelength range (0.1 -> 25 um).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import EcasConfig, GasSpec, GridSpec
from .parser import IRPeakTable, OrcaIRParser, load_peaks
from .physics import (
    LineShape,
    doppler_hwhm,
    line_profile_matrix,
    lorentz_hwhm,
    wavelength_um_to_wavenumber,
)

FloatArray = NDArray[np.float64]


class Unit(str, Enum):
    """Spectral axis shown on a dashboard tab."""

    WAVENUMBER = "wavenumber"
    WAVELENGTH = "wavelength"

    @property
    def axis_label(self) -> str:
        return "Wavenumber (cm⁻¹)" if self is Unit.WAVENUMBER else "Wavelength (µm)"


@dataclass(frozen=True)
class AtmosphericConditions:
    """Hashable physical state; doubles as the cache key for the profile matrix."""

    pressure_atm: float
    temperature_k: float
    envelope_hwhm_cm1: float = 0.0
    line_shape: LineShape = LineShape.VOIGT

    def __post_init__(self) -> None:
        if not (self.pressure_atm > 0 and self.temperature_k > 0):
            raise ValueError("pressure and temperature must be > 0")
        if self.envelope_hwhm_cm1 < 0:
            raise ValueError("envelope width must be >= 0")


class SpectralGrid:
    """
    Static tracking grid.  ``t`` never changes; both unit axes are derived once and
    frozen (read-only) so the GUI can hand them to pyqtgraph without copies.
    """

    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.t: FloatArray = np.linspace(0.0, 1.0, spec.n_points)
        nu_lo, nu_hi = spec.wavenumber_range_cm1
        lam_lo, lam_hi = spec.wavelength_range_um
        axis_nu = nu_lo + self.t * (nu_hi - nu_lo)
        axis_lam = lam_lo + self.t * (lam_hi - lam_lo)
        self._axis: Dict[Unit, FloatArray] = {Unit.WAVENUMBER: axis_nu, Unit.WAVELENGTH: axis_lam}
        # Physical wavenumber at each grid point (needed to evaluate line shapes).
        self._nu: Dict[Unit, FloatArray] = {
            Unit.WAVENUMBER: axis_nu,
            Unit.WAVELENGTH: wavelength_um_to_wavenumber(axis_lam),
        }
        self._limits = {
            Unit.WAVENUMBER: (float(nu_lo), float(nu_hi)),
            Unit.WAVELENGTH: (float(lam_lo), float(lam_hi)),
        }
        for arr in (self.t, *self._axis.values(), *self._nu.values()):
            arr.setflags(write=False)

    @property
    def n_points(self) -> int:
        return int(self.t.size)

    def axis(self, unit: Unit) -> FloatArray:
        """Plot-axis values (cm^-1 or um) for ``unit``."""
        return self._axis[Unit(unit)]

    def wavenumber(self, unit: Unit) -> FloatArray:
        """Physical wavenumber (cm^-1) at every grid point of ``unit``."""
        return self._nu[Unit(unit)]

    def limits(self, unit: Unit) -> Tuple[float, float]:
        return self._limits[Unit(unit)]


class SpectrumEngine:
    """
    Multi-gas forward model.  No Qt, no I/O.

    Index bookkeeping lives in plain dictionaries:

    * ``gas_index``   - ``{gas_id: column in tau matrix / slot in multiplier vector}``
    * ``line_slices`` - ``{gas_id: slice into the flattened line arrays}``
    """

    def __init__(
        self,
        grid: SpectralGrid,
        gases: Sequence[GasSpec],
        peaks: Mapping[str, IRPeakTable],
        tau_scale: float,
        conditions: AtmosphericConditions,
    ):
        if not gases:
            raise ValueError("at least one gas is required")
        self.grid = grid
        self.gases: Tuple[GasSpec, ...] = tuple(gases)
        self.tau_scale = float(tau_scale)
        self.conditions = conditions

        self.gas_index: Dict[str, int] = {g.gas_id: i for i, g in enumerate(self.gases)}
        self.line_slices: Dict[str, slice] = {}

        # ---- flatten every gas's peaks into shared (M,) arrays (one-time setup) ----
        nu_parts, int_parts, gas_parts = [], [], []
        start = 0
        for g in self.gases:
            tbl = peaks[g.gas_id]
            n = len(tbl)
            self.line_slices[g.gas_id] = slice(start, start + n)
            start += n
            nu_parts.append(tbl.wavenumber_cm1)
            int_parts.append(tbl.intensity_km_mol)
            gas_parts.append(np.full(n, self.gas_index[g.gas_id], dtype=np.int64))
        self._line_nu: FloatArray = np.concatenate(nu_parts)
        self._line_int: FloatArray = np.concatenate(int_parts)
        self._line_gas = np.concatenate(gas_parts)
        if self._line_nu.size == 0:
            raise ValueError("no IR-active lines in any gas")

        # Per-line physical parameters via fancy indexing of per-gas vectors.
        gamma0 = np.array([g.gamma0_cm1_atm for g in self.gases])
        n_exp = np.array([g.n_exponent for g in self.gases])
        mass = np.array([g.mass_amu for g in self.gases])
        self._line_gamma0 = gamma0[self._line_gas]
        self._line_nexp = n_exp[self._line_gas]
        self._line_mass = mass[self._line_gas]

        # Intensity-assignment matrix S (M, G): row i has I_i in the column of its gas.
        m_lines = self._line_nu.size
        self._S = np.zeros((m_lines, len(self.gases)))
        self._S[np.arange(m_lines), self._line_gas] = self._line_int

        self._cache: Dict[Unit, Tuple[AtmosphericConditions, FloatArray]] = {}
        self.profile_evaluations = 0       # diagnostics: how often Phi was rebuilt

    # ------------------------------------------------------------- properties
    @property
    def n_gases(self) -> int:
        return len(self.gases)

    @property
    def n_lines(self) -> int:
        return int(self._line_nu.size)

    def default_multipliers(self) -> FloatArray:
        return np.array([g.default_multiplier for g in self.gases], dtype=np.float64)

    # ------------------------------------------------------- conditions / cache
    def update_conditions(self, conditions: AtmosphericConditions) -> None:
        """Set the atmosphere.  Cheap: the matrix is rebuilt lazily on next use."""
        self.conditions = conditions

    def tau_matrix(self, unit: Unit) -> FloatArray:
        """
        Per-gas optical depth for a unit multiplier, shape ``(N, G)``.
        Cached per unit and rebuilt only when the conditions have changed.
        """
        unit = Unit(unit)
        hit = self._cache.get(unit)
        if hit is not None and hit[0] == self.conditions:
            return hit[1]

        tau = self.tau_on(self.grid.wavenumber(unit))
        self._cache[unit] = (self.conditions, tau)
        self.profile_evaluations += 1
        return tau

    def tau_on(self, nu_cm1: FloatArray) -> FloatArray:
        """
        Per-gas optical depth (unit multipliers) at ARBITRARY wavenumbers, ``(K, G)``.
        Evaluates the profiles exactly at ``nu_cm1`` (no interpolation); used by the
        inverse fitter to evaluate the model on an observation's own x-values.
        Not cached - callers (fit setup) invoke it once.
        """
        c = self.conditions
        x = np.asarray(nu_cm1, dtype=np.float64)[:, None]                # (K, 1)
        gamma_l = (
            lorentz_hwhm(c.pressure_atm, c.temperature_k, self._line_gamma0, self._line_nexp)
            + c.envelope_hwhm_cm1
        )[None, :]                                                       # (1, M)
        alpha_d = doppler_hwhm(self._line_nu, c.temperature_k, self._line_mass)[None, :]
        phi = line_profile_matrix(x, self._line_nu[None, :], gamma_l, alpha_d, c.line_shape)
        phi *= self.tau_scale                                            # in place: fresh array
        return phi @ self._S                                             # (K, G)

    # ------------------------------------------------------------------ model
    def optical_depth(self, unit: Unit, multipliers: FloatArray) -> FloatArray:
        """Total optical depth ``tau_total(x)``: an (N, G) @ (G,) mat-vec."""
        return self.tau_matrix(unit) @ np.asarray(multipliers, dtype=np.float64)

    def transmittance(self, unit: Unit, multipliers: FloatArray) -> FloatArray:
        """Beer-Lambert transmittance of the whole mixture, ``exp(-tau_total)``."""
        return np.exp(-self.optical_depth(unit, multipliers))

    def layer_transmittances(self, unit: Unit, multipliers: FloatArray) -> FloatArray:
        """Per-gas transmittance ``exp(-m_g tau_g)``, shape ``(N, G)`` (broadcast, no loop)."""
        return np.exp(-self.tau_matrix(unit) * np.asarray(multipliers, dtype=np.float64)[None, :])

    def multipliers_from_dict(self, values: Mapping[str, float]) -> FloatArray:
        """Build a multiplier vector from ``{gas_id: value}`` (unlisted gases -> default)."""
        m = self.default_multipliers()
        idx = np.fromiter((self.gas_index[k] for k in values), dtype=np.int64, count=len(values))
        m[idx] = np.fromiter(values.values(), dtype=np.float64, count=len(values))
        return m


@dataclass
class BuildReport:
    """Outcome of :func:`build_engine`: which gases loaded and which failed."""

    loaded: Tuple[str, ...] = ()
    errors: Dict[str, str] = field(default_factory=dict)


def build_engine(
    cfg: EcasConfig, parser: Optional[OrcaIRParser] = None
) -> Tuple[SpectrumEngine, BuildReport]:
    """
    Parse every enabled gas's ORCA track and assemble the engine.

    A gas whose file is missing/unparseable is skipped and reported rather than
    aborting start-up; if *no* gas survives, ``RuntimeError`` is raised.
    """
    parser = parser or OrcaIRParser()
    peaks: Dict[str, IRPeakTable] = {}
    ok, errors = [], {}
    for gas in cfg.active_gases:
        try:
            peaks[gas.gas_id] = load_peaks(gas.resolved_source(cfg.base_dir), parser)
            ok.append(gas)
        except (OSError, ValueError, KeyError) as exc:
            errors[gas.gas_id] = f"{type(exc).__name__}: {exc}"
    if not ok:
        raise RuntimeError(f"No gas could be loaded: {errors}")

    p = cfg.physics
    engine = SpectrumEngine(
        SpectralGrid(cfg.grid), ok, peaks, p.tau_scale,
        AtmosphericConditions(p.pressure_atm, p.temperature_k, p.envelope_hwhm_cm1, p.line_shape),
    )
    return engine, BuildReport(tuple(g.gas_id for g in ok), errors)
