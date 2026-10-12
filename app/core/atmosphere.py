"""Optional petitRADTRANS forward modeling for clear, isothermal transmission spectra."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .transit import DepthSpectrum

FloatArray = NDArray[np.float64]

_R_JUP_CM = 7.1492e9
_R_SUN_CM = 6.957e10
_SPECIES_MOLAR_MASS = {
    "H2O": 18.01528,
    "CO2": 44.0095,
    "CH4": 16.0425,
    "O3": 47.9982,
    "N2O": 44.0128,
    "CO-NatAbund": 28.0101,
    "NH3": 17.0305,
    "HCN": 27.0253,
    "Na": 22.9898,
    "K": 39.0983,
}
_WINDOWS_FORTRAN_DLL_HANDLES: list[Any] = []
_WINDOWS_FORTRAN_DLL_PATHS: set[str] = set()


def _prepare_windows_fortran_dll_search() -> None:
    """Register discoverable MSYS2/Fortran runtime directories for Windows pRT imports."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    candidates: list[Path] = []
    compiler = shutil.which("gfortran")
    if compiler:
        candidates.append(Path(compiler).parent)
    for root_value in (
        os.environ.get("MSYS2_ROOT"),
        os.environ.get("MSYS2_HOME"),
        r"C:\msys64",
        str(Path.home() / "msys64"),
    ):
        if root_value:
            root = Path(root_value)
            candidates.extend(root / environment / "bin" for environment in (
                "ucrt64", "mingw64", "mingw32", "clang64"))
    candidates.extend(Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep) if entry)

    registered = _WINDOWS_FORTRAN_DLL_PATHS
    for directory in candidates:
        key = str(directory).casefold()
        if key not in registered and directory.is_dir() and any(directory.glob("libgfortran*.dll")):
            _WINDOWS_FORTRAN_DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            registered.add(key)


class AtmosphereModelError(ValueError):
    """Raised when an atmospheric forward model cannot be configured or evaluated."""


@dataclass
class PetitRadtransParameters:
    """Inputs for a deliberately simple isothermal, constant-composition pRT model."""

    temperature_k: float = 1200.0
    gravity_cgs: float = 1000.0
    planet_radius_rjup: float = 1.0
    stellar_radius_rsun: float = 1.0
    reference_pressure_bar: float = 0.01
    mixing_ratios: Dict[str, float] = field(default_factory=lambda: {
        "H2O": 1e-3, "CO2": 1e-4, "CH4": 1e-5,
    })
    n_layers: int = 100

    def __post_init__(self) -> None:
        positive = {
            "temperature_k": self.temperature_k,
            "gravity_cgs": self.gravity_cgs,
            "planet_radius_rjup": self.planet_radius_rjup,
            "stellar_radius_rsun": self.stellar_radius_rsun,
            "reference_pressure_bar": self.reference_pressure_bar,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise AtmosphereModelError(f"{name} must be finite and positive")
        if self.n_layers < 20:
            raise AtmosphereModelError("at least 20 pressure layers are required")
        if not self.mixing_ratios:
            raise AtmosphereModelError("provide at least one absorbing gas")
        unknown = sorted(set(self.mixing_ratios) - _SPECIES_MOLAR_MASS.keys())
        if unknown:
            raise AtmosphereModelError(
                "unsupported pRT gas name(s): " + ", ".join(unknown)
                + ". Supported: " + ", ".join(_SPECIES_MOLAR_MASS))
        if any(not np.isfinite(value) or value <= 0.0 for value in self.mixing_ratios.values()):
            raise AtmosphereModelError("gas volume mixing ratios must be finite and positive")
        if sum(self.mixing_ratios.values()) >= 1.0:
            raise AtmosphereModelError("the total trace-gas volume mixing ratio must be below 1")


@dataclass
class TransmissionAtmosphereModel:
    """A model transit-depth spectrum; distinct from measured :class:`DepthSpectrum` data."""

    wavelength_um: FloatArray
    depth_ppm: FloatArray
    radius_rjup: FloatArray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength_um = np.asarray(self.wavelength_um, dtype=np.float64)
        self.depth_ppm = np.asarray(self.depth_ppm, dtype=np.float64)
        self.radius_rjup = np.asarray(self.radius_rjup, dtype=np.float64)
        if (self.wavelength_um.ndim != 1 or self.depth_ppm.shape != self.wavelength_um.shape
                or self.radius_rjup.shape != self.wavelength_um.shape):
            raise AtmosphereModelError("model wavelength, depth, and radius arrays must be matching 1-D arrays")
        good = (np.isfinite(self.wavelength_um) & (self.wavelength_um > 0.0)
                & np.isfinite(self.depth_ppm) & np.isfinite(self.radius_rjup))
        if good.sum() < 8:
            raise AtmosphereModelError("pRT returned fewer than 8 finite model wavelength points")
        self.wavelength_um = self.wavelength_um[good]
        self.depth_ppm = self.depth_ppm[good]
        self.radius_rjup = self.radius_rjup[good]
        order = np.argsort(self.wavelength_um, kind="stable")
        self.wavelength_um = self.wavelength_um[order]
        self.depth_ppm = self.depth_ppm[order]
        self.radius_rjup = self.radius_rjup[order]
        if np.any(np.diff(self.wavelength_um) <= 0.0):
            raise AtmosphereModelError("model wavelengths must be unique")


@dataclass
class AtmosphereComparison:
    """Forward-model values integrated into measured channels; no model fit is implied."""

    wavelength_um: FloatArray
    measured_depth_ppm: FloatArray
    uncertainty_ppm: FloatArray
    model_depth_ppm: FloatArray
    residual_ppm: FloatArray
    residual_sigma: FloatArray
    chi_squared: float
    chi_squared_per_point: float


@dataclass
class AbundanceScanResult:
    """Conditional one-species profile scan with explicit edge diagnostics.

    The scan is a bounded likelihood profile: the gas VMR is evaluated only on
    ``volume_mixing_ratio_grid`` and the constant depth offset is solved
    analytically at every grid point.  The reported interval is therefore not
    a posterior interval and a missing bound means that the requested grid did
    not establish a Δχ²=1 crossing.
    """

    species: str
    volume_mixing_ratio_grid: FloatArray
    delta_chi_squared: FloatArray
    best_volume_mixing_ratio: float
    best_chi_squared: float
    best_depth_offset_ppm: float
    lower_1sigma_vmr: Optional[float]
    upper_1sigma_vmr: Optional[float]
    n_channels: int
    wavelength_um: FloatArray
    best_model_depth_ppm: FloatArray
    warnings: tuple[str, ...] = ()
    boundary_status: str = "interior"
    conditional_limit_kind: Optional[str] = None
    conditional_limit_vmr: Optional[float] = None


def depth_spectrum_channel_edges(wavelength_um: FloatArray) -> FloatArray:
    """Infer channel edges from strictly increasing wavelength centers."""
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    if (wavelength.ndim != 1 or wavelength.size < 2 or not np.isfinite(wavelength).all()
            or np.any(wavelength <= 0.0) or np.any(np.diff(wavelength) <= 0.0)):
        raise AtmosphereModelError("measured wavelengths must be at least two finite, positive, increasing centers")
    edges = np.empty(wavelength.size + 1, dtype=np.float64)
    edges[1:-1] = (wavelength[:-1] + wavelength[1:]) / 2.0
    edges[0] = wavelength[0] - (wavelength[1] - wavelength[0]) / 2.0
    edges[-1] = wavelength[-1] + (wavelength[-1] - wavelength[-2]) / 2.0
    return edges


def compare_model_to_depth_spectrum(
    model: TransmissionAtmosphereModel,
    measured: DepthSpectrum,
) -> AtmosphereComparison:
    """Integrate a model over inferred top-hat channel edges and compare without fitting."""
    wavelength = np.asarray(measured.wavelength_um, dtype=np.float64)
    depth_ppm = np.asarray(measured.depth, dtype=np.float64) * 1e6
    uncertainty_ppm = np.asarray(measured.depth_err, dtype=np.float64) * 1e6
    if depth_ppm.shape != wavelength.shape or uncertainty_ppm.shape != wavelength.shape:
        raise AtmosphereModelError("measured wavelength, depth, and uncertainty must be matching 1-D arrays")
    edges = depth_spectrum_channel_edges(wavelength)
    model_wavelength = model.wavelength_um
    if edges[0] < model_wavelength[0] or edges[-1] > model_wavelength[-1]:
        raise AtmosphereModelError(
            "model wavelength coverage does not include the full inferred measured-channel edges")

    cumulative = np.zeros_like(model_wavelength)
    cumulative[1:] = np.cumsum(
        np.diff(model_wavelength) * (model.depth_ppm[:-1] + model.depth_ppm[1:]) / 2.0)

    def integral_at(points: FloatArray) -> FloatArray:
        indices = np.searchsorted(model_wavelength, points, side="right") - 1
        indices = np.clip(indices, 0, model_wavelength.size - 2)
        dx = points - model_wavelength[indices]
        slopes = ((model.depth_ppm[indices + 1] - model.depth_ppm[indices])
                  / (model_wavelength[indices + 1] - model_wavelength[indices]))
        return cumulative[indices] + model.depth_ppm[indices] * dx + 0.5 * slopes * dx**2

    model_depth_ppm = (integral_at(edges[1:]) - integral_at(edges[:-1])) / np.diff(edges)
    valid = np.isfinite(depth_ppm) & np.isfinite(uncertainty_ppm) & (uncertainty_ppm > 0.0)
    if not valid.any():
        raise AtmosphereModelError("no measured channels have finite depths and positive uncertainties")
    residual_ppm = depth_ppm - model_depth_ppm
    residual_sigma = np.full_like(residual_ppm, np.nan)
    residual_sigma[valid] = residual_ppm[valid] / uncertainty_ppm[valid]
    chi_squared = float(np.sum(residual_sigma[valid] ** 2))
    n_valid = int(valid.sum())
    return AtmosphereComparison(
        wavelength, depth_ppm, uncertainty_ppm, model_depth_ppm, residual_ppm,
        residual_sigma, chi_squared, chi_squared / n_valid)


def vmr_to_mass_fractions(mixing_ratios: Mapping[str, float]) -> Tuple[Dict[str, float], float]:
    """Convert trace-gas volume mixing ratios and an H2/He remainder to pRT mass fractions."""
    if not mixing_ratios:
        raise AtmosphereModelError("provide at least one absorbing gas")
    unknown = sorted(set(mixing_ratios) - _SPECIES_MOLAR_MASS.keys())
    if unknown:
        raise AtmosphereModelError("unsupported gas name(s): " + ", ".join(unknown))
    trace_total = float(sum(mixing_ratios.values()))
    if (not np.isfinite(trace_total) or trace_total <= 0.0 or trace_total >= 1.0
            or any(not np.isfinite(v) or v <= 0.0 for v in mixing_ratios.values())):
        raise AtmosphereModelError("trace-gas mixing ratios must be positive and sum to less than 1")

    vmr = dict(mixing_ratios)
    background = 1.0 - trace_total
    helium_to_hydrogen = 0.17
    vmr["H2"] = background / (1.0 + helium_to_hydrogen)
    vmr["He"] = background * helium_to_hydrogen / (1.0 + helium_to_hydrogen)
    molar_masses = dict(_SPECIES_MOLAR_MASS, H2=2.01588, He=4.002602)
    mean_molar_mass = sum(vmr[species] * molar_masses[species] for species in vmr)
    mass_fractions = {
        species: vmr[species] * molar_masses[species] / mean_molar_mass
        for species in vmr
    }
    return mass_fractions, mean_molar_mass


def _create_prt_atmosphere(
    parameters: PetitRadtransParameters,
    wavelength_range_um: Tuple[float, float],
    radtrans_factory: Callable[..., Any],
) -> Any:
    lo, hi = map(float, wavelength_range_um)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise AtmosphereModelError("wavelength range must be finite, positive, and increasing")
    lo *= 0.995
    hi *= 1.005
    return radtrans_factory(
        pressures=np.logspace(-6, 2, parameters.n_layers),
        line_species=list(parameters.mixing_ratios),
        rayleigh_species=["H2", "He"],
        gas_continuum_contributors=["H2--H2", "H2--He"],
        wavelength_boundaries=[lo, hi],
    )


def _calculate_prt_model(
    atmosphere: Any,
    parameters: PetitRadtransParameters,
    mixing_ratios: Mapping[str, float],
    model_version: str,
) -> TransmissionAtmosphereModel:
    mass_fractions, mean_molar_mass = vmr_to_mass_fractions(mixing_ratios)
    temperatures = np.full(parameters.n_layers, parameters.temperature_k, dtype=np.float64)
    model_wavelength_cm, radius_cm, _ = atmosphere.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions={
            species: np.full(parameters.n_layers, fraction, dtype=np.float64)
            for species, fraction in mass_fractions.items()
        },
        mean_molar_masses=np.full(parameters.n_layers, mean_molar_mass, dtype=np.float64),
        reference_gravity=parameters.gravity_cgs,
        planet_radius=parameters.planet_radius_rjup * _R_JUP_CM,
        reference_pressure=parameters.reference_pressure_bar,
    )

    wavelength_um = np.asarray(model_wavelength_cm, dtype=np.float64) * 1e4
    radius_cm = np.asarray(radius_cm, dtype=np.float64)
    if wavelength_um.shape != radius_cm.shape:
        raise AtmosphereModelError("petitRADTRANS returned incompatible wavelength and radius arrays")
    radius_rjup = radius_cm / _R_JUP_CM
    depth_ppm = (radius_cm / (parameters.stellar_radius_rsun * _R_SUN_CM)) ** 2 * 1e6
    return TransmissionAtmosphereModel(
        wavelength_um, depth_ppm, radius_rjup,
        meta={
            "model": "petitRADTRANS",
            "model_version": model_version,
            "radiative_transfer": "correlated-k default",
            "temperature_k": parameters.temperature_k,
            "gravity_cm_s2": parameters.gravity_cgs,
            "planet_radius_rjup_at_reference_pressure": parameters.planet_radius_rjup,
            "stellar_radius_rsun": parameters.stellar_radius_rsun,
            "reference_pressure_bar": parameters.reference_pressure_bar,
            "volume_mixing_ratios": dict(mixing_ratios),
            "line_species": list(mixing_ratios),
            "assumptions": [
                "isothermal temperature profile",
                "constant volume mixing ratios with altitude",
                "clear atmosphere with H2/He background",
                "hydrostatic transmission geometry",
                "no instrumental line-spread convolution",
            ],
        },
    )


def scan_single_gas_abundance(
    parameters: PetitRadtransParameters,
    measured: DepthSpectrum,
    species: str,
    volume_mixing_ratios: FloatArray,
    radtrans_factory: Optional[Callable[..., Any]] = None,
) -> AbundanceScanResult:
    """Profile-scan one gas VMR with all other inputs fixed and a fitted constant depth offset.

    The Δχ²=1 interval is a conditional, asymptotic one-parameter diagnostic. It does not include
    correlated noise, instrument response, model inadequacy, or uncertainty in other model inputs.
    """
    if species not in parameters.mixing_ratios:
        raise AtmosphereModelError(f"{species} is not included in the configured opacity species")
    vmr_grid = np.asarray(volume_mixing_ratios, dtype=np.float64)
    if (vmr_grid.ndim != 1 or vmr_grid.size < 3 or not np.isfinite(vmr_grid).all()
            or np.any(vmr_grid <= 0.0) or np.any(np.diff(vmr_grid) <= 0.0)):
        raise AtmosphereModelError("abundance scan requires at least three positive, increasing VMR values")
    if np.any(vmr_grid + sum(parameters.mixing_ratios.values()) - parameters.mixing_ratios[species] >= 1.0):
        raise AtmosphereModelError("abundance scan values and fixed gases must sum to less than 1")
    edges = depth_spectrum_channel_edges(measured.wavelength_um)
    wavelength_range = (float(edges[0]), float(edges[-1]))
    if radtrans_factory is None:
        _prepare_windows_fortran_dll_search()
        try:
            import petitRADTRANS
            from petitRADTRANS.radtrans import Radtrans
        except ImportError as exc:
            raise AtmosphereModelError(
                "petitRADTRANS 3 could not be imported. Install it with the optional atmosphere "
                "setup described in README.md and ensure its Fortran runtime is available; on "
                "Windows, restart ECAS after installing the compiler/runtime."
            ) from exc
        radtrans_factory = Radtrans
        model_version = str(getattr(petitRADTRANS, "__version__", "unknown"))
    else:
        model_version = "injected test backend"
    atmosphere = _create_prt_atmosphere(parameters, wavelength_range, radtrans_factory)
    measured_depth = np.asarray(measured.depth, dtype=np.float64) * 1e6
    uncertainty = np.asarray(measured.depth_err, dtype=np.float64) * 1e6
    valid = np.isfinite(measured_depth) & np.isfinite(uncertainty) & (uncertainty > 0.0)
    if not valid.any():
        raise AtmosphereModelError("no measured channels have finite depths and positive uncertainties")
    weights = 1.0 / uncertainty[valid] ** 2
    chi_squared = np.empty(vmr_grid.size, dtype=np.float64)
    offsets = np.empty(vmr_grid.size, dtype=np.float64)
    channel_models: list[FloatArray] = []
    for index, vmr in enumerate(vmr_grid):
        mixing_ratios = dict(parameters.mixing_ratios)
        mixing_ratios[species] = float(vmr)
        model = _calculate_prt_model(atmosphere, parameters, mixing_ratios, model_version)
        comparison = compare_model_to_depth_spectrum(model, measured)
        offset = float(np.average(
            comparison.measured_depth_ppm[valid] - comparison.model_depth_ppm[valid],
            weights=weights))
        offsets[index] = offset
        channel_models.append(comparison.model_depth_ppm + offset)
        chi_squared[index] = float(np.sum(
            ((comparison.measured_depth_ppm[valid] - comparison.model_depth_ppm[valid] - offset)
             / comparison.uncertainty_ppm[valid]) ** 2))
    best_index = int(np.argmin(chi_squared))
    delta_chi_squared = chi_squared - chi_squared[best_index]
    log_vmr = np.log10(vmr_grid)

    def crossing(indices: range) -> Optional[float]:
        for i in indices:
            j = i + 1
            if (delta_chi_squared[i] - 1.0) * (delta_chi_squared[j] - 1.0) <= 0.0:
                if delta_chi_squared[i] == 1.0:
                    return float(vmr_grid[i])
                if delta_chi_squared[j] == 1.0:
                    return float(vmr_grid[j])
                fraction = (1.0 - delta_chi_squared[i]) / (
                    delta_chi_squared[j] - delta_chi_squared[i])
                return float(10.0 ** (log_vmr[i] + fraction * (log_vmr[j] - log_vmr[i])))
        return None

    lower = crossing(range(best_index - 1, -1, -1)) if best_index > 0 else None
    upper = crossing(range(best_index, vmr_grid.size - 1)) if best_index < vmr_grid.size - 1 else None
    warnings: list[str] = []
    if best_index == 0 or best_index == vmr_grid.size - 1:
        warnings.append("best-fit VMR is at the scan boundary; expand the grid before interpreting it")
    if lower is None:
        warnings.append("lower Δχ²=1 bound is not established within the scan grid")
    if upper is None:
        warnings.append("upper Δχ²=1 bound is not established within the scan grid")
    boundary_status = "interior"
    conditional_limit_kind: Optional[str] = None
    conditional_limit_vmr: Optional[float] = None
    if best_index == 0:
        boundary_status = "lower_scan_edge"
        if upper is not None:
            conditional_limit_kind = "upper"
            conditional_limit_vmr = upper
    elif best_index == vmr_grid.size - 1:
        boundary_status = "upper_scan_edge"
        if lower is not None:
            conditional_limit_kind = "lower"
            conditional_limit_vmr = lower
    return AbundanceScanResult(
        species, vmr_grid, delta_chi_squared, float(vmr_grid[best_index]),
        float(chi_squared[best_index]), float(offsets[best_index]), lower, upper, int(valid.sum()),
        np.asarray(measured.wavelength_um, dtype=np.float64).copy(),
        channel_models[best_index], tuple(warnings), boundary_status,
        conditional_limit_kind, conditional_limit_vmr)


def compute_petit_radtrans_transmission(
    parameters: PetitRadtransParameters,
    wavelength_range_um: Tuple[float, float],
    radtrans_factory: Optional[Callable[..., Any]] = None,
) -> TransmissionAtmosphereModel:
    """Calculate a clear, isothermal transmission spectrum with petitRADTRANS 3."""
    lo, hi = map(float, wavelength_range_um)
    if radtrans_factory is None:
        _prepare_windows_fortran_dll_search()
        try:
            import petitRADTRANS
            from petitRADTRANS.radtrans import Radtrans
        except ImportError as exc:
            raise AtmosphereModelError(
                "petitRADTRANS 3 could not be imported. Install it with the optional atmosphere "
                "setup described in README.md and ensure its Fortran runtime is available; on "
                "Windows, restart ECAS after installing the compiler/runtime."
            ) from exc
        radtrans_factory = Radtrans
        model_version = str(getattr(petitRADTRANS, "__version__", "unknown"))
    else:
        model_version = "injected test backend"

    atmosphere = _create_prt_atmosphere(parameters, (lo, hi), radtrans_factory)
    return _calculate_prt_model(
        atmosphere, parameters, parameters.mixing_ratios, model_version)
