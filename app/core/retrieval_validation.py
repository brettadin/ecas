"""Repeatable synthetic validation helpers for conditional transmission scans.

This module deliberately does not choose a forward model or a sampler.  Callers
can generate a spectrum with the real petitRADTRANS adapter, pass it through
``make_synthetic_depth_spectrum``, run the existing profile scan, and then use
``summarize_profile_scan`` to measure recovery bias and interval coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .atmosphere import AbundanceScanResult, AtmosphereModelError, TransmissionAtmosphereModel
from .transit import DepthSpectrum


@dataclass(frozen=True)
class ProfileValidationRecord:
    """Outcome of one injected VMR/noise-seed validation case."""

    injected_vmr: float
    noise_seed: int
    recovered_vmr: float
    log10_bias: float
    interval_covers_injected: bool
    interval_bounded: bool
    boundary_limited: bool


@dataclass(frozen=True)
class ProfileValidationSummary:
    """Aggregate diagnostics for a collection of conditional profile scans."""

    records: tuple[ProfileValidationRecord, ...]
    median_absolute_log10_bias: float
    interval_coverage_fraction: float
    bounded_interval_fraction: float
    boundary_limited_fraction: float


def make_synthetic_depth_spectrum(
    model: TransmissionAtmosphereModel,
    wavelength_um: np.ndarray,
    uncertainty_ppm: float | np.ndarray,
    depth_offset_ppm: float = 0.0,
    seed: int = 0,
) -> DepthSpectrum:
    """Sample a noisy channel-center observation from a model spectrum.

    The function uses independent Gaussian noise in transit depth, matching the
    current quick-look likelihood.  It interpolates the model at supplied
    measured channel centers and intentionally does not emulate throughput,
    line-spread functions, or correlated systematics.
    """
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    uncertainty = np.asarray(uncertainty_ppm, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2 or not np.isfinite(wavelength).all():
        raise AtmosphereModelError("synthetic wavelengths must be a finite 1-D array with at least two points")
    if np.any(np.diff(wavelength) <= 0.0):
        raise AtmosphereModelError("synthetic wavelengths must be strictly increasing")
    if uncertainty.ndim == 0:
        uncertainty = np.full(wavelength.shape, float(uncertainty), dtype=np.float64)
    if uncertainty.shape != wavelength.shape or not np.isfinite(uncertainty).all() or np.any(uncertainty <= 0.0):
        raise AtmosphereModelError("synthetic uncertainties must be positive and match wavelength centers")
    if wavelength[0] < model.wavelength_um[0] or wavelength[-1] > model.wavelength_um[-1]:
        raise AtmosphereModelError("synthetic wavelengths are outside the model coverage")
    if not np.isfinite(depth_offset_ppm):
        raise AtmosphereModelError("synthetic depth offset must be finite")
    rng = np.random.default_rng(seed)
    model_depth = np.interp(wavelength, model.wavelength_um, model.depth_ppm)
    noisy_depth_ppm = model_depth + float(depth_offset_ppm) + rng.normal(0.0, uncertainty)
    return DepthSpectrum(wavelength, noisy_depth_ppm * 1e-6, uncertainty * 1e-6)


def _validation_record(
    result: AbundanceScanResult,
    injected_vmr: float,
    noise_seed: int,
) -> ProfileValidationRecord:
    if not np.isfinite(injected_vmr) or injected_vmr <= 0.0:
        raise AtmosphereModelError("injected VMR must be finite and positive")
    bounded = result.lower_1sigma_vmr is not None and result.upper_1sigma_vmr is not None
    covers = bool(
        bounded
        and result.lower_1sigma_vmr <= injected_vmr <= result.upper_1sigma_vmr
    )
    return ProfileValidationRecord(
        float(injected_vmr), int(noise_seed), result.best_volume_mixing_ratio,
        float(np.log10(result.best_volume_mixing_ratio / injected_vmr)), covers,
        bounded, any("boundary" in warning for warning in result.warnings),
    )


def summarize_profile_scan(
    results: Iterable[tuple[AbundanceScanResult, float, int]],
) -> ProfileValidationSummary:
    """Summarize recovery bias and interval coverage across scan outcomes.

    ``interval_coverage_fraction`` counts an unbounded interval as not covered;
    this is intentionally conservative for deciding whether a proposed
    uncertainty method is usable.  Empty input is rejected rather than
    returning misleading zero-valued diagnostics.
    """
    records = tuple(_validation_record(result, injected_vmr, seed)
                    for result, injected_vmr, seed in results)
    if not records:
        raise AtmosphereModelError("at least one profile-scan result is required")
    return ProfileValidationSummary(
        records=records,
        median_absolute_log10_bias=float(np.median([abs(record.log10_bias) for record in records])),
        interval_coverage_fraction=float(np.mean([record.interval_covers_injected for record in records])),
        bounded_interval_fraction=float(np.mean([record.interval_bounded for record in records])),
        boundary_limited_fraction=float(np.mean([record.boundary_limited for record in records])),
    )


def run_profile_validation(
    model_for_vmr: Callable[[float], TransmissionAtmosphereModel],
    scan_observation: Callable[[DepthSpectrum], AbundanceScanResult],
    wavelength_um: np.ndarray,
    uncertainty_ppm: float | np.ndarray,
    injected_vmrs: Iterable[float],
    noise_seeds: Iterable[int],
    depth_offset_ppm: float = 0.0,
) -> ProfileValidationSummary:
    """Run a seeded grid of synthetic profile-scan validation cases.

    ``model_for_vmr`` should construct the forward model using the real backend
    when validating pRT.  ``scan_observation`` should bind the agreed fixed
    parameters, gas species, VMR grid, and backend, leaving only the synthetic
    observation as its argument.  Keeping both callbacks explicit makes it
    difficult to accidentally validate a different model from the one being
    scanned.
    """
    vmrs = tuple(float(vmr) for vmr in injected_vmrs)
    seeds = tuple(int(seed) for seed in noise_seeds)
    if not vmrs or not seeds:
        raise AtmosphereModelError("validation requires at least one injected VMR and one noise seed")
    if not np.isfinite(depth_offset_ppm):
        raise AtmosphereModelError("validation depth offset must be finite")
    outcomes: list[tuple[AbundanceScanResult, float, int]] = []
    for injected_vmr in vmrs:
        model = model_for_vmr(injected_vmr)
        for seed in seeds:
            observation = make_synthetic_depth_spectrum(
                model, wavelength_um, uncertainty_ppm, depth_offset_ppm, seed)
            outcomes.append((scan_observation(observation), injected_vmr, seed))
    return summarize_profile_scan(outcomes)
