"""
ECAS exports (Phase 6) - Qt-free CSV writers and a Markdown report builder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from .engine import AtmosphericConditions, SpectrumEngine, Unit
from .fitting import Anomaly, Candidate, FitResult, ObservedSpectrum
from .sources import CitationLog
from .transit import DepthSpectrum

PathLike = Union[str, Path]


def export_model_csv(path: PathLike, engine: SpectrumEngine, unit: Unit, multipliers) -> None:
    """Axis, total transmittance, then one per-gas transmittance column (all vectorised)."""
    m = np.asarray(multipliers, dtype=float)
    table = np.column_stack([engine.grid.axis(unit), engine.transmittance(unit, m),
                             engine.layer_transmittances(unit, m)])
    header = ",".join([unit.value, "transmittance", *(f"T_{g.gas_id}" for g in engine.gases)])
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.8g")


def export_fit_csv(path: PathLike, obs: ObservedSpectrum, fit: FitResult) -> None:
    """Observation grid: x, observed, model, residual, sigma, residual/sigma."""
    used = fit.used                                   # the points actually fitted (window applied)
    table = np.column_stack([used.x, fit.normalised_y if fit.normalised_y is not None else used.y,
                             fit.model, fit.residual, fit.sigma, fit.residual / fit.sigma])
    np.savetxt(path, table, delimiter=",", comments="", fmt="%.8g",
               header=f"{used.unit.value},observed,model,residual,sigma,residual_sigma")


def export_anomalies_csv(path: PathLike, anomalies: Sequence[Anomaly]) -> None:
    rows = ["nu_peak_cm1,wavelength_um,nu_lo_cm1,nu_hi_cm1,significance_sigma,peak_residual,n_points"]
    rows += [f"{a.nu_peak_cm1:.3f},{1e4 / a.nu_peak_cm1:.4f},{a.nu_lo_cm1:.3f},{a.nu_hi_cm1:.3f},"
             f"{a.significance_sigma:.2f},{a.peak_residual:.5g},{a.n_points}" for a in anomalies]
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def export_depth_spectrum_csv(path: PathLike, spectrum: DepthSpectrum) -> None:
    """Export the measured transit-depth spectrum and 1-sigma errors in explicit units."""
    table = np.column_stack([spectrum.wavelength_um, spectrum.depth * 1e6, spectrum.depth_err * 1e6])
    np.savetxt(path, table, delimiter=",", comments="", fmt="%.8g",
               header="wavelength_um,transit_depth_ppm,uncertainty_1sigma_ppm")


def build_report_markdown(
    engine: SpectrumEngine,
    conditions: AtmosphericConditions,
    multipliers,
    fit: Optional[FitResult] = None,
    anomalies: Sequence[Anomaly] = (),
    candidates: Sequence[Candidate] = (),
    transit_lines: Sequence[str] = (),
    notes: str = "",
    identification_lines: Sequence[str] = (),
    citations: Optional[CitationLog] = None,
) -> str:
    """Self-contained Markdown summary of the current analysis."""
    m = np.asarray(multipliers, dtype=float)
    out = ["# ECAS analysis report", ""]
    if notes:
        out += [notes, ""]
    out += ["## Atmosphere", "",
            f"- Pressure: {conditions.pressure_atm:.4g} atm",
            f"- Temperature: {conditions.temperature_k:.1f} K",
            f"- Band envelope HWHM: {conditions.envelope_hwhm_cm1:.2f} cm⁻¹",
            f"- Line shape: {conditions.line_shape.value}", "",
            "## Gas multipliers", "", "| Gas | Multiplier | ±1σ |", "|---|---:|---:|"]
    for g in engine.gases:
        i = engine.gas_index[g.gas_id]
        unc = f"{fit.uncertainties[i]:.3g}" if fit is not None else "–"
        out.append(f"| {g.formula} {g.name} | {m[i]:.3f} | {unc} |")
    if fit is not None:
        out += ["", "## Fit", "", f"- χ² = {fit.chi2:.2f}, reduced χ² = {fit.reduced_chi2:.2f}",
                f"- Noise σ {'estimated from residual MAD' if fit.sigma_estimated else 'from input'}",
                f"- Solver: {fit.message}"]
    if anomalies:
        out += ["", "## Unidentified signatures", "", "| ν (cm⁻¹) | λ (µm) | Significance | Type |",
                "|---:|---:|---:|---|"]
        out += [f"| {a.nu_peak_cm1:.1f} | {1e4 / a.nu_peak_cm1:.2f} | {a.significance_sigma:.1f}σ | "
                f"{'extra absorption' if a.extra_absorption else 'excess flux'} |" for a in anomalies]
    if candidates:
        out += ["", "Candidate molecules: " + ", ".join(f"{c.molecule} ({c.score:.0%})" for c in candidates)]
    if transit_lines:
        out += ["", "## Transit photometry", "", "```", *transit_lines, "```"]
    if identification_lines:
        out += ["", "## Molecule identification", "", "```", *identification_lines, "```"]
    if citations is not None:
        out += ["", citations.references_markdown().rstrip()]
    return "\n".join(out) + "\n"
