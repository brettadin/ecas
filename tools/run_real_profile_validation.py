"""Run the bounded transmission-profile validation with real petitRADTRANS.

This is intentionally a command-line experiment rather than a UI action. It
keeps the expensive real-backend study reproducible and prevents validation
results from being confused with ordinary application output.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.atmosphere import (  # noqa: E402
    PetitRadtransParameters,
    compute_petit_radtrans_transmission,
    depth_spectrum_channel_edges,
    scan_single_gas_abundance,
)
from app.core.retrieval_validation import run_profile_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10,
                        help="number of independent noise seeds per injected VMR")
    parser.add_argument("--vmrs", type=float, nargs="+", default=[1e-4, 1e-3],
                        help="injected H2O VMR values")
    parser.add_argument("--offsets-ppm", type=float, nargs="+", default=[40.0],
                        help="constant depth offsets to test")
    parser.add_argument("--uncertainties-ppm", type=float, nargs="+", default=[2.5],
                        help="independent channel uncertainties to test")
    parser.add_argument("--synthetic-temperatures-k", type=float, nargs="+", default=[1200.0],
                        help="temperatures used only to generate synthetic data; the scan stays at 1200 K")
    parser.add_argument("--synthetic-gravities-cgs", type=float, nargs="+", default=[1000.0],
                        help="gravities used only to generate synthetic data; the scan stays at 1000 cgs")
    parser.add_argument("--synthetic-reference-pressures-bar", type=float, nargs="+", default=[0.01],
                        help="reference pressures used only for synthetic data; the scan stays at 0.01 bar")
    parser.add_argument("--synthetic-planet-radii-rjup", type=float, nargs="+", default=[1.0],
                        help="planet radii used only for synthetic data; the scan stays at 1 Rjup")
    parser.add_argument("--synthetic-stellar-radii-rsun", type=float, nargs="+", default=[1.0],
                        help="stellar radii used only for synthetic data; the scan stays at 1 Rsun")
    parser.add_argument("--synthetic-response-slopes", type=float, nargs="+", default=[0.0],
                        help="linear fractional throughput slopes used only for synthetic data; 0 is matched")
    parser.add_argument("--synthetic-lsf-sigmas-channels", type=float, nargs="+", default=[0.0],
                        help="Gaussian line-spread widths in channel units used only for synthetic data; 0 is matched")
    parser.add_argument("--synthetic-correlation-lengths-channels", type=float, nargs="+", default=[0.0],
                        help="exponential channel-noise correlation lengths used only for synthetic data; 0 is independent")
    parser.add_argument("--summary-only", action="store_true",
                        help="print aggregate and per-injection metrics without raw records")
    args = parser.parse_args()

    centers = np.linspace(2.4025, 2.4975, 21)
    edges = depth_spectrum_channel_edges(centers)
    # Keep a reasonably fine log grid and explicitly evaluate every injected
    # value so recovery/coverage is not confounded by grid omission.
    scan_grid = np.unique(np.sort(np.concatenate([
        np.logspace(-5, -2, 33), np.asarray(args.vmrs, dtype=float)
    ])))
    parameters = PetitRadtransParameters(
        temperature_k=1200.0,
        gravity_cgs=1000.0,
        planet_radius_rjup=1.0,
        stellar_radius_rsun=1.0,
        reference_pressure_bar=0.01,
        # The recorded native setup includes H2O HITEMP plus H2/He CIA. Keep
        # this validation run offline and do not trigger pRT's Keeper fallback
        # for uninstalled CO2/CH4 opacity files.
        mixing_ratios={"H2O": 1e-3},
        n_layers=100,
    )

    model_cache = {}

    def model_for_vmr(vmr: float, temperature_k: float, gravity_cgs: float,
                      reference_pressure_bar: float, planet_radius_rjup: float,
                      stellar_radius_rsun: float):
        cache_key = (float(vmr), float(temperature_k), float(gravity_cgs),
                     float(reference_pressure_bar), float(planet_radius_rjup),
                     float(stellar_radius_rsun))
        if cache_key in model_cache:
            return model_cache[cache_key]
        model_parameters = PetitRadtransParameters(
            temperature_k=temperature_k,
            gravity_cgs=gravity_cgs,
            planet_radius_rjup=planet_radius_rjup,
            stellar_radius_rsun=stellar_radius_rsun,
            reference_pressure_bar=reference_pressure_bar,
            mixing_ratios={**parameters.mixing_ratios, "H2O": vmr},
            n_layers=parameters.n_layers,
        )
        model = compute_petit_radtrans_transmission(
            model_parameters, (float(edges[0]), float(edges[-1])))
        model_cache[cache_key] = model
        return model

    def scan_observation(observation):
        return scan_single_gas_abundance(
            parameters, observation, "H2O", scan_grid)

    span = float(centers[-1] - centers[0])

    def make_response(slope: float):
        def response(wavelength):
            normalized = 2.0 * (np.asarray(wavelength) - float(centers[0])) / span - 1.0
            return 1.0 + float(slope) * normalized
        return response

    def evaluate_condition(temperature_k, gravity_cgs, reference_pressure_bar, planet_radius_rjup,
                           stellar_radius_rsun, response_slope, lsf_sigma_channels,
                           correlation_length_channels, uncertainty_ppm, offset_ppm):
        summary = run_profile_validation(
            model_for_vmr=lambda vmr: model_for_vmr(vmr, temperature_k, gravity_cgs,
                                                    reference_pressure_bar, planet_radius_rjup,
                                                    stellar_radius_rsun),
            scan_observation=scan_observation, wavelength_um=centers,
            uncertainty_ppm=uncertainty_ppm, injected_vmrs=args.vmrs,
            noise_seeds=range(args.seeds), depth_offset_ppm=offset_ppm,
            instrument_response=None if response_slope == 0.0 else make_response(response_slope),
            line_spread_sigma_um=(None if lsf_sigma_channels == 0.0 else
                                  lsf_sigma_channels * float(np.median(np.diff(centers)))),
            correlation_length_channels=(None if correlation_length_channels == 0.0 else
                                         correlation_length_channels),
        )
        condition = asdict(summary)
        condition.update({
            "synthetic_temperature_k": float(temperature_k), "synthetic_gravity_cgs": float(gravity_cgs),
            "synthetic_reference_pressure_bar": float(reference_pressure_bar),
            "synthetic_planet_radius_rjup": float(planet_radius_rjup),
            "synthetic_stellar_radius_rsun": float(stellar_radius_rsun),
            "synthetic_response_slope": float(response_slope),
            "synthetic_lsf_sigma_channels": float(lsf_sigma_channels),
            "synthetic_correlation_length_channels": float(correlation_length_channels),
            "scan_temperature_k": float(parameters.temperature_k), "scan_gravity_cgs": float(parameters.gravity_cgs),
            "scan_reference_pressure_bar": float(parameters.reference_pressure_bar),
            "scan_planet_radius_rjup": float(parameters.planet_radius_rjup),
            "scan_stellar_radius_rsun": float(parameters.stellar_radius_rsun),
            "fixed_parameter_misspecification": bool(
                temperature_k != parameters.temperature_k or gravity_cgs != parameters.gravity_cgs or
                reference_pressure_bar != parameters.reference_pressure_bar or
                planet_radius_rjup != parameters.planet_radius_rjup or
                stellar_radius_rsun != parameters.stellar_radius_rsun or
                response_slope != 0.0 or lsf_sigma_channels != 0.0 or
                correlation_length_channels != 0.0),
        })
        if args.summary_only:
            records = condition.pop("records")
            condition["by_injected_vmr"] = {
                f"{injected:.6g}": {
                    "n_cases": len(cell),
                    "median_absolute_log10_bias": float(np.median([abs(r["log10_bias"]) for r in cell])),
                    "interval_coverage_fraction": float(np.mean([r["interval_covers_injected"] for r in cell])),
                    "bounded_interval_fraction": float(np.mean([r["interval_bounded"] for r in cell])),
                    "boundary_limited_fraction": float(np.mean([r["boundary_limited"] for r in cell])),
                }
                for injected in sorted({r["injected_vmr"] for r in records})
                for cell in [[r for r in records if r["injected_vmr"] == injected]]
            }
        return condition

    payload = {"conditions": {}}
    for temperature_k in args.synthetic_temperatures_k:
        for gravity_cgs in args.synthetic_gravities_cgs:
            for pressure_bar in args.synthetic_reference_pressures_bar:
                for planet_radius in args.synthetic_planet_radii_rjup:
                    for stellar_radius in args.synthetic_stellar_radii_rsun:
                        for response_slope in args.synthetic_response_slopes:
                            for lsf_sigma in args.synthetic_lsf_sigmas_channels:
                                for correlation_length in args.synthetic_correlation_lengths_channels:
                                    for uncertainty_ppm in args.uncertainties_ppm:
                                        for offset_ppm in args.offsets_ppm:
                                            key = (f"T={temperature_k:g},g={gravity_cgs:g},P0={pressure_bar:g},"
                                                   f"Rp={planet_radius:g},Rstar={stellar_radius:g},"
                                                   f"response={response_slope:g},lsf={lsf_sigma:g},"
                                                   f"corr={correlation_length:g},"
                                                   f"sigma={uncertainty_ppm:g},offset={offset_ppm:g}")
                                            payload["conditions"][key] = evaluate_condition(
                                                temperature_k, gravity_cgs, pressure_bar, planet_radius,
                                                stellar_radius, response_slope, lsf_sigma,
                                                correlation_length,
                                                uncertainty_ppm, offset_ppm)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
