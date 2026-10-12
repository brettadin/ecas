"""
ECAS mock-data generator.

Lets the whole pipeline (parser -> physics -> dashboard) run with no ORCA files on
disk.  Two entry points mirror the two things the parser deals with:

* :func:`mock_peak_table` - the parser's *output* (arrays), for unit tests of the
  physics/engine layers.
* :func:`render_mock_orca_output` - ORCA-style *text*, for exercising the regex
  parser itself.  Several structural variants are produced on purpose.

The (wavenumber, intensity) pairs below are representative harmonic-level values
for r2SCAN-3c-type calculations, NOT production data; replace them by pointing a
gas entry in ``gases.json`` at a real ORCA ``.out`` file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .parser import IRPeakTable

# (wavenumber cm^-1, intensity km/mol); degenerate modes are listed once per mode,
# exactly as ORCA prints them.  IR-silent modes are included on purpose (they must
# be filtered by the parser).
MOCK_MODES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "H2O": ((1595.0, 63.0), (3657.0, 4.5), (3756.0, 58.0)),
    "CO2": ((667.3, 28.0), (667.3, 28.0), (1340.0, 0.0), (2349.0, 550.0)),
    "CH4": (
        (1306.0, 11.0), (1306.0, 11.0), (1306.0, 11.0),
        (1534.0, 0.0), (1534.0, 0.0),
        (2917.0, 0.0),
        (3019.0, 20.0), (3019.0, 20.0), (3019.0, 20.0),
    ),
    "O3": ((701.0, 9.5), (1042.0, 11.0), (1103.0, 130.0)),
    "N2O": ((589.0, 14.0), (589.0, 14.0), (1285.0, 9.0), (2224.0, 260.0)),
    "SO2": ((518.0, 10.0), (1151.0, 20.0), (1362.0, 90.0)),
}
_LINEAR = frozenset({"CO2", "N2O"})        # 5 zero modes instead of 6
VARIANTS = ("orca5", "orca4", "orca6", "bare")
_AU_TO_KM_MOL = 974.88                     # 1 a.u. of T**2 in km/mol


def available_mock_gases() -> Tuple[str, ...]:
    """Keys accepted by the generators and by ``mock://<KEY>`` sources."""
    return tuple(MOCK_MODES)


def _modes(key: str, jitter_cm1: float, seed: Optional[int], include_imaginary: bool):
    key = key.upper()
    if key not in MOCK_MODES:
        raise KeyError(f"Unknown mock gas {key!r}; available: {available_mock_gases()}")
    freq = np.array([m[0] for m in MOCK_MODES[key]], dtype=np.float64)
    inten = np.array([m[1] for m in MOCK_MODES[key]], dtype=np.float64)
    if jitter_cm1:
        freq = freq + np.random.default_rng(seed).normal(0.0, jitter_cm1, freq.shape)
    n_zero = 5 if key in _LINEAR else 6
    pad_f = np.zeros(n_zero)
    pad_i = np.zeros(n_zero)
    if include_imaginary:
        pad_f[0] = -35.2                    # a small imaginary mode, as printed by ORCA
    return np.concatenate([pad_f, freq]), np.concatenate([pad_i, inten])


# ------------------------------------------------------------ array generator
def mock_peak_table(
    key: str,
    jitter_cm1: float = 0.0,
    seed: Optional[int] = 0,
) -> IRPeakTable:
    """Return what the parser *would* output for ``key`` (zero/silent modes removed)."""
    freq, inten = _modes(key, jitter_cm1, seed, include_imaginary=False)
    mode = np.arange(freq.size, dtype=np.int64)
    keep = (freq > 1.0) & (inten > 0.0)
    order = np.argsort(freq[keep], kind="stable")
    return IRPeakTable(
        wavenumber_cm1=freq[keep][order],
        intensity_km_mol=inten[keep][order],
        mode_index=mode[keep][order],
        meta={"source": f"mock-array://{key.upper()}", "n_kept": int(keep.sum())},
    )


# ------------------------------------------------------------- text generator
_PREAMBLE = """\
                                 *****************
                                 * O   R   C   A *
                                 *****************

 Requested: IR SPECTRUM and thermochemistry (decoy line, not a block title)

---------------------------
VIBRATIONAL FREQUENCIES
---------------------------

Scaling factor for frequencies =  1.000000000  (already applied!)

"""
_FOOTNOTES = """\
* The epsilon (eps) is given for a Dirac delta lineshape.
*   eps = (2*Pi*N_A/ln(10))*(rpi/ (4*Pi*eps_0*3*h*c))*|T|^2*nu
* The Int is given by Int = eps / (L/(mol*cm)) * ...
* The T**2 are given in atomic units; TX, TY, TZ are the transition dipole components.
"""


def _row(variant: str, mode: int, f: float, i: float) -> str:
    t2 = i / _AU_TO_KM_MOL
    tx = np.sqrt(t2 / 3.0)
    if variant == "orca4":
        return f"{mode:4d}:{f:12.2f}{i:14.2f}  ({tx:10.6f} {tx:10.6f} {tx:10.6f})"
    if variant == "orca6":                  # scientific notation, tighter spacing
        return f"{mode:3d}: {f:9.2f} {i * 40:12.5e} {i:9.2f} {t2:12.6e}  ({tx:.6e} {tx:.6e} {tx:.6e})"
    return f"{mode:4d}:{f:10.2f}{i * 40:11.6f}{i:10.2f}{t2:11.6f}  ({tx:9.6f} {tx:9.6f} {tx:9.6f})"


_HEADERS = {
    "orca4": (" Mode   freq (cm**-1)   T**2 (km/mol)   TX        TY        TZ", ""),
    "orca5": (" Mode   freq       eps      Int      T**2         TX        TY        TZ",
              "       cm**-1   L/(mol*cm) km/mol    a.u."),
    "orca6": (" Mode   freq       eps      Int      T**2         TX        TY        TZ",
              "       cm**-1   L/(mol*cm) km/mol    a.u."),
    "bare": (" Mode   freq       eps      Int      T**2         TX        TY        TZ",
             "       cm**-1   L/(mol*cm) km/mol    a.u."),
}
_TAILS = {
    "orca5": _FOOTNOTES + "\n-------------------------\nTHERMOCHEMISTRY AT 1 atm\n-------------------------\n",
    "orca6": _FOOTNOTES,
    "orca4": "\n***************************\n*  ORCA THERMOCHEMISTRY   *\n***************************\n",
    "bare": "",                              # no terminator at all: parser must hit EOF
}


def render_mock_orca_output(
    key: str,
    variant: str = "orca5",
    jitter_cm1: float = 0.0,
    seed: Optional[int] = 0,
    include_imaginary: bool = False,
) -> str:
    """
    Build an ORCA-flavoured ``.out`` text containing an IR SPECTRUM block.

    ``variant`` selects the structural flavour: ``orca5`` (eps/Int columns plus
    asterisk footnotes), ``orca4`` (``T**2 (km/mol)`` column, banner terminator),
    ``orca6`` (CRLF line endings, scientific notation) and ``bare`` (no terminator,
    parser must stop at end-of-file).  The preamble contains decoy lines that look
    like mode rows and mention "IR SPECTRUM" to prove the parser anchors correctly.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    freq, inten = _modes(key, jitter_cm1, seed, include_imaginary)

    decoys: List[str] = [f"{i:4d}:{0.0:12.2f} cm**-1" for i in range(3)]
    header, units = _HEADERS[variant]
    rule = "-" * 76
    lines: List[str] = [_PREAMBLE.rstrip("\n"), *decoys, "",
                        "-----------", "IR SPECTRUM", "-----------", "", header]
    if units:
        lines.append(units)
    lines.append(rule)
    lines += [_row(variant, m, float(f), float(i)) for m, (f, i) in enumerate(zip(freq, inten))]
    lines += ["", _TAILS[variant], "TOTAL RUN TIME: 0 days 0 hours 0 minutes 4 seconds"]
    text = "\n".join(lines) + "\n"
    return text.replace("\n", "\r\n") if variant == "orca6" else text


def write_mock_orca_files(
    directory: Path, keys: Optional[Sequence[str]] = None, variant: str = "orca5"
) -> Dict[str, Path]:
    """Write ``<KEY>.out`` files (for testing real-file loading) and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for key in keys or available_mock_gases():
        path = directory / f"{key}.out"
        path.write_text(render_mock_orca_output(key, variant=variant), encoding="utf-8", newline="")
        out[key] = path
    return out


# ------------------------------------------------------- mock observation (Phase 4)
def make_mock_observation(
    engine,
    unit=None,
    truth: Optional[Dict[str, float]] = None,
    anomaly: Optional[Tuple[str, float]] = ("SO2", 3.0),
    n_points: int = 1500,
    noise: float = 0.004,
    seed: Optional[int] = 1,
):
    """
    Synthesise a noisy "observed" transmittance spectrum for exercising the inverse engine.

    ``truth`` maps gas ids -> multipliers (others keep their defaults).  ``anomaly`` =
    ``(mock_key, multiplier)`` injects a molecule the engine does NOT contain, so it must
    surface as an unidentified residual; pass ``None`` for a clean observation.
    """
    from .engine import Unit
    from .fitting import ObservedSpectrum
    from .physics import lorentzian

    unit = unit or Unit.WAVENUMBER
    rng = np.random.default_rng(seed)
    x = np.linspace(450.0, 4100.0, n_points) if unit is Unit.WAVENUMBER else np.linspace(2.5, 24.0, n_points)
    nu = x if unit is Unit.WAVENUMBER else 1.0e4 / x

    mult = engine.multipliers_from_dict(truth or {})
    tau = engine.tau_on(nu) @ mult
    if anomaly is not None:
        tbl = mock_peak_table(anomaly[0])
        c = engine.conditions
        gamma = c.envelope_hwhm_cm1 + 0.1 * c.pressure_atm
        tau = tau + anomaly[1] * engine.tau_scale * (
            lorentzian(nu[:, None], tbl.wavenumber_cm1[None, :], gamma) @ tbl.intensity_km_mol
        )
    y = np.exp(-tau) + rng.normal(0.0, noise, x.shape)
    return ObservedSpectrum(x, y, unit, None,
                            meta={"source": "mock-observation", "truth": dict(zip(
                                (g.gas_id for g in engine.gases), mult)), "anomaly": anomaly,
                                "noise": noise})


# ------------------------------------------------------ mock photometry (Phase 5)
def make_mock_lightcurve(
    period: float = 3.2,
    rp_rs: float = 0.1,
    a_rs: float = 9.0,
    b: float = 0.3,
    t0: float = 1.1,
    span_days: float = 14.0,
    cadence_min: float = 4.0,
    noise_ppm: float = 600.0,
    trend_amp: float = 0.004,
    seed: Optional[int] = 3,
):
    """
    Synthetic stellar photometry: transits + slow variability + white noise.  Returns
    ``(LightCurve, TransitParams)``; the params are the injected truth.
    """
    from .transit import LightCurve, TransitParams, transit_model

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, span_days, cadence_min / 1440.0)
    truth = TransitParams(t0, period, rp_rs, a_rs, b)
    trend = 1.0 + trend_amp * np.sin(2 * np.pi * t / 6.5 + 0.7)
    flux = transit_model(t, truth) * trend + rng.normal(0.0, noise_ppm * 1e-6, t.shape)
    return LightCurve(t, flux, None, {"source": "mock-lightcurve"}), truth


def make_mock_spectral_lightcurves(
    engine,
    truth,
    n_times: int = 400,
    n_channels: int = 400,
    atmosphere_strength: float = 0.15,
    noise_ppm: float = 500.0,
    multipliers=None,
    seed: Optional[int] = 5,
    span_durations: float = 1.2,
):
    """
    Spectral light curves whose per-channel depth carries the engine's own atmosphere:
    ``depth(lambda) = k^2 * (1 + strength * (1 - T_engine(lambda)))``.  Closing the loop
    forward model -> photometry -> extraction -> inverse fit.  Returns
    ``(time, flux (Nt, Nw), wavelength_um)``.
    """
    from .transit import transit_model

    rng = np.random.default_rng(seed)
    dur = truth.duration_days
    t = np.linspace(truth.t0 - span_durations * dur, truth.t0 + span_durations * dur, n_times)   # around one transit
    nu = np.linspace(450.0, 4100.0, n_channels)
    m = engine.default_multipliers() if multipliers is None else np.asarray(multipliers, float)
    trans = np.exp(-(engine.tau_on(nu) @ m))
    k_lambda = truth.rp_rs * np.sqrt(1.0 + atmosphere_strength * (1.0 - trans))  # (Nw,)
    s = (1.0 - transit_model(t, truth)) / truth.rp_rs**2                         # (Nt,) shape
    flux = 1.0 - s[:, None] * (k_lambda**2)[None, :]
    flux = flux + rng.normal(0.0, noise_ppm * 1e-6, flux.shape)
    return t, flux, 1.0e4 / nu


def make_mock_spectral_timeseries(engine, truth, n_times: int = 360, n_channels: int = 400, multipliers=None,
                                  seed: Optional[int] = 5):
    """A :class:`~app.core.timeseries.SpectralTimeSeries` of one transit through the engine's atmosphere."""
    from .timeseries import SpectralTimeSeries

    t, flux, wl = make_mock_spectral_lightcurves(engine, truth, n_times=n_times, n_channels=n_channels,
                                                 multipliers=multipliers, seed=seed, span_durations=3.0)
    return SpectralTimeSeries(t, wl, flux, None, {"source": "MOCK spectral time series", "time_offset": 0.0,
                                                  "time_kind": "mock days"})
