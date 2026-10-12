"""
ECAS inverse engine - Qt-free.

Pipeline
--------
1. :func:`load_spectrum_csv` / :func:`build_observation`   any table -> :class:`ObservedSpectrum`
2. :func:`find_features`          model-free peak finding on the data (what is in this spectrum?)
3. :func:`fit_multipliers`        bounded non-linear least squares on the gas multipliers,
                                  optional fitted continuum, optional fit window
4. :func:`find_anomalies`         k-sigma residual excursions -> :class:`Anomaly` regions
5. :class:`CandidateLibrary`      identification against engine bands + a band table

The forward model is the engine's Beer-Lambert mixture evaluated *exactly* at the
observation's x-values.  Inside the optimiser everything is one ``(K, G) @ (G,)`` mat-vec
plus ``exp`` with an analytic Jacobian.

Observations are *transmittance* with the continuum at 1.0.  Import converts percent,
absorbance, optical depth, or arbitrary flux scales (continuum normalisation) and records
what it did in ``meta["import_notes"]``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import percentile_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths

from .engine import SpectrumEngine, Unit
from .tables import RawTable, TableError, read_table, suggest_spectrum_columns

FloatArray = NDArray[np.float64]
_MAD_TO_SIGMA = 1.4826


class FitError(ValueError):
    """Raised for unusable observations or degenerate fits."""


# =============================================================== import conversions
class XUnit(str, Enum):
    """How the x column of a file is expressed."""

    AUTO = "auto"
    WAVENUMBER = "cm⁻¹"
    MICRON = "µm"
    NANOMETER = "nm"
    ANGSTROM = "Å"


class YMode(str, Enum):
    """How the y column of a file should be turned into transmittance (continuum = 1)."""

    AUTO = "auto"
    TRANSMITTANCE = "transmittance (0-1)"
    PERCENT = "percent (0-100)"
    ABSORBANCE = "absorbance (log10)"
    OPTICAL_DEPTH = "optical depth (natural log)"
    NORMALIZE = "normalise to continuum (÷95th percentile)"


def _detect_xunit(x: FloatArray, text: str) -> Tuple[XUnit, str]:
    t = text.lower()
    if re.search(r"angstr|ångstr|å\b|\[a\]|\(a\)", t):
        return XUnit.ANGSTROM, "header says Ångström"
    if re.search(r"(?<![a-z])nm(?![a-z])|nanomet", t):
        return XUnit.NANOMETER, "header says nm"
    if re.search(r"(?<![a-z])(um|µm|μm)(?![a-z])|micron|microm", t):
        return XUnit.MICRON, "header says µm"
    if re.search(r"cm\^?[-−⁻]?\s*1|cm-1|cm⁻¹|wavenumber|1/cm|wn\b", t):
        return XUnit.WAVENUMBER, "header says cm⁻¹"
    hi, med = float(np.nanmax(x)), float(np.nanmedian(x))
    if hi <= 60.0:
        return XUnit.MICRON, f"no unit in header; values 0-{hi:.0f} look like µm"
    if med > 5000.0:
        return XUnit.NANOMETER, "no unit in header; values >5000 look like nm"
    return XUnit.WAVENUMBER, "no unit in header; assumed cm⁻¹ (override if wrong)"


def _to_native(x: FloatArray, unit: XUnit) -> Tuple[FloatArray, Unit]:
    if unit is XUnit.WAVENUMBER:
        return x, Unit.WAVENUMBER
    if unit is XUnit.MICRON:
        return x, Unit.WAVELENGTH
    if unit is XUnit.NANOMETER:
        return x / 1.0e3, Unit.WAVELENGTH
    return x / 1.0e4, Unit.WAVELENGTH                      # Ångström


def _convert_y(
    y: FloatArray, sigma: Optional[FloatArray], mode: YMode, header: str
) -> Tuple[FloatArray, Optional[FloatArray], YMode, str]:
    """Return ``(transmittance, sigma, resolved_mode, note)``."""
    p95 = float(np.nanpercentile(y, 95))
    note = ""
    if mode is YMode.AUTO:
        h = header.lower()
        if re.search(r"absorbance|\babs\b", h):
            mode = YMode.ABSORBANCE
        elif re.search(r"optical depth|\btau\b", h):
            mode = YMode.OPTICAL_DEPTH
        elif re.search(r"%|percent", h) and p95 > 1.5:
            mode = YMode.PERCENT
        elif 0.5 <= p95 <= 1.2:
            mode = YMode.TRANSMITTANCE
        elif 50.0 < p95 <= 105.0:
            mode = YMode.PERCENT
        else:
            mode = YMode.NORMALIZE
        note = f"y mode auto-detected: {mode.value}"
    s = None if sigma is None else np.asarray(sigma, dtype=np.float64)
    if mode is YMode.TRANSMITTANCE:
        return y, s, mode, note
    if mode is YMode.PERCENT:
        return y / 100.0, None if s is None else s / 100.0, mode, note
    if mode is YMode.ABSORBANCE:
        t = 10.0 ** (-y)
        return t, None if s is None else s * t * np.log(10.0), mode, note
    if mode is YMode.OPTICAL_DEPTH:
        t = np.exp(-y)
        return t, None if s is None else s * t, mode, note
    scale = p95
    if not scale > 0:
        raise FitError("cannot normalise: the 95th-percentile y value is not positive")
    note = (note + "; " if note else "") + f"divided by {scale:.4g} (95th percentile) so the continuum is ~1"
    return y / scale, None if s is None else s / scale, YMode.NORMALIZE, note


# ================================================================== observation
@dataclass
class ObservedSpectrum:
    """Transmittance ``y`` (continuum = 1, optional 1-sigma ``sigma``) versus ``x`` in ``unit``."""

    x: FloatArray
    y: FloatArray
    unit: Unit
    sigma: Optional[FloatArray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=np.float64)
        ok = np.isfinite(self.x) & np.isfinite(self.y) & (self.x > 0)
        if self.sigma is not None:
            ok &= np.isfinite(self.sigma) & (self.sigma > 0)
        if ok.sum() < 8:
            raise FitError("observation needs at least 8 finite points with x > 0")
        order = np.argsort(self.x[ok], kind="stable")
        self.x, self.y = self.x[ok][order], self.y[ok][order]
        if self.sigma is not None:
            self.sigma = self.sigma[ok][order]

    @property
    def wavenumber_cm1(self) -> FloatArray:
        return self.x if self.unit is Unit.WAVENUMBER else 1.0e4 / self.x

    def subset(self, mask: NDArray[np.bool_]) -> "ObservedSpectrum":
        """New observation holding only ``mask`` points (same unit/meta)."""
        return ObservedSpectrum(self.x[mask], self.y[mask], self.unit,
                                None if self.sigma is None else self.sigma[mask], dict(self.meta))

    def x_in(self, unit: Unit) -> FloatArray:
        """x values expressed on the ``unit`` axis (cm⁻¹ <-> µm conversion is 1e4/x)."""
        return self.x if unit is self.unit else 1.0e4 / self.x


def build_observation(
    table: RawTable,
    x_col: int,
    y_col: int,
    sigma_col: Optional[int] = None,
    x_unit: XUnit = XUnit.AUTO,
    y_mode: YMode = YMode.AUTO,
    name: Optional[str] = None,
) -> ObservedSpectrum:
    """Interpret three columns of ``table`` as an observation, recording every assumption."""
    x = table.column(x_col).astype(np.float64)
    y = table.column(y_col).astype(np.float64)
    sigma = None if sigma_col is None else table.column(sigma_col).astype(np.float64)
    notes = list(table.notes)

    hint = f"{table.names[x_col]} {table.header}"
    if x_unit is XUnit.AUTO:
        x_unit, why = _detect_xunit(x, hint)
        notes.append(f"x unit auto-detected: {x_unit.value} ({why})")
    x_native, unit = _to_native(x, x_unit)
    y_t, s_t, resolved, ynote = _convert_y(y, sigma, y_mode, f"{table.names[y_col]} {table.header}")
    if ynote:
        notes.append(ynote)
    if float(np.nanmax(y_t)) > 1.6 or float(np.nanmin(y_t)) < -0.2:
        notes.append("warning: transmittance values fall far outside 0-1; check the y mode")
    density_match = re.match(r"^\s*F(LAM|NU)(?:$|[\s\[])", table.names[y_col], re.I)
    spectrum_kind = "emission_flux" if density_match else "normalized_spectrum"
    density_kind = f"f{density_match.group(1).lower()}" if density_match else None
    if density_kind:
        notes.append(
            f"archive {table.names[y_col]} flux normalized for relative emission-spectrum analysis; "
            "not a transmission spectrum")
    return ObservedSpectrum(
        x_native, y_t, unit, s_t,
        meta={"source": table.source, "name": name or Path(table.source).name or "observation",
              "x_unit": x_unit.value, "y_mode": resolved.value, "import_notes": notes,
              "spectrum_kind": spectrum_kind, "flux_density_kind": density_kind,
              "columns": [table.names[x_col], table.names[y_col],
                          None if sigma_col is None else table.names[sigma_col]]})


def load_spectrum_csv(path: Union[str, Path], unit: Optional[Unit] = None) -> ObservedSpectrum:
    """
    Read a spectrum file with automatic column/unit/scale detection (see :mod:`tables`).
    ``unit`` optionally forces cm⁻¹ (``Unit.WAVENUMBER``) or µm (``Unit.WAVELENGTH``).
    """
    try:
        table = read_table(path)
    except TableError as exc:
        raise FitError(str(exc)) from exc
    cols = suggest_spectrum_columns(table)
    forced = {None: XUnit.AUTO, Unit.WAVENUMBER: XUnit.WAVENUMBER, Unit.WAVELENGTH: XUnit.MICRON}[unit]
    return build_observation(table, cols["x"], cols["y"], cols["sigma"], forced)


# ========================================================================= fit
@dataclass
class FitResult:
    """Outcome of :func:`fit_multipliers`; arrays refer to ``used`` (the points actually fitted)."""

    multipliers: FloatArray                 # (G,)
    uncertainties: FloatArray               # (G,) 1-sigma (0 for fixed gases)
    model: FloatArray                       # (K,) best-fit model incl. continuum
    residual: FloatArray                    # (K,) observed - model
    sigma: FloatArray                       # (K,)
    chi2: float
    reduced_chi2: float
    success: bool
    message: str
    n_evaluations: int
    sigma_estimated: bool
    used: ObservedSpectrum                  # the (possibly windowed) observation that was fitted
    baseline_coeffs: FloatArray = field(default_factory=lambda: np.zeros(0))   # (c0[, c1])
    normalised_y: Optional[FloatArray] = None   # used.y / continuum (== used.y if no baseline)
    n_excluded: int = 0                     # points dropped by the fit window


def _baseline_basis(nu: FloatArray, n_terms: int) -> FloatArray:
    """(K, n_terms) basis [1, u] with u in [-1, 1] across the fitted range."""
    span = max(float(nu.max() - nu.min()), 1e-9)
    u = 2.0 * (nu - nu.min()) / span - 1.0
    return np.column_stack([np.ones_like(u), u])[:, :n_terms]


def fit_multipliers(
    engine: SpectrumEngine,
    obs: ObservedSpectrum,
    x0: Optional[FloatArray] = None,
    free: Optional[Sequence[bool]] = None,
    baseline: int = 0,
    window_cm1: Optional[Tuple[float, float]] = None,
) -> FitResult:
    """
    Fit the per-gas multipliers by bounded trust-region least squares (``trf``).

    Parameters
    ----------
    baseline   0: continuum fixed at 1.  1: fit a continuum level.  2: level + linear tilt.
    window_cm1 Only fit points with ``lo <= nu <= hi`` (default: all).  Use this to keep the
               fit inside the spectral range the model actually covers; bands outside it
               (overtones, electronic transitions) are not modelled and would otherwise show
               up as spurious "unidentified" residuals.
    """
    g = engine.n_gases
    x0 = engine.default_multipliers() if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    free_mask = np.ones(g, bool) if free is None else np.asarray(free, dtype=bool)
    if free_mask.shape != (g,) or not free_mask.any():
        raise FitError("`free` must be a length-G boolean mask with at least one True")
    if baseline not in (0, 1, 2):
        raise FitError("baseline must be 0, 1 or 2")
    upper = np.array([gs.multiplier_max for gs in engine.gases])

    mask = np.ones(obs.x.size, bool)
    if window_cm1 is not None:
        nu_all = obs.wavenumber_cm1
        mask = (nu_all >= window_cm1[0]) & (nu_all <= window_cm1[1])
        if mask.sum() < 8:
            raise FitError(f"only {int(mask.sum())} data points lie inside the fit window "
                           f"{window_cm1[0]:.0f}-{window_cm1[1]:.0f} cm⁻¹; widen the window or "
                           f"untick 'restrict fit to model range'")
    used = obs if mask.all() else obs.subset(mask)
    nu = used.wavenumber_cm1
    y = used.y
    sigma = np.ones_like(y) if used.sigma is None else used.sigma

    tau = engine.tau_on(nu)                                       # (K, G)
    tau_free = tau[:, free_mask]
    base = tau[:, ~free_mask] @ x0[~free_mask]
    n_free = int(free_mask.sum())
    basis = _baseline_basis(nu, baseline) if baseline else np.zeros((y.size, 0))

    def parts(p):
        e = np.exp(-(base + tau_free @ p[:n_free]))
        b = basis @ p[n_free:] if baseline else 1.0
        return e, b

    def run(sig):
        def resid(p):
            e, b = parts(p)
            return (b * e - y) / sig
        def jac(p):
            e, b = parts(p)
            d_m = (-(b * e))[:, None] * tau_free
            d_b = basis * e[:, None]
            return np.column_stack([d_m, d_b]) / sig[:, None]
        p0 = np.concatenate([np.clip(x0[free_mask], 0, upper[free_mask]),
                             [1.0, 0.0][:baseline]])
        lo = np.concatenate([np.zeros(n_free), [0.0, -1.0][:baseline]])
        hi = np.concatenate([upper[free_mask], [100.0, 1.0][:baseline]])
        return least_squares(resid, p0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac")

    sol = run(sigma)
    est = used.sigma is None
    if est:                                                       # unit-less noise: MAD of residual
        e, b = parts(sol.x)
        r = y - b * e
        s = _MAD_TO_SIGMA * np.median(np.abs(r - np.median(r)))
        sigma = np.full_like(y, max(float(s), 1e-9))

    m = x0.copy()
    m[free_mask] = sol.x[:n_free]
    coeffs = sol.x[n_free:].copy()
    b_full = basis @ coeffs if baseline else np.ones_like(y)
    model = b_full * np.exp(-(tau @ m))
    resid = y - model
    chi2 = float(np.sum((resid / sigma) ** 2))
    dof = max(y.size - sol.x.size, 1)

    cov = np.linalg.pinv(sol.jac.T @ sol.jac * (1.0 / np.square(sigma).mean() if est else 1.0))
    unc = np.zeros(g)
    unc[free_mask] = np.sqrt(np.clip(np.diag(cov)[:n_free], 0, None))
    return FitResult(m, unc, model, resid, sigma, chi2, chi2 / dof, bool(sol.success),
                     str(sol.message), int(sol.nfev), est, used, coeffs,
                     y / np.maximum(b_full, 1e-9), int((~mask).sum()))


# ============================================================= anomalies / features
@dataclass(frozen=True)
class Anomaly:
    """A localised spectral feature: residual excursion (fit) or absorption peak (data-only)."""

    nu_peak_cm1: float
    nu_lo_cm1: float
    nu_hi_cm1: float
    significance_sigma: float          # peak |residual| / sigma  (or peak SNR for features)
    peak_residual: float               # signed observed - model at the peak (negative = absorption)
    n_points: int

    @property
    def extra_absorption(self) -> bool:
        """True when the data lie *below* the model (an unmodelled absorber)."""
        return self.peak_residual < 0


def find_anomalies(
    obs: ObservedSpectrum, fit: FitResult, threshold_sigma: float = 3.0, min_points: int = 3
) -> List[Anomaly]:
    """Contiguous runs of ``|residual|/sigma > threshold`` on the points that were fitted."""
    z = fit.residual / fit.sigma
    sig = (np.abs(z) > threshold_sigma).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], sig, [0]))))
    starts, ends = edges[::2], edges[1::2]
    nu = fit.used.wavenumber_cm1
    out: List[Anomaly] = []
    for a, b in zip(starts, ends):
        if b - a < min_points:
            continue
        k = a + int(np.argmax(np.abs(z[a:b])))
        out.append(Anomaly(float(nu[k]), float(nu[a:b].min()), float(nu[a:b].max()),
                           float(abs(z[k])), float(fit.residual[k]), int(b - a)))
    return sorted(out, key=lambda an: -an.significance_sigma)


def find_features(obs: ObservedSpectrum, min_snr: float = 5.0, max_features: int = 25) -> List[Anomaly]:
    """
    Model-free absorption-peak finder: no fit and no gas list needed.

    The local continuum is a rolling 90th percentile; absorption is ``1 - y/continuum``;
    noise comes from point-to-point scatter (MAD of the first difference / sqrt 2); peaks
    need prominence >= ``min_snr`` x noise.  Returned as :class:`Anomaly` (``significance``
    is the peak SNR, ``peak_residual`` is minus the fractional depth).
    """
    n = obs.y.size
    if n < 30:
        raise FitError("need at least 30 points for peak finding")
    win = int(np.clip(n // 8, 15, 1001)) | 1
    cont = np.maximum(percentile_filter(obs.y, 90, size=win, mode="nearest"), 1e-12)
    a = 1.0 - obs.y / cont
    d = np.diff(a)
    noise = max(_MAD_TO_SIGMA * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0), 1e-9)
    peaks, props = find_peaks(a, prominence=min_snr * noise, width=2)
    # Prominence alone passes the *global* noise maximum (peak +3 sigma above a -3 sigma trough),
    # so also require the peak itself to stand well above the typical level.
    keep = (a[peaks] - np.median(a)) >= 0.75 * min_snr * noise
    peaks, props = peaks[keep], {k: v[keep] for k, v in props.items()}
    if peaks.size == 0:
        return []
    _, _, left, right = peak_widths(a, peaks, rel_height=0.5)
    idx = np.arange(n, dtype=np.float64)
    nu = obs.wavenumber_cm1
    nu_l, nu_r = np.interp(left, idx, nu), np.interp(right, idx, nu)
    out = [Anomaly(float(nu[p]), float(min(l, r)), float(max(l, r)),
                   float(props["prominences"][i] / noise), float(-a[p]),
                   max(int(round(right[i] - left[i])), 1))
           for i, (p, l, r) in enumerate(zip(peaks, nu_l, nu_r))]
    return sorted(out, key=lambda f: -f.significance_sigma)[:max_features]


# ================================================================= identification
@dataclass(frozen=True)
class Candidate:
    molecule: str
    name: str
    score: float                       # strength-weighted fraction of bands matched, 0..1
    matched_bands_cm1: Tuple[float, ...]


@dataclass(frozen=True)
class BandMatch:
    """One molecule band lying near a detected feature."""

    molecule: str
    band_cm1: float
    distance_cm1: float
    weight: float


class CandidateLibrary:
    """Band-centre look-up table; ranks molecules against feature positions."""

    DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "candidate_bands.json"

    def __init__(self, table: Dict[str, Dict[str, Any]]):
        self.molecules = tuple(table)
        self.names = {k: v.get("name", k) for k, v in table.items()}
        nu, w, idx = [], [], []
        for i, key in enumerate(self.molecules):
            for centre, weight in table[key]["bands"]:
                nu.append(centre); w.append(weight); idx.append(i)
        self._nu, self._w = np.array(nu, float), np.array(w, float)
        self._idx = np.array(idx, dtype=np.int64)
        self._wsum = np.bincount(self._idx, weights=self._w, minlength=len(self.molecules))

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "CandidateLibrary":
        raw = json.loads(Path(path or cls.DEFAULT_PATH).read_text(encoding="utf-8"))
        return cls(raw["molecules"])

    @classmethod
    def from_engine(cls, engine: SpectrumEngine, path: Optional[Union[str, Path]] = None) -> "CandidateLibrary":
        """The engine's own gases (their ORCA bands) merged with the JSON table."""
        try:
            table = json.loads(Path(path or cls.DEFAULT_PATH).read_text(encoding="utf-8"))["molecules"]
        except (OSError, ValueError, KeyError):
            table = {}
        for gas in engine.gases:
            sl = engine.line_slices[gas.gas_id]
            nu, inten = engine._line_nu[sl], engine._line_int[sl]
            if nu.size:
                table[gas.gas_id] = {"name": gas.name,
                                     "bands": [[float(a), float(b / inten.max())] for a, b in zip(nu, inten)]}
        return cls(table)

    def rank(self, anomaly_nu_cm1: Sequence[float], tolerance_cm1: float = 25.0,
             exclude: Sequence[str] = (), top_k: int = 5) -> List[Candidate]:
        """Score = sum(w_b [band within tolerance of any feature]) / sum(w_b), all bands at once."""
        a = np.asarray(anomaly_nu_cm1, dtype=np.float64)
        if a.size == 0:
            return []
        hit = (np.abs(a[:, None] - self._nu[None, :]) <= tolerance_cm1).any(axis=0)
        score = np.bincount(self._idx, weights=self._w * hit, minlength=len(self.molecules)) / self._wsum
        out = []
        for i in np.argsort(-score, kind="stable"):
            key = self.molecules[i]
            if score[i] <= 0 or key in exclude:
                continue
            sel = (self._idx == i) & hit
            out.append(Candidate(key, self.names[key], float(score[i]), tuple(self._nu[sel])))
            if len(out) == top_k:
                break
        return out

    def match_features(self, feature_nu_cm1: Sequence[float], tolerance_cm1: float = 25.0,
                       top: int = 3) -> List[List[BandMatch]]:
        """For each feature, the nearest/strongest library bands (an ``(F, B)`` distance matrix)."""
        f = np.asarray(feature_nu_cm1, dtype=np.float64)
        dist = np.abs(f[:, None] - self._nu[None, :])
        score = self._w[None, :] * np.clip(1.0 - dist / tolerance_cm1, 0.0, None)
        out: List[List[BandMatch]] = []
        for i in range(f.size):                               # post-processing: few features
            cand = np.flatnonzero(score[i] > 0)
            cand = cand[np.argsort(-score[i, cand])][: top * 2]
            seen, matches = set(), []
            for j in cand:
                mol = self.molecules[self._idx[j]]
                if mol in seen:
                    continue
                seen.add(mol)
                matches.append(BandMatch(mol, float(self._nu[j]), float(dist[i, j]), float(self._w[j])))
                if len(matches) == top:
                    break
            out.append(matches)
        return out
