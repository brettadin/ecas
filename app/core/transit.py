"""
ECAS observational engine (Phase 5) - transit photometry.  Qt-free.

Contents
--------
* :class:`LightCurve`, :func:`load_lightcurve_csv`
* :func:`transit_model`            geometric transit with quadratic limb darkening
* :func:`analyze_lightcurve`       detrend -> BLS period search -> least-squares refit
                                   -> derived planet properties (:class:`TransitResult`)
* :func:`orbit_positions`          3-D planet position for the viewport / sky view
* :func:`extract_transmission_spectrum`   per-wavelength depths from spectral light curves
* :func:`depth_spectrum_to_observation`   bridge into the Phase 4 inverse fitter

Model approximations (documented, not hidden)
---------------------------------------------
Circular orbit.  Covered area is the *exact* uniform-disc overlap; limb darkening is
applied as a weight evaluated at the planet-centre position (accurate for Rp/R* <~ 0.1).
Stellar radius/mass/Teff come from the user (:class:`StellarParams`); the light curve
alone cannot give absolute sizes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]

R_SUN_KM = 695_700.0
R_EARTH_KM = 6_371.0
R_JUP_KM = 71_492.0
AU_KM = 149_597_870.7
G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
R_SUN_M = 6.957e8
DAY_S = 86_400.0
T_SUN_K = 5772.0


class TransitError(ValueError):
    """Raised for unusable light curves or failed transit searches."""


# ------------------------------------------------------------------- data types
@dataclass
class LightCurve:
    """Flux time-series: ``time`` in days, ``flux`` (any normalisation), optional errors."""

    time: FloatArray
    flux: FloatArray
    flux_err: Optional[FloatArray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64)
        self.flux = np.asarray(self.flux, dtype=np.float64)
        if self.flux_err is not None:
            self.flux_err = np.asarray(self.flux_err, dtype=np.float64)
        ok = np.isfinite(self.time) & np.isfinite(self.flux)
        if self.flux_err is not None:
            ok &= np.isfinite(self.flux_err) & (self.flux_err > 0)
        if ok.sum() < 50:
            raise TransitError("light curve needs at least 50 finite points")
        order = np.argsort(self.time[ok], kind="stable")
        self.time, self.flux = self.time[ok][order], self.flux[ok][order]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[ok][order]
        keep = np.concatenate(([True], np.diff(self.time) > 0))      # strictly increasing times
        if not keep.all():
            self.time, self.flux = self.time[keep], self.flux[keep]
            if self.flux_err is not None:
                self.flux_err = self.flux_err[keep]
        if self.time.size < 50:
            raise TransitError("light curve needs at least 50 points with distinct times")


@dataclass(frozen=True)
class StellarParams:
    """Host-star inputs needed to turn ratios into physical units."""

    radius_rsun: float = 1.0
    mass_msun: float = 1.0
    teff_k: float = T_SUN_K


@dataclass(frozen=True)
class TransitParams:
    """Circular-orbit transit geometry (all lengths in stellar radii)."""

    t0: float                  # mid-transit time (days)
    period: float              # days
    rp_rs: float               # planet/star radius ratio k
    a_rs: float                # semi-major axis / R*
    b: float = 0.0             # impact parameter
    u1: float = 0.3            # quadratic limb darkening
    u2: float = 0.2

    @property
    def inclination_deg(self) -> float:
        return float(np.degrees(np.arccos(min(self.b / self.a_rs, 1.0))))

    @property
    def duration_days(self) -> float:
        """Total (first-to-fourth contact) duration T14."""
        s = ((1 + self.rp_rs) ** 2 - self.b**2)
        if s <= 0:
            return 0.0
        sin_i = np.sqrt(1.0 - (self.b / self.a_rs) ** 2)
        return float(self.period / np.pi * np.arcsin(min(np.sqrt(s) / (self.a_rs * sin_i), 1.0)))


# ------------------------------------------------------------------ transit model
def _sky_separation(t: FloatArray, p: TransitParams) -> Tuple[FloatArray, FloatArray]:
    """Sky-projected separation z (stellar radii) and cos(phase); vectorised over ``t``."""
    phi = 2.0 * np.pi * (np.asarray(t, dtype=np.float64) - p.t0) / p.period
    cos_i = p.b / p.a_rs
    z = p.a_rs * np.sqrt(np.sin(phi) ** 2 + (cos_i * np.cos(phi)) ** 2)
    return z, np.cos(phi)


def _overlap_area(z: FloatArray, k: float) -> FloatArray:
    """Exact area of the planet disc (radius k) lying over the unit stellar disc."""
    z = np.asarray(z, dtype=np.float64)
    full = np.pi * k**2
    zc = np.maximum(z, 1e-12)
    arg1 = np.clip((zc**2 + k**2 - 1.0) / (2.0 * zc * k), -1.0, 1.0)
    arg2 = np.clip((zc**2 + 1.0 - k**2) / (2.0 * zc), -1.0, 1.0)
    root = np.clip((-zc + k + 1) * (zc + k - 1) * (zc - k + 1) * (zc + k + 1), 0.0, None)
    lens = k**2 * np.arccos(arg1) + np.arccos(arg2) - 0.5 * np.sqrt(root)
    return np.where(z >= 1.0 + k, 0.0, np.where(z <= 1.0 - k, full, lens))


def transit_model(t: FloatArray, p: TransitParams) -> FloatArray:
    """Normalised flux (1.0 out of transit) for times ``t``; fully vectorised."""
    z, cos_phi = _sky_separation(t, p)
    area = np.where(cos_phi > 0.0, _overlap_area(z, p.rp_rs), 0.0)       # only the near side transits
    mu = np.sqrt(np.clip(1.0 - np.minimum(z, 1.0) ** 2, 0.0, 1.0))
    intensity = 1.0 - p.u1 * (1.0 - mu) - p.u2 * (1.0 - mu) ** 2
    mean_intensity = 1.0 - p.u1 / 3.0 - p.u2 / 6.0
    return 1.0 - (area / np.pi) * intensity / mean_intensity


def orbit_positions(t: FloatArray, p: TransitParams) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Planet position in stellar radii: ``X`` along the orbit on the sky, ``Y`` across it,
    ``Z`` towards the observer (Z > 0: planet in front of the star).
    """
    phi = 2.0 * np.pi * (np.asarray(t, dtype=np.float64) - p.t0) / p.period
    cos_i = p.b / p.a_rs
    sin_i = np.sqrt(1.0 - cos_i**2)
    return p.a_rs * np.sin(phi), p.a_rs * cos_i * np.cos(phi), p.a_rs * sin_i * np.cos(phi)


_SPECTRUM_HINT = re.compile(
    r"wavelength|wavenumber|(?<![a-z])(nm|um|µm)(?![a-z])|micron|cm\^?-1|cm⁻¹|frequency|transmittance", re.I)


def load_lightcurve_csv(path: Union[str, Path]) -> LightCurve:
    """
    Read ``time, flux[, err]`` from CSV/TSV/text or a FITS light curve (TESS/Kepler style,
    PDCSAP flux preferred).  A file whose headers say wavelength/wavenumber/transmittance
    is rejected with a pointer to the Spectroscopy tab instead of being mis-analysed.
    """
    from .tables import TableError, read_table, suggest_lightcurve_columns

    try:
        table = read_table(path)
    except TableError as exc:
        raise TransitError(str(exc)) from exc
    cols = suggest_lightcurve_columns(table)
    label = f"{table.names[cols['t']]} {table.names[cols['y']]} {table.header}"
    hint = _SPECTRUM_HINT.search(label)
    if hint:
        raise TransitError(f"This looks like a spectrum (its header mentions '{hint.group(0)}'), "
                           "not a light curve. Load it on the Spectroscopy tab instead.")
    return LightCurve(table.column(cols["t"]), table.column(cols["y"]),
                      None if cols["sigma"] is None else table.column(cols["sigma"]),
                      {"source": str(path), "columns": [table.names[cols["t"]], table.names[cols["y"]]]})


# ----------------------------------------------------------------------- analysis
@dataclass
class TransitResult:
    """Everything :func:`analyze_lightcurve` learns about the system."""

    params: TransitParams
    param_errors: Dict[str, float]
    depth: float                       # (Rp/R*)^2
    duration_hours: float
    bls_period: float
    bls_snr: float
    rp_rearth: float
    rp_rjup: float
    a_au: float                        # a_rs * R*
    a_au_kepler: float                 # from Kepler's 3rd law with M*
    orbital_speed_kms: float
    teq_k: float
    stellar_density_solar: float
    inclination_deg: float
    time: FloatArray                   # detrended light curve actually fitted
    flux: FloatArray
    model: FloatArray
    chi2_reduced: float
    n_transits: int
    notes: List[str] = field(default_factory=list)

    def with_stellar(self, stellar: "StellarParams") -> "TransitResult":
        """Same fit, new host-star inputs: only the physical quantities are recomputed."""
        return replace(self, **_physical(self.params, stellar))

    def summary_lines(self) -> List[str]:
        p, e = self.params, self.param_errors
        return [f"Note: {n}" for n in self.notes] + [
            f"Period            {p.period:.5f} ± {e['period']:.5f} d",
            f"Mid-transit t0    {p.t0:.5f} ± {e['t0']:.5f} d",
            f"Depth (Rp/R*)²    {self.depth * 1e6:,.0f} ppm",
            f"Rp/R*             {p.rp_rs:.4f} ± {e['rp_rs']:.4f}",
            f"Rp                {self.rp_rjup:.2f} R_Jup = {self.rp_rearth:.1f} R_Earth",
            f"Duration T14      {self.duration_hours:.2f} h",
            f"a/R*              {p.a_rs:.2f} ± {e['a_rs']:.2f}",
            f"Impact b          {p.b:.2f}   inclination {self.inclination_deg:.2f}°",
            f"a (geometry)      {self.a_au:.4f} AU   (Kepler: {self.a_au_kepler:.4f} AU)",
            f"Orbital speed     {self.orbital_speed_kms:.1f} km/s",
            f"T_eq (A=0)        {self.teq_k:.0f} K",
            f"ρ* (transit)      {self.stellar_density_solar:.2f} ρ☉",
            f"Transits seen     {self.n_transits}    reduced χ² {self.chi2_reduced:.2f}",
        ]


def _physical(best: TransitParams, stellar: StellarParams) -> Dict[str, float]:
    """Convert dimensionless fit results into physical units for a given host star."""
    r_star_km = stellar.radius_rsun * R_SUN_KM
    a_km = best.a_rs * r_star_km
    p_s = best.period * DAY_S
    a_kepler_m = (G_SI * stellar.mass_msun * M_SUN_KG * p_s**2 / (4 * np.pi**2)) ** (1 / 3)
    rho_sun = 3 * np.pi * best.a_rs**3 / (G_SI * p_s**2) / (M_SUN_KG / (4 / 3 * np.pi * R_SUN_M**3))
    return dict(
        rp_rearth=best.rp_rs * r_star_km / R_EARTH_KM, rp_rjup=best.rp_rs * r_star_km / R_JUP_KM,
        a_au=a_km / AU_KM, a_au_kepler=a_kepler_m / 1e3 / AU_KM,
        orbital_speed_kms=2 * np.pi * a_km / p_s,
        teq_k=stellar.teff_k * float(np.sqrt(0.5 / best.a_rs)),
        stellar_density_solar=float(rho_sun))


def detrend(time: FloatArray, flux: FloatArray, window_days: float = 1.0) -> FloatArray:
    """Divide by a running median (window in days, converted using the median cadence)."""
    cadence = float(np.median(np.diff(time)))
    size = max(int(round(window_days / cadence)) | 1, 3)
    return flux / median_filter(flux, size=min(size, flux.size // 2 * 2 - 1), mode="nearest")


BLS_BUDGET = 1.5e8        # modelled work units (see _bls_search); ~1-2 s on a laptop
BLS_OVERSAMPLE = 5        # phase-fold bins per shortest duration
MAX_SPAN_DAYS = 20_000.0  # beyond this the "time" axis is not plausibly a light curve


def _bin_uniform(t, f, e, max_points: int):
    """Average consecutive samples so at most ``max_points`` remain (search stage only)."""
    nb = int(np.ceil(t.size / max_points))
    if nb <= 1:
        return t, f, e
    n = t.size // nb * nb
    reduce = lambda a: a[:n].reshape(-1, nb).mean(axis=1)           # noqa: E731
    return reduce(t), reduce(f), reduce(e) / np.sqrt(nb)


def _bls_search(t, f, dy, p_min, p_max, durations):
    from astropy.timeseries import BoxLeastSquares            # lazy: only needed here

    t, f, dy = _bin_uniform(t, f, dy, 40_000)
    span = t.max() - t.min()
    # Period step needed so the transit does not drift by more than a quarter-duration across the
    # data span, capped so the whole search stays within a fixed compute budget.
    n_required = np.log(p_max / p_min) / (0.25 * durations.min() / span)
    # Work per trial period ~ N (binning the data) + phase-fold bins x durations.  The phase-fold
    # bin count grows with the period itself, so use the log-uniform mean period.
    mean_period = (p_max - p_min) / np.log(p_max / p_min)
    per_period = t.size + mean_period * BLS_OVERSAMPLE / durations.min() * durations.size
    n_cap = max(int(BLS_BUDGET / per_period), 500)
    n = int(np.clip(n_required, 500, n_cap))
    periods = np.exp(np.linspace(np.log(p_min), np.log(p_max), n))
    bls = BoxLeastSquares(t, f, dy)
    res = bls.power(periods, durations, objective="snr", oversample=BLS_OVERSAMPLE)
    i = int(np.argmax(res.depth_snr))
    return (float(res.period[i]), float(res.transit_time[i]), float(res.duration[i]),
            float(res.depth[i]), float(res.depth_snr[i]))


def _single_transit_guess(t, f, err, period):
    """Starting values for a single transit: deepest smoothed point, its width, depth and SNR."""
    n = max(int(0.02 * f.size) | 1, 5)
    sm = median_filter(f, size=min(n, f.size // 2 * 2 - 1), mode="nearest")
    i = int(np.argmin(sm))
    depth = float(max(1.0 - sm[i], 1e-6))
    below = sm < 1.0 - depth / 2.0
    lo = hi = i
    while lo > 0 and below[lo - 1]:
        lo -= 1
    while hi < f.size - 1 and below[hi + 1]:
        hi += 1
    cadence = float(np.median(np.diff(t)))
    dur = float(max(t[hi] - t[lo], 3.0 * cadence))
    n_in = max(hi - lo + 1, 1)
    snr = depth / (float(np.median(err)) / np.sqrt(n_in))
    return period, float(t[i]), dur, depth, float(snr)


def _empirical_scatter(f: FloatArray) -> float:
    """Point-to-point noise of ``f`` (robust), independent of any uncertainty column."""
    d = np.diff(f)
    return max(float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)), 1e-9)


def analyze_lightcurve(
    lc: LightCurve,
    stellar: StellarParams = StellarParams(),
    limb_darkening: Tuple[float, float] = (0.3, 0.2),
    detrend_window_days: Optional[float] = 1.0,
    period_range: Optional[Tuple[float, float]] = None,
    min_snr: float = 10.0,
    known_period: Optional[float] = None,
    fit_baseline: Optional[bool] = None,
) -> TransitResult:
    """
    Detrend, find the period with Box Least Squares, then refine (t0, P, Rp/R*, a/R*, b)
    with bounded least squares on the full transit model, and derive physical quantities.

    ``known_period`` (days) is for **single-transit data** (e.g. one JWST visit), where a period
    search is impossible: the search is skipped, the period is held fixed, and only (t0, Rp/R*, a/R*,
    b) are fitted.  Take it from the archive or your own ephemeris.

    ``detrend_window_days=None`` skips detrending (use for already-flat data).  The window
    should exceed ~3x the transit duration or the dip will be partly removed.

    ``fit_baseline`` adds a linear time trend multiplying the transit model. It defaults to on for
    single-transit analysis (``known_period`` given), where slow drifts are the usual systematic and
    bias the depth; off otherwise, where detrending handles the trend over many transits.
    """
    t = lc.time
    span_check = float(t[-1] - t[0])
    cadence = float(np.median(np.diff(t)))
    if span_check > MAX_SPAN_DAYS or span_check / cadence > 5.0e6:
        raise TransitError(f"the time axis spans {span_check:,.0f} d - this does not look like a light "
                           "curve (is it a spectrum?). Use the Spectroscopy tab for spectra.")
    if np.ptp(lc.flux) <= 0 or not np.median(lc.flux) > 0:
        raise TransitError("flux must be positive and not constant")
    notes: List[str] = []
    f = lc.flux / np.median(lc.flux)
    if detrend_window_days and detrend_window_days > span_check / 3.0:
        notes.append(f"detrending skipped: the {detrend_window_days:g} d window is longer than a third of the "
                     f"{span_check:.3f} d span, so it would remove the transit signal itself")
        detrend_window_days = None
    if detrend_window_days:
        f = detrend(t, f, detrend_window_days)
    err = lc.flux_err / np.median(lc.flux) if lc.flux_err is not None else None
    sigma_scatter = _empirical_scatter(f)
    if err is None:                                            # robust scatter -> uniform errors
        err = np.full_like(f, sigma_scatter)
    else:
        # The detection thresholds (BLS and single-transit SNR) scale with the uncertainty column.
        # If that column is on a different scale from the real point-to-point scatter, a transit that
        # is plainly visible gets an SNR near zero. Trust the measured scatter in that case.
        ratio = float(np.median(err)) / sigma_scatter
        if ratio > 3.0 or ratio < 1.0 / 3.0:
            notes.append(f"the uncertainty column is {ratio:.3g}x the measured point-to-point scatter "
                         f"({sigma_scatter:.2e}); the measured scatter was used instead. Check the "
                         "error column's units.")
            err = np.full_like(f, sigma_scatter)

    span = t.max() - t.min()
    if known_period is not None:
        if not known_period > 0:
            raise TransitError("known_period must be positive")
        period0, t00, dur0, depth0, snr = _single_transit_guess(t, f, err, float(known_period))
        # The overall median is a poor baseline when the transit covers a large part of a short window
        # (common for single JWST visits): it sits inside the dip and biases the depth. Re-normalise by
        # the out-of-transit level instead.
        out_of_transit = np.abs(t - t00) > 0.75 * dur0
        if out_of_transit.sum() >= 20:
            factor = float(np.median(lc.flux[out_of_transit]) / np.median(lc.flux))
            if factor > 0 and np.isfinite(factor):
                f = f / factor
                err = err / factor
                # the detection SNR must be measured on the corrected normalisation too
                period0, t00, dur0, depth0, snr = _single_transit_guess(t, f, err, float(known_period))
    else:
        p_lo, p_hi = period_range or (max(0.4, 5 * float(np.median(np.diff(t)))), min(span / 2.0, 60.0))
        if p_hi <= p_lo:
            raise TransitError("light curve too short for a period search (need >= 2 transits). For a single "
                               "transit, give the period from the archive or your ephemeris.")
        durations = np.array([0.02, 0.04, 0.06, 0.09, 0.13, 0.2, 0.3])
        durations = durations[durations < 0.5 * p_lo]
        period0, t00, dur0, depth0, snr = _bls_search(t, f, err, p_lo, p_hi, durations)
    if snr < min_snr:                       # pure noise reaches BLS SNR ~8 over thousands of trial periods
        where = (f"period {period0:.4f} d" if known_period is None else f"transit near t = {t00:.5f} d")
        raise TransitError(
            f"no significant transit found. The strongest candidate ({where}) is a dip of about "
            f"{depth0 * 1e6:,.0f} ppm lasting about {dur0 * 24:.2f} h, with SNR {snr:.1f}; the required "
            f"SNR is {min_snr:g}. A dip this shallow can be a real transit, or noise, or a drift. Look at the "
            "light-curve plot: if the dip is clearly a transit, lower 'Detection SNR' in the Photometry panel "
            "and analyze again.")

    if not (np.isfinite([period0, t00, dur0, depth0]).all() and dur0 > 0 and period0 > 0 and depth0 > 0):
        raise TransitError("the period search returned no usable transit (zero duration or depth); "
                           "the data may not be a transit light curve")
    k0 = float(np.sqrt(max(depth0, 1e-6)))
    a0 = float(np.clip(period0 / (np.pi * dur0), 2.0, 150.0))
    u1, u2 = limb_darkening

    if fit_baseline is None:
        fit_baseline = known_period is not None
    tc = 0.5 * (t.min() + t.max())
    half = max(0.5 * (t.max() - t.min()), 1e-9)
    u = (t - tc) / half                                       # time scaled to [-1, 1] over the data

    def build(x):
        return TransitParams(x[0], x[1], x[2], x[3], x[4], u1, u2)

    def full_model(x):
        m = transit_model(t, build(x))
        return m * (x[6] + x[5] * u) if fit_baseline else m

    # Single transit + known period: Kepler's third law with the host star's mass and radius gives the
    # expected a/R*. Use it as a Gaussian prior (10 % width). Without it, a/R*, b and Rp/R* trade off
    # badly for one short transit and the depth can be wrong.
    a_expected = None
    if known_period is not None and stellar.mass_msun > 0 and stellar.radius_rsun > 0:
        a_m = (G_SI * stellar.mass_msun * M_SUN_KG * (float(known_period) * DAY_S) ** 2 / (4 * np.pi**2)) ** (1 / 3)
        a_expected = a_m / (stellar.radius_rsun * R_SUN_M)
    prior_sigma = 0.1 * a_expected if a_expected else None

    def resid(x):
        r = (full_model(x) - f) / err
        if a_expected:
            r = np.concatenate([r, [(x[3] - a_expected) / prior_sigma]])
        return r

    x0 = [t00, period0, k0, a0, 0.3]
    p_tol = 1e-7 if known_period is not None else 1e-2          # fixed period for single transits
    lo = [t00 - dur0, period0 * (1 - p_tol), 0.002, 1.5, 0.0]
    hi = [t00 + dur0, period0 * (1 + p_tol), 0.5, 300.0, 1.0 + k0]
    xs = [dur0, max(period0 * p_tol * 0.1, 1e-9), 0.01, 1.0, 0.1]
    names = ["t0", "period", "rp_rs", "a_rs", "b"]
    if fit_baseline:
        x0 += [0.0, 1.0]; lo += [-0.5, 0.9]; hi += [0.5, 1.1]; xs += [0.01, 0.01]
        names += ["baseline_slope", "baseline_level"]
    # The shape parameters (a/R*, b) are correlated, so one starting point can settle in a poor local
    # minimum. Fit from several starting points and keep the best (lowest chi-square).
    starts = [(a0, 0.3)]
    if a_expected:
        starts += [(a_expected, 0.2), (a_expected, 0.5)]
    starts += [(a0 * 0.8, 0.5), (a0 * 1.2, 0.1)]
    best = None
    failure = None
    for a_start, b_start in starts:
        x_start = list(x0)
        x_start[3] = float(np.clip(a_start, 1.6, 299.0))
        x_start[4] = float(np.clip(b_start, 0.01, 0.99 + k0))
        try:
            trial = least_squares(resid, np.array(x_start), bounds=(lo, hi), x_scale=xs,
                                  xtol=1e-10, ftol=1e-10, gtol=1e-10)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failure = exc
            continue
        if best is None or trial.cost < best.cost:
            best = trial
    if best is None:
        raise TransitError(f"the transit model fit failed ({failure}); the data may not be a light curve")
    sol = best
    best = build(sol.x)
    model = full_model(sol.x)
    pure = transit_model(t, best)                             # the transit alone, for counting in-transit points
    dof = max(f.size - len(x0), 1)
    red = float(np.sum(((f - model) / err) ** 2) / dof)
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * max(red, 1.0)
    sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    errs = dict(zip(names, map(float, sig)))
    if a_expected:
        notes.append(f"a/R* prior from the host star (expected {a_expected:.1f} from M*, R*, P; 10 % width): "
                     "a/R*, b and Rp/R* are constrained by the star, not by the transit shape alone")
    if fit_baseline:
        notes.append(f"linear baseline fitted (level {sol.x[6]:.5f}, slope {sol.x[5]:+.4f} per half-span of "
                     "data): slow drifts are modelled, not mistaken for the transit")

    n_in = int(np.count_nonzero(pure < 1.0 - 1e-9))
    if n_in < 5:
        raise TransitError(f"only {n_in} data points fall inside the fitted transit - the sampling is "
                           "far too coarse for this signal to be a transit (is this a spectrum?)")
    # "Transits seen" = orbital epochs in which the DATA really dip (not merely epochs the model predicts)
    in_t = pure < 1.0 - 0.5 * best.rp_rs**2
    epoch = np.floor((t[in_t] - best.t0) / best.period + 0.5).astype(int)
    _, inv = np.unique(epoch, return_inverse=True)
    mean_deficit = np.bincount(inv, weights=1.0 - f[in_t]) / np.maximum(np.bincount(inv), 1)
    n_transits = int(np.count_nonzero(mean_deficit > 0.4 * best.rp_rs**2))
    if known_period is None and n_transits < 2:
        raise TransitError("only one transit is visible in these data, so the period cannot be determined "
                           "(single transit). Look the planet up in the archive, or give its known period.")
    return TransitResult(
        params=best, param_errors=errs, depth=best.rp_rs**2,
        duration_hours=best.duration_days * 24.0, bls_period=period0, bls_snr=snr,
        **_physical(best, stellar),
        inclination_deg=best.inclination_deg,
        time=t, flux=f, model=model, chi2_reduced=red, n_transits=n_transits, notes=notes,
    )


# ------------------------------------------------------- transmission spectroscopy
@dataclass
class DepthSpectrum:
    """Transit depth versus wavelength (the planet's transmission spectrum)."""

    wavelength_um: FloatArray
    depth: FloatArray
    depth_err: FloatArray


def extract_transmission_spectrum(
    time: FloatArray,
    flux: FloatArray,                      # (Nt, Nw) spectral light curves
    wavelength_um: FloatArray,             # (Nw,)
    params: TransitParams,
    flux_err: Optional[FloatArray] = None,
) -> DepthSpectrum:
    """
    Per-channel transit depth with the geometry held at the white-light solution.

    Each channel is normalised by its out-of-transit median, then fitted to the white-light
    transit-shape template. If per-integration flux errors are available, the fit uses their
    inverse variances and scales the formal uncertainty when residual scatter is larger.
    Otherwise uncertainty is estimated from the per-channel residual scatter. Uncertainty in
    the estimated out-of-transit baseline and time-correlated noise are not modeled.
    """
    flux = np.asarray(flux, dtype=np.float64)
    if flux.ndim != 2 or flux.shape[0] != np.size(time):
        raise TransitError("flux must have shape (n_times, n_wavelengths)")
    wavelength_um = np.asarray(wavelength_um, dtype=np.float64)
    if wavelength_um.shape != (flux.shape[1],):
        raise TransitError("wavelength_um must have one value per flux channel")
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=np.float64)
        if flux_err.shape != flux.shape:
            raise TransitError("flux_err must have the same shape as flux")
    z, cos_phi = _sky_separation(time, params)
    out_mask = (z > 1.0 + params.rp_rs) | (cos_phi <= 0.0)
    if out_mask.sum() < 5:
        raise TransitError("need at least five out-of-transit integrations to normalize spectral channels")
    baseline = np.median(flux[out_mask], axis=0)
    if not np.isfinite(baseline).all() or np.any(baseline <= 0.0):
        raise TransitError("spectral channels need finite, positive out-of-transit flux")
    norm = flux / baseline[None, :]
    s = (1.0 - transit_model(time, params)) / params.rp_rs**2        # (Nt,)
    signal = 1.0 - norm
    if flux_err is None:
        ss = float(s @ s)
        if ss <= 0.0:
            raise TransitError("transit geometry has no in-transit signal in the sampled data")
        depth = (s @ signal) / ss
        resid = signal - s[:, None] * depth[None, :]
        sigma = resid.std(axis=0, ddof=1) / np.sqrt(ss)
    else:
        sigma_norm = flux_err / baseline[None, :]
        good = np.isfinite(sigma_norm) & (sigma_norm > 0.0)
        weights = np.where(good, 1.0 / np.maximum(sigma_norm, 1e-300) ** 2, 0.0)
        denom = (weights * s[:, None] ** 2).sum(axis=0)
        if np.any(denom <= 0.0):
            raise TransitError("spectral flux uncertainties leave channels without usable transit samples")
        depth = (weights * s[:, None] * signal).sum(axis=0) / denom
        resid = signal - s[:, None] * depth[None, :]
        dof = np.maximum(good.sum(axis=0) - 1, 1)
        reduced_chi2 = (weights * resid**2).sum(axis=0) / dof
        sigma = np.sqrt(1.0 / denom) * np.sqrt(np.maximum(reduced_chi2, 1e-12))   # rescale to the data, both ways
    return DepthSpectrum(wavelength_um, depth, sigma)


def bin_depth_spectrum(spec: DepthSpectrum, bin_size: int) -> DepthSpectrum:
    """Combine adjacent wavelength channels with inverse-variance weighting."""
    if isinstance(bin_size, bool) or int(bin_size) != bin_size or bin_size < 1:
        raise TransitError("wavelength bin size must be a positive integer")
    wavelengths, depths, errors = [], [], []
    for start in range(0, spec.wavelength_um.size, int(bin_size)):
        stop = min(start + int(bin_size), spec.wavelength_um.size)
        wl = np.asarray(spec.wavelength_um[start:stop], dtype=np.float64)
        depth = np.asarray(spec.depth[start:stop], dtype=np.float64)
        error = np.asarray(spec.depth_err[start:stop], dtype=np.float64)
        good = np.isfinite(wl) & (wl > 0.0) & np.isfinite(depth) & np.isfinite(error) & (error > 0.0)
        if not good.any():
            continue
        weights = 1.0 / error[good] ** 2
        total_weight = float(weights.sum())
        wavelengths.append(float(np.average(wl[good], weights=weights)))
        depths.append(float(np.average(depth[good], weights=weights)))
        errors.append(float(np.sqrt(1.0 / total_weight)))
    if len(wavelengths) < 1:
        raise TransitError("no finite wavelength channels with positive uncertainties to bin")
    return DepthSpectrum(np.asarray(wavelengths), np.asarray(depths), np.asarray(errors))


def depth_spectrum_to_observation(spec: DepthSpectrum, strongest_absorption: float = 0.8):
    """
    HEURISTIC bridge into the Phase 4 inverse fitter (which expects transmittance in 0..1).

    The clear-atmosphere floor is the 5th-percentile depth; excess depth above it is
    treated as absorption, scaled so the strongest feature absorbs ``strongest_absorption``
    of the light:  T = 1 - (d - d_floor) / (max(d - d_floor) / strongest_absorption).
    The multipliers the fitter returns are therefore *relative* abundances.
    """
    from .engine import Unit
    from .fitting import ObservedSpectrum

    excess = np.clip(spec.depth - np.percentile(spec.depth, 5.0), 0.0, None)
    scale = max(float(excess.max()), 1e-12) / strongest_absorption
    return ObservedSpectrum(spec.wavelength_um, 1.0 - excess / scale, Unit.WAVELENGTH,
                            spec.depth_err / scale,
                            meta={"source": "transmission-spectrum", "heuristic_scale": scale})
