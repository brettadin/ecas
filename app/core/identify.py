"""
ECAS molecule identification: explain an observed spectrum as a combination of reference spectra.

Model
-----
In optical-depth space the spectrum is (to first order) a *linear* sum::

    tau_obs(nu) = - ln T_obs(nu)  ~=  sum_i  a_i * k_i(nu)  +  continuum(nu)         a_i >= 0

so identification is a bounded linear least-squares problem (``scipy.optimize.lsq_linear``), solved
exactly - no local minima, so overlapping molecules are separated as well as the data allow.

What is reported (and why)
--------------------------
* amplitude +- 1 sigma per reference (covariance of the active set),
* **drop-one** significance: how much chi^2 worsens if that molecule is removed (sqrt(delta chi^2)
  ~ the Gaussian significance of the detection) - a molecule is "detected" only above a threshold,
* **degeneracy diagnostics**: cosine similarity of reference shapes inside the fit window and the
  amplitude correlation matrix from the covariance - strongly overlapping molecules are flagged,
  because their individual amplitudes cannot be trusted even when their sum is well determined,
* **feature attribution**: for each spectral feature, the fraction of its integrated optical depth
  explained by each molecule, and the unexplained remainder.

Honesty rules: references that are not quantitative give *relative* amplitudes only (never
abundances); the linear model ignores saturation within a band, scattering/clouds and temperature
structure.  These limits are written into ``DecompositionResult.warnings``.

Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

from .fitting import Anomaly, ObservedSpectrum
from .library import ReferenceLibrary

FloatArray = NDArray[np.float64]
_MAD_TO_SIGMA = 1.4826
_T_FLOOR = 0.02                    # numerical floor for -ln T
_OPAQUE_T = 0.05                   # points below this are lower limits only and are excluded from the fit
_trapz = getattr(np, "trapezoid", None) or np.trapz


class IdentifyError(ValueError):
    """Raised when a decomposition cannot be set up."""


@dataclass
class ComponentFit:
    ref_id: str
    name: str
    formula: str
    amplitude: float
    uncertainty: float
    delta_chi2: float                 # chi^2 increase when this reference is removed
    significance: float               # sqrt(delta_chi2)
    detected: bool
    quantitative: bool
    source_key: str


@dataclass
class FeatureAttribution:
    nu_peak_cm1: float
    nu_lo_cm1: float
    nu_hi_cm1: float
    shares: Dict[str, float]          # ref_id -> fraction of the *modelled* optical depth in the feature
    unexplained: float                # fraction of the observed optical depth no reference explains
    overlapped: bool                  # two or more molecules each contribute > 20 %

    def best(self) -> Optional[str]:
        return max(self.shares, key=self.shares.get) if self.shares else None


@dataclass
class DecompositionResult:
    components: List[ComponentFit]
    chi2: float
    reduced_chi2: float
    n_points: int
    window_cm1: Tuple[float, float]
    nu: FloatArray
    tau_obs: FloatArray
    tau_model: FloatArray
    tau_by_ref: FloatArray            # (N, R) amplitude-weighted contribution of each reference
    continuum_tau: FloatArray
    sigma_tau: FloatArray
    sigma_estimated: bool
    ref_ids: List[str]
    similarity: FloatArray            # (R, R) cosine similarity of reference shapes in the window
    amp_correlation: FloatArray       # (R, R) from the fit covariance
    condition_number: float
    attributions: List[FeatureAttribution] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rms_t: float = 0.0                # RMS residual in transmittance: comparable between different fits

    def detected(self) -> List[ComponentFit]:
        return [c for c in self.components if c.detected]

    def summary_lines(self) -> List[str]:
        lines = [f"Window {self.window_cm1[0]:.0f}-{self.window_cm1[1]:.0f} cm⁻¹, {self.n_points} points, "
                 f"reduced χ² = {self.reduced_chi2:.2f}, RMS residual (T) = {self.rms_t:.4f}"]
        for c in sorted(self.components, key=lambda c: -c.significance):
            tag = "DETECTED" if c.detected else "not significant"
            kind = "" if c.quantitative else " (relative)"
            lines.append(f"{c.formula or c.name:<10} {c.amplitude:8.3g} ± {c.uncertainty:<8.2g}{kind:<11} "
                         f"{c.significance:6.1f}σ  {tag}")
        for a in self.attributions:
            who = ", ".join(f"{s:.0%} {k}" for k, s in sorted(a.shares.items(), key=lambda kv: -kv[1]) if s >= 0.05)
            lines.append(f"{a.nu_peak_cm1:7.1f} cm⁻¹ → {who or 'unassigned'}"
                         + ("  [overlap]" if a.overlapped else "")
                         + (f"  ({a.unexplained:.0%} unexplained)" if a.unexplained > 0.15 else ""))
        return lines + [f"⚠ {w}" for w in self.warnings]


def _solve(A: FloatArray, b: FloatArray, lower: FloatArray, cols: np.ndarray) -> Tuple[FloatArray, float]:
    sol = lsq_linear(A[:, cols], b, bounds=(lower[cols], np.inf), method="bvls")
    x = np.zeros(A.shape[1])
    x[cols] = sol.x
    return x, float(np.sum((A[:, cols] @ sol.x - b) ** 2))


def decompose(
    obs: ObservedSpectrum,
    library: ReferenceLibrary,
    ref_ids: Optional[Sequence[str]] = None,
    window_cm1: Optional[Tuple[float, float]] = None,
    fit_continuum: bool = True,
    detection_sigma: float = 3.0,
    features: Optional[Sequence[Anomaly]] = None,
) -> DecompositionResult:
    """Fit ``obs`` (transmittance, continuum ~ 1) as a non-negative combination of references."""
    ids = list(ref_ids) if ref_ids is not None else library.ids()
    if not ids:
        raise IdentifyError("the reference library is empty")
    refs = [library.get(i) for i in ids]
    cov_lo, cov_hi = library.coverage_cm1(ids)
    lo, hi = window_cm1 or (cov_lo, cov_hi)
    nu_all = obs.wavenumber_cm1
    mask = (nu_all >= lo) & (nu_all <= hi)
    if mask.sum() < 20:
        raise IdentifyError(f"only {int(mask.sum())} data points inside {lo:.0f}-{hi:.0f} cm⁻¹ "
                            "(the references cover a different range than the data)")
    nu = nu_all[mask]
    order = np.argsort(nu)                                         # wavelength data arrive in descending nu
    nu = nu[order]
    t = obs.y[mask][order]
    sig_all = obs.sigma[mask][order] if obs.sigma is not None else None
    # Nearly opaque points carry only a LOWER LIMIT on optical depth (T is noise around zero), so they
    # would dominate chi^2 without constraining anything: leave them out and say so.
    cut = _OPAQUE_T if sig_all is None else max(_OPAQUE_T, 3.0 * float(np.median(sig_all)))
    keep = t >= cut
    n_opaque = int((~keep).sum())
    if keep.sum() < 20:
        raise IdentifyError("almost every point in the window is opaque (T < 5 %); nothing to fit")
    nu, t = nu[keep], t[keep]
    sig_all = sig_all[keep] if sig_all is not None else None
    t_c = np.clip(t, _T_FLOOR, None)
    tau_obs = -np.log(t_c)
    sigma_given = obs.sigma is not None
    sig_t = sig_all

    K = np.column_stack([r.on(nu) for r in refs])                   # (N, R)
    warnings: List[str] = []
    no_cov = K.max(axis=0) <= 0
    for i in np.flatnonzero(no_cov):
        warnings.append(f"{refs[i].name}: reference has no data inside the fit window - excluded")
    span = max(float(nu[-1] - nu[0]), 1e-9)
    u = 2.0 * (nu - nu[0]) / span - 1.0
    C = np.column_stack([np.ones_like(u), u]) if fit_continuum else np.zeros((nu.size, 0))
    n_r, n_c = K.shape[1], C.shape[1]
    A_raw = np.column_stack([K, C])
    lower = np.concatenate([np.zeros(n_r), np.full(n_c, -np.inf)])
    usable = np.concatenate([~no_cov, np.ones(n_c, bool)])
    cols = np.flatnonzero(usable)

    def weights(sigma: FloatArray) -> Tuple[FloatArray, FloatArray]:
        return A_raw / sigma[:, None], tau_obs / sigma

    sigma_est = not sigma_given
    if sigma_given:
        sigma_tau = np.maximum(sig_t / t_c, 1e-6)
    else:                                                          # robust noise from a first unweighted pass
        A0, b0 = weights(np.ones_like(tau_obs))
        x0, _ = _solve(A0, b0, lower, cols)
        # Measurement noise is additive in TRANSMITTANCE, so estimate it there (robust MAD of the first
        # pass) and propagate to optical depth: sigma_tau = sigma_T / T.  A constant sigma_tau would
        # over-weight deep band cores and make chi^2 meaningless for comparing models.
        r0_t = t_c - np.exp(-(A_raw @ x0))
        s_t = max(_MAD_TO_SIGMA * float(np.median(np.abs(r0_t - np.median(r0_t)))), 1e-6)
        sigma_tau = s_t / t_c
    A, b = weights(sigma_tau)
    x, chi2 = _solve(A, b, lower, cols)
    amp = x[:n_r]
    dof = max(nu.size - int(np.count_nonzero(x[cols])) - 0, 1)

    active = np.flatnonzero((amp > 0) & usable[:n_r])
    act_cols = np.concatenate([active, n_r + np.arange(n_c)]).astype(int)
    cov = np.linalg.pinv(A[:, act_cols].T @ A[:, act_cols])
    unc = np.zeros(n_r)
    unc[active] = np.sqrt(np.clip(np.diag(cov)[: active.size], 0.0, None))
    inactive = np.flatnonzero(usable[:n_r] & (amp <= 0))
    for i in inactive:                                             # 1-sigma upper bound if it were present
        norm = float(np.linalg.norm(A[:, i]))
        unc[i] = 1.0 / norm if norm > 0 else np.inf

    sig_corr = np.zeros((n_r, n_r))
    if active.size:
        d = np.sqrt(np.clip(np.diag(cov)[: active.size], 1e-300, None))
        corr = cov[: active.size, : active.size] / np.outer(d, d)
        sig_corr[np.ix_(active, active)] = corr
    np.fill_diagonal(sig_corr, 1.0)

    # drop-one significance
    delta = np.zeros(n_r)
    for i in active:
        keep = np.array([c for c in cols if c != i])
        _, chi2_without = _solve(A, b, lower, keep)
        delta[i] = max(chi2_without - chi2, 0.0)
    signif = np.sqrt(delta)

    # degeneracy diagnostics on the weighted, normalised reference shapes
    Kw = A[:, :n_r]
    norms = np.linalg.norm(Kw, axis=0)
    Kn = np.divide(Kw, norms, out=np.zeros_like(Kw), where=norms > 0)
    sim = Kn.T @ Kn
    good = norms > 0
    cond = float(np.linalg.cond(Kn[:, good])) if good.sum() > 1 else 1.0
    for i in range(n_r):
        for j in range(i + 1, n_r):
            if good[i] and good[j] and sim[i, j] > 0.8 and (amp[i] > 0 or amp[j] > 0):
                warnings.append(f"{refs[i].name} and {refs[j].name} overlap strongly here "
                                f"(shape similarity {sim[i, j]:.2f}"
                                + (f", amplitude correlation {sig_corr[i, j]:+.2f}" if sig_corr[i, j] else "")
                                + "): their individual amplitudes are degenerate - trust the sum, not the split.")

    tau_by_ref = K * amp[None, :]
    cont = C @ x[n_r:] if n_c else np.zeros_like(tau_obs)
    tau_model = tau_by_ref.sum(axis=1) + cont

    comps = [ComponentFit(refs[i].ref_id, refs[i].name, refs[i].formula, float(amp[i]), float(unc[i]),
                          float(delta[i]), float(signif[i]),
                          bool(amp[i] > 0 and signif[i] >= detection_sigma), refs[i].quantitative,
                          refs[i].source_key) for i in range(n_r) if usable[i]]
    if any(c.detected and not c.quantitative for c in comps):
        warnings.append("Amplitudes of non-quantitative references are RELATIVE (shape only): they are not "
                        "abundances or column densities.")
    if n_opaque:
        warnings.append(f"{n_opaque} nearly opaque points (T < {cut:.0%}) were excluded: optical depth there is "
                        "only a lower limit, so saturated band cores do not constrain the amplitudes.")
    warnings.append("Linear optical-depth model: ignores in-band saturation structure, clouds/hazes and "
                    "temperature gradients.")

    attributions = _attribute(features or (), nu, tau_obs - cont, K, amp, ids) if features else []
    return DecompositionResult(
        comps, float(chi2), float(chi2 / dof), int(nu.size), (float(nu[0]), float(nu[-1])), nu, tau_obs,
        tau_model, tau_by_ref, cont, sigma_tau, sigma_est, ids, sim, sig_corr, cond, attributions, warnings,
        rms_t=float(np.sqrt(np.mean((t - np.exp(-tau_model)) ** 2))))


def _attribute(features, nu, tau_net, K, amp, ids) -> List[FeatureAttribution]:
    """Per-feature share of integrated optical depth explained by each reference."""
    out = []
    for f in features:
        sel = (nu >= f.nu_lo_cm1) & (nu <= f.nu_hi_cm1)
        if sel.sum() < 3:                                            # narrow feature: take the nearest points
            k = int(np.argmin(np.abs(nu - f.nu_peak_cm1)))
            sel = np.zeros_like(sel)
            sel[max(k - 1, 0): k + 2] = True
        x = nu[sel]
        part = np.array([_trapz(K[sel, i] * amp[i], x) for i in range(K.shape[1])])
        model_int = float(part.sum())
        obs_int = float(_trapz(np.clip(tau_net[sel], 0.0, None), x))
        shares = {ids[i]: float(part[i] / model_int) for i in range(len(ids))
                  if model_int > 0 and part[i] > 0}
        unexplained = float(np.clip(1.0 - model_int / obs_int, 0.0, 1.0)) if obs_int > 0 else 0.0
        out.append(FeatureAttribution(f.nu_peak_cm1, f.nu_lo_cm1, f.nu_hi_cm1, shares, unexplained,
                                      sum(s > 0.2 for s in shares.values()) >= 2))
    return out
