"""
ECAS spectral time series: a spectrum for every moment of a transit observation.

Transit photometry alone is a single number per time; the *spectrum* of the planet's atmosphere
comes from time-series spectroscopy recorded during the same observation, e.g. JWST NIRSpec/NIRCam/
MIRI time-series observations whose ``*_x1dints.fits`` products hold one extracted 1-D spectrum per
integration (obtained from MAST - see :mod:`app.core.archives`).  A :class:`SpectralTimeSeries`
holds that ``(time, wavelength)`` flux matrix and provides:

* :meth:`white_light`            the summed light curve (so the usual transit analysis applies),
* :meth:`relative_spectrum_at`   *spectrum at time t*: F(t)/F_out per wavelength - flat out of
                                 transit, dipping by the wavelength-dependent depth in transit,
* :meth:`transmission_spectrum`  per-wavelength transit depth (the planet's transmission spectrum).

Loaders: :func:`load_x1dints` (JWST) and :func:`load_matrix_csv` (first row = wavelengths, first
column = times). JWST x1dints vector columns hold multiple integrations in one EXTRACT1D table;
older/synthetic files with one integration per extension are also accepted. Qt-free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .transit import (
    DepthSpectrum, LightCurve, TransitParams, _sky_separation, extract_transmission_spectrum, transit_model,
)

FloatArray = NDArray[np.float64]


class TimeSeriesError(ValueError):
    """Raised for unreadable or inconsistent spectral time series."""


@dataclass
class CorrelatedNoiseDiagnostics:
    """Robust residual-scatter growth versus integration bin size; not an error correction."""

    bin_sizes: np.ndarray
    beta_median: FloatArray
    beta_low: FloatArray
    beta_high: FloatArray
    white_noise_ppm_median: float
    n_channels: int


@dataclass
class SpectralTimeSeries:
    time: FloatArray                         # (Nt,) days (offset removed; see meta["time_offset"])
    wavelength_um: FloatArray                # (Nw,) ascending
    flux: FloatArray                         # (Nt, Nw)
    flux_err: Optional[FloatArray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64)
        self.wavelength_um = np.asarray(self.wavelength_um, dtype=np.float64)
        self.flux = np.asarray(self.flux, dtype=np.float64)
        if self.flux.ndim != 2 or self.flux.shape != (self.time.size, self.wavelength_um.size):
            raise TimeSeriesError(f"flux must have shape (n_times, n_wavelengths) = "
                                  f"({self.time.size}, {self.wavelength_um.size}); got {self.flux.shape}")
        if self.flux_err is not None:
            self.flux_err = np.asarray(self.flux_err, dtype=np.float64)
            if self.flux_err.shape != self.flux.shape:
                raise TimeSeriesError("flux_err must match flux in shape")
        good_w = np.isfinite(self.wavelength_um) & (self.wavelength_um > 0) & np.isfinite(self.flux).all(axis=0)
        good_t = np.isfinite(self.time) & np.isfinite(self.flux[:, good_w]).all(axis=1)
        if good_t.sum() < 50 or good_w.sum() < 8:
            raise TimeSeriesError("need at least 50 finite integrations and 8 finite wavelength channels "
                                  "to build the transit light curve")
        self.time, self.wavelength_um, self.flux = self.time[good_t], self.wavelength_um[good_w], self.flux[np.ix_(good_t, good_w)]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[np.ix_(good_t, good_w)]
        o = np.argsort(self.wavelength_um)
        self.wavelength_um, self.flux = self.wavelength_um[o], self.flux[:, o]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[:, o]
        ot = np.argsort(self.time, kind="stable")
        self.time, self.flux = self.time[ot], self.flux[ot]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[ot]

    @property
    def n_times(self) -> int:
        return int(self.time.size)

    @property
    def n_wavelengths(self) -> int:
        return int(self.wavelength_um.size)

    def white_light(self) -> LightCurve:
        """Wavelength-summed light curve (the input to the ordinary transit analysis)."""
        total = self.flux.sum(axis=1)
        err = None if self.flux_err is None else np.sqrt((self.flux_err ** 2).sum(axis=1))
        return LightCurve(self.time, total, err, {"source": self.meta.get("source", "spectral time series"),
                                                  "white_light_of": f"{self.n_wavelengths} channels"})

    def _out_mask(self, params: Optional[TransitParams]) -> np.ndarray:
        if params is not None:
            z, cos_phi = _sky_separation(self.time, params)
            mask = (z > 1.0 + params.rp_rs) | (cos_phi <= 0.0)
            if mask.sum() >= 5:
                return mask
        edge = max(int(0.15 * self.n_times), 3)                      # fallback: both ends of the series
        mask = np.zeros(self.n_times, bool)
        mask[:edge] = mask[-edge:] = True
        return mask

    def out_of_transit_spectrum(self, params: Optional[TransitParams] = None) -> FloatArray:
        return np.median(self.flux[self._out_mask(params)], axis=0)

    def relative_spectrum_at(self, t: float, params: Optional[TransitParams] = None,
                             half_window_days: float = 0.0) -> Tuple[FloatArray, FloatArray, int]:
        """
        Spectrum at time ``t`` relative to the out-of-transit spectrum: ``(wavelength, F/F_out, n)``.
        Integrations within ``half_window_days`` are averaged (``n`` of them); with no window or none
        inside it, the nearest integration is used.
        """
        sel = np.abs(self.time - t) <= half_window_days if half_window_days > 0 else np.zeros(self.n_times, bool)
        if not sel.any():
            sel[int(np.argmin(np.abs(self.time - t)))] = True
        ref = np.where(self.out_of_transit_spectrum(params) > 0, self.out_of_transit_spectrum(params), np.nan)
        return self.wavelength_um, self.flux[sel].mean(axis=0) / ref, int(sel.sum())

    def transmission_spectrum(self, params: TransitParams) -> DepthSpectrum:
        """Per-channel transit depth with the geometry fixed to ``params`` (see transit.py)."""
        return extract_transmission_spectrum(
            self.time, self.flux, self.wavelength_um, params, flux_err=self.flux_err)

    def characterize_correlated_noise(
        self, params: TransitParams, max_bin_size: int = 20,
    ) -> CorrelatedNoiseDiagnostics:
        """Measure robust residual-scatter growth after removing a fixed-geometry transit template.

        Consecutive integrations are grouped without crossing gaps larger than 1.5 median cadences.
        The beta-style ratios are diagnostics only and never alter extraction uncertainties.
        """
        if isinstance(max_bin_size, bool) or int(max_bin_size) != max_bin_size or max_bin_size < 1:
            raise TimeSeriesError("maximum noise bin size must be a positive integer")
        max_bin_size = min(int(max_bin_size), self.n_times // 5)
        if max_bin_size < 1:
            raise TimeSeriesError("at least five integrations are required for noise diagnostics")

        baseline = self.out_of_transit_spectrum(params)
        if not np.isfinite(baseline).all() or np.any(baseline <= 0.0):
            raise TimeSeriesError("noise diagnostics need finite, positive out-of-transit flux")
        spectrum = self.transmission_spectrum(params)
        template = (1.0 - transit_model(self.time, params)) / params.rp_rs**2
        residual = self.flux / baseline[None, :] - (1.0 - template[:, None] * spectrum.depth[None, :])
        cadence = float(np.median(np.diff(self.time)))
        if not np.isfinite(cadence) or cadence <= 0.0:
            raise TimeSeriesError("noise diagnostics need increasing integration times")
        boundaries = np.flatnonzero(np.diff(self.time) > 1.5 * cadence) + 1
        segments = [part for part in np.split(np.arange(self.n_times), boundaries) if part.size]

        def robust_scale(values: FloatArray) -> float:
            center = float(np.median(values))
            return float(1.4826 * np.median(np.abs(values - center)))

        white_scales = np.array([robust_scale(residual[:, channel])
                                 for channel in range(self.n_wavelengths)])
        valid_channels = np.isfinite(white_scales) & (white_scales > 0.0)
        if not valid_channels.any():
            raise TimeSeriesError("residual scatter is zero or invalid in every wavelength channel")

        bin_sizes = np.arange(1, max_bin_size + 1, dtype=np.int64)
        beta = np.full((bin_sizes.size, self.n_wavelengths), np.nan, dtype=np.float64)
        for row, bin_size in enumerate(bin_sizes):
            for channel in np.flatnonzero(valid_channels):
                means = []
                for segment in segments:
                    n_complete = segment.size // bin_size
                    if n_complete:
                        samples = residual[segment[:n_complete * bin_size], channel]
                        means.extend(samples.reshape(n_complete, bin_size).mean(axis=1))
                if len(means) >= 5:
                    beta[row, channel] = (
                        robust_scale(np.asarray(means)) * np.sqrt(bin_size) / white_scales[channel])

        supported = np.isfinite(beta).any(axis=1)
        if not supported.any():
            raise TimeSeriesError("not enough complete integration bins for noise diagnostics")
        beta_valid = beta[supported]
        return CorrelatedNoiseDiagnostics(
            bin_sizes[supported],
            np.nanmedian(beta_valid, axis=1),
            np.nanpercentile(beta_valid, 16, axis=1),
            np.nanpercentile(beta_valid, 84, axis=1),
            float(np.median(white_scales[valid_channels]) * 1e6),
            int(valid_channels.sum()),
        )


# ===================================================================== loaders
def _to_um(wl: FloatArray, unit: str) -> FloatArray:
    """Wavelengths -> micrometres.  ``auto``: <= 60 is um, anything larger is nm (time-series
    instruments report wavelengths, not wavenumbers); pass ``cm-1`` explicitly for wavenumbers."""
    unit = unit.lower()
    if unit == "auto":
        unit = "um" if float(np.nanmax(wl)) <= 60.0 else "nm"
    if unit in ("um", "µm", "micron", "microns"):
        return wl
    if unit == "nm":
        return wl / 1.0e3
    if unit in ("cm-1", "cm⁻¹", "wavenumber"):
        return 1.0e4 / wl
    raise TimeSeriesError(f"unknown wavelength unit {unit!r}")


def load_matrix_csv(path: Union[str, Path], wavelength_unit: str = "auto") -> SpectralTimeSeries:
    """
    CSV/TSV/whitespace matrix: the first row holds the wavelengths (first cell is a label such as
    ``time``), each following row is ``time, flux_1 ... flux_N``.  ``#`` comments are ignored.
    """
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) < 51:
        raise TimeSeriesError("a spectral time series needs a wavelength row and at least 50 time rows")
    split = lambda ln: [t for t in re.split(r"[,;\t ]+", ln) if t]          # noqa: E731
    head = split(lines[0])
    try:
        wl = np.array([float(v) for v in head[1:]])
    except ValueError as exc:
        raise TimeSeriesError("first row must be a label followed by the wavelengths") from exc
    rows = []
    for ln in lines[1:]:
        try:
            vals = [float(v) for v in split(ln)]
        except ValueError:
            continue
        if len(vals) == wl.size + 1:
            rows.append(vals)
    if len(rows) < 50:
        raise TimeSeriesError(f"only {len(rows)} rows have {wl.size + 1} numeric columns (time + {wl.size} wavelengths)")
    arr = np.array(rows)
    return SpectralTimeSeries(arr[:, 0], _to_um(wl, wavelength_unit), arr[:, 1:],
                              meta={"source": str(path), "time_offset": 0.0})


def load_x1dints(path: Union[str, Path]) -> SpectralTimeSeries:
    """JWST ``*_x1dints.fits`` with vector EXTRACT1D rows and an INT_TIMES table."""
    try:
        from astropy.io import fits
    except ImportError as exc:                                             # pragma: no cover
        raise TimeSeriesError("reading FITS needs astropy (pip install astropy)") from exc
    try:
        with fits.open(path, memmap=False) as hdul:
            wl, fluxes, errs, times, time_kind = None, [], [], None, ""
            for hdu in hdul:
                name = str(hdu.name).upper()
                if name == "EXTRACT1D" and getattr(hdu, "columns", None) is not None and hdu.data is not None:
                    cols = {n.upper(): n for n in hdu.columns.names}
                    if "WAVELENGTH" not in cols or "FLUX" not in cols:
                        raise TimeSeriesError("EXTRACT1D extension lacks WAVELENGTH/FLUX columns")
                    w = np.asarray(hdu.data[cols["WAVELENGTH"]], dtype=np.float64)
                    f = np.asarray(hdu.data[cols["FLUX"]], dtype=np.float64)
                    e = np.asarray(hdu.data[cols["FLUX_ERROR"]], dtype=np.float64) if "FLUX_ERROR" in cols else None
                    if w.ndim == 1:
                        w = w[None, :]
                    if f.ndim == 1:
                        f = f[None, :]
                    if e is not None and e.ndim == 1:
                        e = e[None, :]
                    if w.ndim != 2 or f.shape != w.shape or (e is not None and e.shape != w.shape):
                        raise TimeSeriesError("EXTRACT1D wavelength, flux and error arrays have incompatible shapes")
                    if wl is None:
                        wl = w[0]
                    elif w.shape[1] != wl.size or not np.allclose(w, wl[None, :], rtol=1e-8, atol=0.0,
                                                                    equal_nan=True):
                        raise TimeSeriesError("integrations have different wavelength grids")
                    fluxes.append(f)
                    if "FLUX_ERROR" in cols:
                        errs.append(e)
                elif name == "INT_TIMES" and hdu.data is not None:
                    names = {n.lower(): n for n in hdu.columns.names}
                    pick = next((n for k, n in names.items() if "mid" in k and "bjd" in k), None)
                    time_kind = "BJD_TDB"
                    if pick is None:
                        pick = next((n for k, n in names.items() if "mid" in k and "mjd" in k), None)
                        time_kind = "MJD_UTC"
                    if pick is not None:
                        times = np.asarray(hdu.data[pick], dtype=np.float64)
    except OSError as exc:
        raise TimeSeriesError(f"cannot read {path}: {exc}") from exc
    n_integrations = sum(part.shape[0] for part in fluxes)
    if n_integrations < 20:
        raise TimeSeriesError(f"found {n_integrations} EXTRACT1D integrations; this does not look like an x1dints file "
                              "(a combined *_x1d.fits holds one spectrum - load it as an ordinary spectrum)")
    flux = np.concatenate(fluxes, axis=0)
    if wl is None:
        raise TimeSeriesError("no EXTRACT1D wavelength grid found")
    meta: Dict[str, Any] = {"source": str(path), "flux_units": "as in file (JWST: Jy)"}
    if times is not None and times.size == flux.shape[0]:
        times = times[: flux.shape[0]]
        meta.update(time_offset=float(times[0]), time_kind=time_kind)
        times = times - times[0]
    elif times is not None:
        raise TimeSeriesError(f"found {times.size} integration timestamps for {flux.shape[0]} spectra")
    else:
        times = np.arange(flux.shape[0], dtype=np.float64)
        meta.update(time_offset=0.0, time_kind="integration number (no INT_TIMES found: set times manually)")
    flux_err = np.concatenate(errs, axis=0) if len(errs) == len(fluxes) else None
    return SpectralTimeSeries(times, wl, flux, flux_err, meta)
