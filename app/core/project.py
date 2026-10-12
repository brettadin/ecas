"""
ECAS project files (Phase 6) - Qt-free.

A ``.ecas`` project is a zip holding ``state.json`` (scalars, settings, metadata) and
``arrays.npz`` (the observation and light-curve arrays).  Plain, inspectable, and
independent of the GUI: the window captures/restores a :class:`ProjectState`.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .engine import Unit
from .fitting import ObservedSpectrum
from .transit import DepthSpectrum, LightCurve
from .timeseries import SpectralTimeSeries

PROJECT_VERSION = 1
EXTENSION = ".ecas"


class ProjectError(ValueError):
    """Raised for unreadable or incompatible project files."""


@dataclass
class ProjectState:
    """Everything needed to reproduce a session (results are re-derived on load)."""

    conditions: Dict[str, Any] = field(default_factory=dict)     # P, T, envelope, line_shape
    multipliers: Dict[str, float] = field(default_factory=dict)  # gas_id -> value
    anomaly_threshold: float = 3.0
    stellar: Dict[str, float] = field(default_factory=dict)      # radius_rsun, mass_msun, teff_k
    detrend_window_days: float = 1.0
    config_path: Optional[str] = None
    notes: str = ""
    observation: Optional[ObservedSpectrum] = None
    lightcurve: Optional[LightCurve] = None
    spectral_series: Optional[SpectralTimeSeries] = None
    depth_spectrum: Optional[DepthSpectrum] = None


def save_project(path: Union[str, Path], state: ProjectState) -> Path:
    """Write ``state`` to ``path`` (``.ecas`` appended if missing)."""
    path = Path(path)
    if path.suffix != EXTENSION:
        path = path.with_suffix(path.suffix + EXTENSION if path.suffix else EXTENSION)
    arrays: Dict[str, np.ndarray] = {}
    doc: Dict[str, Any] = {
        "version": PROJECT_VERSION,
        "conditions": state.conditions, "multipliers": state.multipliers,
        "anomaly_threshold": state.anomaly_threshold, "stellar": state.stellar,
        "detrend_window_days": state.detrend_window_days,
        "config_path": state.config_path, "notes": state.notes,
        "observation": None, "lightcurve": None, "spectral_series": None, "depth_spectrum": None,
    }
    if state.observation is not None:
        o = state.observation
        arrays.update(obs_x=o.x, obs_y=o.y)
        if o.sigma is not None:
            arrays["obs_sigma"] = o.sigma
        doc["observation"] = {"unit": o.unit.value, "meta": _jsonable(o.meta)}
    if state.lightcurve is not None:
        lc = state.lightcurve
        arrays.update(lc_time=lc.time, lc_flux=lc.flux)
        if lc.flux_err is not None:
            arrays["lc_err"] = lc.flux_err
        doc["lightcurve"] = {"meta": _jsonable(lc.meta)}
    if state.spectral_series is not None:
        series = state.spectral_series
        arrays.update(series_time=series.time, series_wavelength=series.wavelength_um,
                      series_flux=series.flux)
        if series.flux_err is not None:
            arrays["series_flux_err"] = series.flux_err
        doc["spectral_series"] = {"meta": _jsonable(series.meta)}
    if state.depth_spectrum is not None:
        spectrum = state.depth_spectrum
        arrays.update(depth_wavelength=spectrum.wavelength_um, depth=spectrum.depth,
                      depth_err=spectrum.depth_err)
        doc["depth_spectrum"] = {"present": True}

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("state.json", json.dumps(doc, indent=2))
        zf.writestr("arrays.npz", buf.getvalue())
    return path


def load_project(path: Union[str, Path]) -> ProjectState:
    """Read a project; raises :class:`ProjectError` on any structural problem."""
    try:
        with zipfile.ZipFile(path) as zf:
            doc = json.loads(zf.read("state.json"))
            arrays = np.load(io.BytesIO(zf.read("arrays.npz")))
            data = {k: arrays[k] for k in arrays.files}
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError, ValueError) as exc:
        raise ProjectError(f"Cannot read project {path}: {exc}") from exc
    if doc.get("version") != PROJECT_VERSION:
        raise ProjectError(f"Unsupported project version {doc.get('version')!r}")

    state = ProjectState(
        conditions=doc.get("conditions", {}), multipliers=doc.get("multipliers", {}),
        anomaly_threshold=float(doc.get("anomaly_threshold", 3.0)),
        stellar=doc.get("stellar", {}), detrend_window_days=float(doc.get("detrend_window_days", 1.0)),
        config_path=doc.get("config_path"), notes=doc.get("notes", ""),
    )
    try:
        if doc.get("observation"):
            o = doc["observation"]
            state.observation = ObservedSpectrum(data["obs_x"], data["obs_y"], Unit(o["unit"]),
                                                 data.get("obs_sigma"), o.get("meta", {}))
        if doc.get("lightcurve"):
            state.lightcurve = LightCurve(data["lc_time"], data["lc_flux"], data.get("lc_err"),
                                          doc["lightcurve"].get("meta", {}))
        if doc.get("spectral_series"):
            series = doc["spectral_series"]
            state.spectral_series = SpectralTimeSeries(
                data["series_time"], data["series_wavelength"], data["series_flux"],
                data.get("series_flux_err"), series.get("meta", {}))
        if doc.get("depth_spectrum"):
            spectrum = doc["depth_spectrum"]
            state.depth_spectrum = DepthSpectrum(
                data["depth_wavelength"], data["depth"], data["depth_err"])
    except (KeyError, ValueError) as exc:
        raise ProjectError(f"Project arrays are inconsistent: {exc}") from exc
    return state


def _jsonable(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only JSON-safe metadata values (numpy scalars converted)."""
    def convert(v: Any) -> Any:
        if isinstance(v, np.generic):
            return convert(v.item())
        if isinstance(v, dict):
            return {str(k): converted for k, value in v.items()
                    if (converted := convert(value)) is not _OMIT}
        if isinstance(v, list):
            return [converted for value in v if (converted := convert(value)) is not _OMIT]
        return v if isinstance(v, (str, int, float, bool, type(None))) else _OMIT

    out = convert(meta)
    return out


_OMIT = object()
