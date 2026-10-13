"""
ECAS reference-spectrum library.

A :class:`ReferenceSpectrum` is the *shape* a molecule imprints on a transmission spectrum:
``k(nu)`` such that the optical depth contributed by the molecule is ``amplitude * k(nu)``.
Whether ``amplitude`` is a real column density depends on the source:

* ``quantitative=True``   - an absorption coefficient / cross-section with known units (e.g. HITRAN
  cross-sections, NIST quantitative IR).  Amplitudes can later be turned into concentrations.
* ``quantitative=False``  - a spectral *shape* only (e.g. NIST evaluated/Coblentz transmittance
  spectra, ORCA harmonic bands, mock data).  Amplitudes are relative and **must not** be quoted as
  abundances.  The UI and reports say so.

Sources of references
---------------------
* :meth:`ReferenceLibrary.from_engine`   the gases already in the simulator (their ORCA/mock bands)
* :func:`import_jcamp`                   any JCAMP-DX file (NIST WebBook downloads, PNNL, ...); the
                                         header (title, CAS, formula, origin) is kept as provenance
* :func:`reference_from_arrays`          anything else (e.g. HITRAN cross-sections fetched elsewhere)

Qt-free.
"""
from __future__ import annotations

import json
import bz2
import gzip
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .tables import RawTable, TableError, read_table

FloatArray = NDArray[np.float64]
DEFAULT_DIR = Path.home() / ".ecas" / "library"


class LibraryError(ValueError):
    """Raised for unusable reference spectra or library files."""


@dataclass
class ReferenceSpectrum:
    ref_id: str
    name: str
    formula: str
    nu_cm1: FloatArray                      # ascending
    k: FloatArray                           # >= 0, same length
    source_key: str                         # key into core.sources.SOURCES
    quantitative: bool = False
    y_kind: str = "shape"                   # shape | absorbance | cross_section | tau_per_multiplier
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nu_cm1 = np.asarray(self.nu_cm1, dtype=np.float64)
        self.k = np.clip(np.asarray(self.k, dtype=np.float64), 0.0, None)
        if self.nu_cm1.size < 8 or self.nu_cm1.size != self.k.size:
            raise LibraryError(f"reference {self.ref_id!r} needs >= 8 matching x/y points")
        order = np.argsort(self.nu_cm1, kind="stable")
        self.nu_cm1, self.k = self.nu_cm1[order], self.k[order]
        if not np.isfinite(self.k).all() or self.k.max() <= 0:
            raise LibraryError(f"reference {self.ref_id!r} has no positive finite values")

    @property
    def coverage_cm1(self):
        return float(self.nu_cm1[0]), float(self.nu_cm1[-1])

    def on(self, nu_cm1: FloatArray) -> FloatArray:
        """Resample onto ``nu_cm1``; zero outside the reference's own range (no extrapolation)."""
        return np.interp(np.asarray(nu_cm1, dtype=np.float64), self.nu_cm1, self.k, left=0.0, right=0.0)


# ----------------------------------------------------------------------- building
def reference_from_arrays(ref_id, name, formula, nu_cm1, k, source_key, quantitative=False,
                          y_kind="shape", **meta) -> ReferenceSpectrum:
    return ReferenceSpectrum(ref_id, name, formula, nu_cm1, k, source_key, quantitative, y_kind, dict(meta))


def nist_webbook_url(identifier: str, index: int = 0, spectrum: str = "ir") -> str:
    """Return a NIST WebBook IR or UV/Vis JCAMP endpoint."""
    value = str(identifier).strip().upper().replace("-", "")
    if value.startswith("C"):
        value = value[1:]
    if not value.isdigit():
        raise LibraryError("NIST WebBook lookup needs a numeric CAS/WebBook identifier")
    if int(index) < 0:
        raise LibraryError("NIST spectrum index must be non-negative")
    kind = str(spectrum).strip().lower()
    if kind in ("ir", "ir-spec", "infrared"):
        nist_type = "IR-SPEC"
    elif kind in ("uv", "uv-vis", "uv/vis", "uvvis"):
        nist_type = "UVVis"
    else:
        raise LibraryError("NIST spectrum type must be 'ir' or 'uvvis'")
    return ("https://webbook.nist.gov/cgi/cbook.cgi?" +
            urllib.parse.urlencode({"JCAMP": f"C{value}", "Index": int(index), "Type": nist_type}))


def fetch_nist_jcamp(identifier: str, index: int = 0, fetch=None, spectrum: str = "ir") -> tuple[str, str]:
    """Fetch one NIST WebBook IR/UV/Vis JCAMP document, returning ``(url, text)``.

    ``fetch`` is injectable for offline tests. Network access is deliberately
    opt-in through this function; callers must still record the returned URL.
    """
    url = nist_webbook_url(identifier, index, spectrum)
    if fetch is None:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8", errors="replace")
        except OSError as exc:
            raise LibraryError(f"NIST WebBook request failed: {exc}") from exc
    else:
        try:
            text = fetch(url)
        except Exception as exc:
            raise LibraryError(f"NIST WebBook request failed: {exc}") from exc
    if not str(text).lstrip().startswith("##"):
        raise LibraryError("NIST did not return a JCAMP-DX spectrum; check the identifier and spectrum index")
    return url, str(text)


def import_nist_jcamp(identifier: str, index: int = 0, fetch=None, spectrum: str = "ir") -> ReferenceSpectrum:
    """Fetch and parse one NIST WebBook IR or UV/Vis reference."""
    url, text = fetch_nist_jcamp(identifier, index, fetch, spectrum)
    with tempfile.NamedTemporaryFile("w", suffix=".jdx", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        ref = import_jcamp(path, source_key="nist_webbook",
                           ref_id=f"nist:{str(identifier).strip().upper().lstrip('C')}:{index}")
    finally:
        path.unlink(missing_ok=True)
    ref.meta.update({"nist_identifier": str(identifier), "nist_index": int(index), "query": url,
                     "retrieved_via": "NIST WebBook JCAMP endpoint", "spectrum": str(spectrum)})
    return ref


def hapi_cross_section_reference(molecule: str, nu_cm1, cross_section_cm2,
                                 temperature_k: float, pressure_atm: float = 1.0,
                                 source_url: str = "https://hitran.org/hapi/") -> ReferenceSpectrum:
    """Build a quantitative HITRAN/HAPI cross-section reference from sampled output.

    HAPI itself is optional and credential-gated; this adapter accepts the
    sampled cross-section returned by a caller's HAPI calculation, avoiding a
    hard dependency and making units/provenance explicit.
    """
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise LibraryError("HAPI cross-section temperature must be positive")
    x = np.asarray(nu_cm1, dtype=np.float64)
    y = np.asarray(cross_section_cm2, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 8 or not np.isfinite(y).all() or np.any(y < 0.0):
        raise LibraryError("HAPI cross-sections need matching finite 1-D wavenumber and non-negative arrays")
    return reference_from_arrays(
        f"hitran:{molecule}:{temperature_k:g}K:{pressure_atm:g}atm", molecule, molecule, x, y,
        "hapi", quantitative=True, y_kind="cross_section", units="cm2 molecule-1",
        temperature_k=float(temperature_k), pressure_atm=float(pressure_atm), query=source_url,
        retrieval_note="Cross-section supplied by the caller's HAPI calculation; HITRAN credentials required.")


def _open_text(path: Union[str, Path]):
    raw = Path(path).open("rb")
    suffix = str(path).lower()
    if suffix.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if suffix.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    raw.close()
    return Path(path).open("r", encoding="utf-8", errors="replace")


def import_exomol_transitions(transitions_path: Union[str, Path], states_path: Union[str, Path],
                              molecule: str, temperature_k: float = 1500.0,
                              ref_id: Optional[str] = None) -> ReferenceSpectrum:
    """Import an ExoMol ``.trans``/``.states`` pair as a line-strength shape.

    The mandatory ExoMol fields are parsed (state ID, energy, degeneracy, and
    upper/lower IDs plus Einstein A). This is intentionally shape-only: a full
    pressure/temperature cross-section requires a line profile and partition
    function, which ECAS does not infer from raw files.
    """
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise LibraryError("ExoMol temperature must be positive")
    states: Dict[int, tuple[float, float]] = {}
    try:
        with _open_text(states_path) as stream:
            for line in stream:
                fields = line.split()
                if not fields or fields[0].startswith("#") or len(fields) < 3:
                    continue
                try:
                    states[int(fields[0])] = (float(fields[1]), float(fields[2]))
                except ValueError:
                    continue
        nu, strength = [], []
        boltzmann = 0.69503476  # cm-1 K-1
        with _open_text(transitions_path) as stream:
            for line in stream:
                fields = line.split()
                if not fields or fields[0].startswith("#") or len(fields) < 3:
                    continue
                try:
                    upper, lower, einstein_a = int(fields[0]), int(fields[1]), float(fields[2])
                    e_lower, g_lower = states[lower]
                    e_upper, _ = states[upper]
                except (ValueError, KeyError):
                    continue
                frequency = e_upper - e_lower
                if frequency > 0.0 and einstein_a > 0.0:
                    nu.append(frequency)
                    strength.append(g_lower * np.exp(-e_lower / (boltzmann * temperature_k)) * einstein_a)
    except OSError as exc:
        raise LibraryError(f"cannot read ExoMol line-list files: {exc}") from exc
    if len(nu) < 8:
        raise LibraryError("ExoMol files yielded fewer than 8 usable transitions")
    return reference_from_arrays(
        ref_id or f"exomol:{molecule}:{Path(transitions_path).stem}", molecule, molecule,
        np.asarray(nu), np.asarray(strength), "exomol", quantitative=False, y_kind="line_strength",
        states_file=str(states_path), transitions_file=str(transitions_path), temperature_k=float(temperature_k),
        assumptions="lower-state Boltzmann weighting and Einstein-A strengths; no line profile or partition normalization")


def import_jcamp(path: Union[str, Path], source_key: str = "nist_webbook",
                 ref_id: Optional[str] = None) -> ReferenceSpectrum:
    """
    Import an uncompressed JCAMP-DX spectrum as a reference.  The y-axis is converted to a
    non-negative optical-depth-like shape: transmittance -> ``-ln T``; absorbance -> ``A ln 10``;
    absorption coefficient / cross-section is kept as is and marked quantitative.
    """
    try:
        table = read_table(path)
    except TableError as exc:
        raise LibraryError(str(exc)) from exc
    if table.kind != "jcamp":
        raise LibraryError(f"{path} is not a JCAMP-DX file")
    tags = table.tags
    xu = tags.get("XUNITS", "").upper()
    yu = tags.get("YUNITS", "").upper()
    x, y = table.data[:, 0].astype(float), table.data[:, 1].astype(float)

    if "MICROMETER" in xu or "MICRON" in xu or xu in ("UM", "µM"):
        x = 1.0e4 / x
    elif "NANOMETER" in xu or xu == "NM":
        x = 1.0e7 / x
    elif not ("1/CM" in xu or "CM-1" in xu or "WAVENUMBER" in xu):
        raise LibraryError(f"unsupported X units {xu!r} in JCAMP file (need 1/CM, micrometers or nm)")

    quantitative = False
    if "TRANSMITTANCE" in yu:
        t = y / 100.0 if np.nanmax(y) > 1.5 else y
        k, kind = -np.log(np.clip(t, 1.0e-4, 1.0)), "shape"
    elif "ABSORBANCE" in yu:
        k, kind = y * np.log(10.0), "absorbance"
    elif "LOG EPSILON" in yu or "LOG MOLAR" in yu or "MOLAR ABSORPTIVITY" in yu:
        # NIST UV/Vis commonly reports log10(molar absorptivity).  It is a
        # useful spectral shape, but without path length/concentration it is
        # not an abundance-ready cross-section in this library.
        baseline = float(np.nanpercentile(y, 5.0))
        k, kind = np.maximum(y - baseline, 0.0), "uv_log_epsilon"
    elif "ABSORPTION COEFFICIENT" in yu or "CROSS" in yu or "ABSORPTIVITY" in yu:
        k, kind, quantitative = y, "cross_section", True
    else:
        raise LibraryError(f"cannot interpret Y units {yu!r}; expected transmittance, absorbance or "
                           "an absorption coefficient")
    name = tags.get("TITLE", "").strip() or Path(path).stem
    formula = tags.get("MOLFORM", "").strip().replace(" ", "")
    cas = tags.get("CASREGISTRYNO", "").strip()
    rid = ref_id or (f"cas:{cas}" if cas else f"file:{Path(path).stem}")
    return ReferenceSpectrum(
        rid, name, formula, x, k, source_key, quantitative, kind,
        {"cas": cas, "state": tags.get("STATE", ""), "origin": tags.get("ORIGIN", ""),
         "owner": tags.get("OWNER", ""), "resolution": tags.get("RESOLUTION", ""),
         "source_reference": tags.get("SOURCEREFERENCE", ""), "data_type": tags.get("DATATYPE", ""),
         "file": str(path), "x_units": xu, "y_units": yu})


class ReferenceLibrary:
    """A named collection of references, optionally persisted to a directory."""

    def __init__(self, refs: Iterable[ReferenceSpectrum] = ()):
        self._refs: Dict[str, ReferenceSpectrum] = {}
        for r in refs:
            self.add(r)

    # ----- container
    def add(self, ref: ReferenceSpectrum) -> None:
        self._refs[ref.ref_id] = ref

    def remove(self, ref_id: str) -> None:
        self._refs.pop(ref_id, None)

    def get(self, ref_id: str) -> ReferenceSpectrum:
        return self._refs[ref_id]

    def ids(self) -> List[str]:
        return list(self._refs)

    def __len__(self) -> int:
        return len(self._refs)

    def __iter__(self):
        return iter(self._refs.values())

    def __contains__(self, ref_id: str) -> bool:
        return ref_id in self._refs

    def coverage_cm1(self, ref_ids: Optional[Sequence[str]] = None):
        refs = [self._refs[i] for i in (ref_ids or self._refs)]
        return min(r.coverage_cm1[0] for r in refs), max(r.coverage_cm1[1] for r in refs)

    # ----- engine-derived references
    @classmethod
    def from_engine(cls, engine, n_points: int = 6000) -> "ReferenceLibrary":
        """
        One reference per simulator gas: its optical depth for a *unit multiplier* under the
        engine's current conditions.  The fitted amplitude of such a reference is therefore directly
        comparable with that gas's slider value.
        """
        nu = np.linspace(*engine.grid.spec.wavenumber_range_cm1, n_points)
        tau = engine.tau_on(nu)                                    # (N, G)
        refs = []
        for gas in engine.gases:
            is_mock = gas.orca_out.startswith("mock://")
            refs.append(ReferenceSpectrum(
                gas.gas_id, gas.name, gas.formula, nu, tau[:, engine.gas_index[gas.gas_id]],
                "mock" if is_mock else "orca", False, "tau_per_multiplier",
                {"origin": "simulator band model", "orca_out": gas.orca_out,
                 "conditions": f"P={engine.conditions.pressure_atm:g} atm, T={engine.conditions.temperature_k:g} K"}))
        return cls(refs)

    def merged(self, other: "ReferenceLibrary") -> "ReferenceLibrary":
        out = ReferenceLibrary(self)
        for r in other:
            out.add(r)
        return out

    # ----- persistence
    def save(self, directory: Union[str, Path] = DEFAULT_DIR) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        index = []
        for r in self:
            fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", r.ref_id) + ".npz"
            np.savez_compressed(d / fname, nu=r.nu_cm1, k=r.k)
            index.append({"ref_id": r.ref_id, "name": r.name, "formula": r.formula, "file": fname,
                          "source_key": r.source_key, "quantitative": r.quantitative,
                          "y_kind": r.y_kind, "meta": r.meta})
        (d / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: Union[str, Path] = DEFAULT_DIR) -> "ReferenceLibrary":
        d = Path(directory)
        idx = d / "index.json"
        if not idx.exists():
            return cls()
        try:
            entries = json.loads(idx.read_text(encoding="utf-8"))
            lib = cls()
            for e in entries:
                with np.load(d / e["file"]) as z:
                    lib.add(ReferenceSpectrum(e["ref_id"], e["name"], e["formula"], z["nu"], z["k"],
                                              e["source_key"], e["quantitative"], e["y_kind"], e.get("meta", {})))
            return lib
        except (OSError, KeyError, ValueError) as exc:
            raise LibraryError(f"cannot read library in {d}: {exc}") from exc
