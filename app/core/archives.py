"""
ECAS online archive clients (Qt-free, network access injectable for tests).

NASA Exoplanet Archive (TAP)
    :func:`fetch_planet` queries the ``ps`` (Planetary Systems) table with ``default_flag = 1`` and
    returns a :class:`PlanetRecord` carrying every value *with the literature reference the archive
    gives for it* (``pl_refname`` / ``st_refname``).  Column names were checked against the archive's
    own TAP schema (``/TAP/tables``).  Published transmission spectra are not fetched
    automatically: the archive's Atmospheric Spectroscopy Table is the authoritative place to
    browse and download them (it produces a wget script / files you then open with *Load…*), and its
    TAP column names were not verified, so no unverified query is shipped.

MAST (TESS / JWST / HST)
    :func:`search_mast`, :func:`list_spectral_products`, :func:`download_products` wrap
    ``astroquery.mast`` (optional dependency).  TESS light curves are available through
    :func:`search_mast_lightcurves` and :func:`download_mast_lightcurve`. JWST per-integration
    spectra are available through :func:`search_mast_spectral_timeseries` and
    :func:`download_mast_spectral_timeseries`. The wrappers are covered by tests that inject a fake
    ``Observations`` object.

Everything fetched can be registered in a :class:`~app.core.sources.CitationLog`.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sources import CitationLog
from .transit import StellarParams, TransitParams

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
ATMOSPHERES_URL = "https://exoplanetarchive.ipac.caltech.edu/docs/atmospheres/atmospheres_home.html"
USER_AGENT = "ECAS/0.8 (exoplanet atmosphere analysis; research use)"
CACHE_TTL_S = 24 * 3600

PLANET_COLUMNS = (
    "pl_name", "hostname", "default_flag", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_tranmid",
    "pl_trandur", "pl_trandep", "pl_ratror", "pl_ratdor", "pl_imppar", "pl_orbincl", "pl_orbsmax",
    "pl_rade", "pl_radj", "pl_bmassj", "pl_eqt", "st_teff", "st_rad", "st_mass", "st_met", "sy_vmag",
    "sy_kmag", "sy_dist", "pl_ntranspec", "pl_refname", "st_refname", "pl_pubdate", "rowupdate",
)
_TEXT_COLUMNS = {"pl_name", "hostname", "pl_refname", "st_refname", "pl_pubdate", "rowupdate"}


class ArchiveError(RuntimeError):
    """Raised for network failures, empty results and missing optional dependencies."""


Fetcher = Callable[[str], str]


def fetch_text(url: str, timeout: float = 30.0, cache_dir: Optional[Path] = None) -> str:
    """HTTP GET returning text, with an optional on-disk cache (24 h) so repeat queries are instant."""
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".txt")
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_TTL_S:
            return cache_file.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ArchiveError(f"could not reach the archive ({exc}). Check your internet connection; "
                           "nothing was changed.") from exc
    if cache_file is not None:
        cache_file.write_text(text, encoding="utf-8")
    return text


def _adql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_tap_url(query: str, fmt: str = "csv") -> str:
    return f"{TAP_URL}?query={urllib.parse.quote_plus(query)}&format={fmt}"


def _clean_ref(raw: str) -> str:
    """The archive returns references as HTML anchors; keep the visible text and the link."""
    if not raw:
        return ""
    href = re.search(r'href=["\']?([^\s"\'>]+)', raw)
    text = re.sub(r"<[^>]+>", "", raw).strip()
    return f"{text} <{href.group(1)}>" if href and text else (text or raw)


@dataclass
class PlanetRecord:
    """One planet's archive row (default parameter set) with provenance."""

    values: Dict[str, Any]
    query: str
    url: str
    retrieved: str
    refs: Dict[str, str] = field(default_factory=dict)       # 'planet' / 'star' -> cleaned reference text

    def get(self, key: str, default: Any = None) -> Any:
        v = self.values.get(key)
        return default if v is None else v

    @property
    def name(self) -> str:
        return str(self.values.get("pl_name", ""))

    def to_stellar_params(self) -> Tuple[StellarParams, List[str]]:
        """Host-star inputs for the transit analysis; the list names what the archive lacked."""
        missing, d = [], StellarParams()
        def pick(col, default, label):
            v = self.values.get(col)
            if v is None:
                missing.append(label)
                return default
            return float(v)
        return StellarParams(pick("st_rad", d.radius_rsun, "R★"), pick("st_mass", d.mass_msun, "M★"),
                             pick("st_teff", d.teff_k, "Teff")), missing

    def summary_lines(self) -> List[str]:
        def fmt(col, label, unit="", prec=4):
            v = self.values.get(col)
            return f"{label:<22}{v:.{prec}g} {unit}".rstrip() if isinstance(v, float) else f"{label:<22}–"
        lines = [f"{self.name}   (host {self.values.get('hostname', '?')})",
                 fmt("pl_orbper", "Period", "d", 8), fmt("pl_tranmid", "Transit midpoint", "d", 10),
                 fmt("pl_trandur", "T14", "h"), fmt("pl_ratror", "Rp/R★"), fmt("pl_ratdor", "a/R★"),
                 fmt("pl_imppar", "Impact b"), fmt("pl_radj", "Rp", "R_Jup"), fmt("pl_orbsmax", "a", "AU"),
                 fmt("pl_eqt", "T_eq (archive)", "K", 5), fmt("st_teff", "Teff", "K", 5),
                 fmt("st_rad", "R★", "R☉"), fmt("st_mass", "M★", "M☉")]
        n = self.values.get("pl_ntranspec")
        if isinstance(n, float) and n > 0:
            lines.append(f"Published transmission spectra in the archive: {int(n)}  (see {ATMOSPHERES_URL})")
        lines += [f"Planet parameters: {self.refs.get('planet', '–')}", f"Stellar parameters: {self.refs.get('star', '–')}"]
        return lines

    def cite(self, log: CitationLog) -> None:
        log.record("nea_planetary_systems",
                   f"{self.name} (planet params: {self.refs.get('planet', '?')}; star params: "
                   f"{self.refs.get('star', '?')})", query=self.url, rows=1)


def _parse_csv_row(text: str) -> List[Dict[str, Any]]:
    rows = []
    for rec in csv.DictReader(io.StringIO(text)):
        row: Dict[str, Any] = {}
        for k, v in rec.items():
            if k is None:
                continue
            v = (v or "").strip()
            if k in _TEXT_COLUMNS:
                row[k] = v or None
            else:
                try:
                    row[k] = float(v) if v != "" else None
                except ValueError:
                    row[k] = v or None
        rows.append(row)
    return rows


def fetch_planet(name: str, fetch: Optional[Fetcher] = None, cache_dir: Optional[Path] = None) -> PlanetRecord:
    """Default-parameter-set row for ``name`` (exact archive name, e.g. ``'WASP-39 b'``)."""
    name = name.strip()
    if not name:
        raise ArchiveError("enter a planet name, e.g. 'WASP-39 b'")
    query = (f"select {','.join(PLANET_COLUMNS)} from ps where pl_name = {_adql_quote(name)} "
             "and default_flag = 1")
    url = build_tap_url(query)
    text = (fetch or (lambda u: fetch_text(u, cache_dir=cache_dir)))(url)
    if text.lstrip().startswith("<") or "ERROR" in text[:200].upper():
        raise ArchiveError(f"the archive rejected the query: {text[:200].strip()}")
    rows = _parse_csv_row(text)
    if not rows:
        raise ArchiveError(f"no planet named {name!r} in the archive (names are exact, e.g. 'WASP-39 b'; "
                           "try 'Search names')")
    row = rows[0]
    return PlanetRecord(row, query, url, time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                        {"planet": _clean_ref(row.get("pl_refname") or ""), "star": _clean_ref(row.get("st_refname") or "")})


def search_planet_names(prefix: str, limit: int = 15, fetch: Optional[Fetcher] = None,
                        cache_dir: Optional[Path] = None) -> List[str]:
    """Planet names starting with ``prefix`` (case-insensitive), for the 'did you mean' helper."""
    prefix = prefix.strip()
    if len(prefix) < 2:
        raise ArchiveError("type at least two characters")
    query = (f"select top {int(limit)} pl_name from ps where upper(pl_name) like "
             f"{_adql_quote(prefix.upper() + '%')} and default_flag = 1 order by pl_name")
    text = (fetch or (lambda u: fetch_text(u, cache_dir=cache_dir)))(build_tap_url(query))
    if text.lstrip().startswith("<") or "ERROR" in text[:200].upper():
        raise ArchiveError(f"the archive rejected the query: {text[:200].strip()}")
    return [r["pl_name"] for r in _parse_csv_row(text) if r.get("pl_name")]


# ================================================================== MAST (optional)
SPECTRAL_SUFFIXES = ("_x1dints.fits", "_x1d.fits", "_whtlt.fits")
MAST_COLUMNS = ("obs_id", "obs_collection", "instrument_name", "filters", "target_name", "proposal_id",
                "proposal_pi", "obs_title", "dataproduct_type", "calib_level", "t_min", "t_exptime",
                "provenance_name", "distance", "obsid")


def _observations(observations: Any = None):
    if observations is not None:
        return observations
    try:
        from astroquery.mast import Observations
    except ImportError as exc:
        raise ArchiveError("MAST access needs astroquery:  pip install astroquery") from exc
    return Observations


def search_mast(target: str, collection: str = "JWST", observations: Any = None,
                dataproduct_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Observations of ``target`` in ``collection`` (JWST/HST/...), as plain dictionaries."""
    obs = _observations(observations)
    criteria = {"obs_collection": collection, "target_name": target}
    if dataproduct_type is not None:
        criteria["dataproduct_type"] = dataproduct_type
    try:
        table = obs.query_criteria(**criteria)
    except Exception as exc:                                  # astroquery raises many types
        raise ArchiveError(f"MAST search failed: {exc}") from exc
    cols = [c for c in MAST_COLUMNS if c in getattr(table, "colnames", MAST_COLUMNS)]
    return [{c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in cols} for row in table]


def search_mast_lightcurves(target: str, observations: Any = None) -> List[Dict[str, Any]]:
    """Search public TESS time-series observations for a host star."""
    obs = _observations(observations)
    try:
        table = obs.query_object(target, radius="0.01 deg")
    except Exception as exc:
        raise ArchiveError(f"MAST could not resolve or search {target!r}: {exc}") from exc
    mask = ((table["obs_collection"] == "TESS") & (table["dataproduct_type"] == "timeseries"))
    candidates = table[mask]
    try:
        products = obs.get_product_list(candidates) if len(candidates) else []
    except Exception as exc:
        raise ArchiveError(f"could not list TESS light-curve products for {target!r}: {exc}") from exc

    def is_lightcurve(product):
        try:
            filename = str(product["productFilename"]).lower()
            subgroup = str(product["productSubGroupDescription"]).upper()
        except (KeyError, IndexError, TypeError):
            return False
        return filename.endswith("lc.fits") and subgroup.endswith("LC")

    available_obs = {str(product["obsID"]) for product in products if is_lightcurve(product)}
    rows = []
    available = [column for column in MAST_COLUMNS if column in table.colnames]
    for row in candidates:
        if str(row["obsid"]) not in available_obs:
            continue
        record = {}
        for column in available:
            value = row[column]
            if np.ma.is_masked(value):
                value = None
            elif hasattr(value, "item"):
                value = value.item()
            record[column] = value
        if record.get("obs_id"):
            rows.append(record)
    if not rows:
        raise ArchiveError(f"MAST has no TESS time-series observations for {target!r}")
    return rows


def search_mast_spectral_timeseries(target: str, observations: Any = None) -> List[Dict[str, Any]]:
    """Find JWST calibrated per-integration spectra suitable for transmission extraction."""
    obs = _observations(observations)
    try:
        table = obs.query_object(target, radius="0.01 deg")
        candidates = table[table["obs_collection"] == "JWST"]
        products = obs.get_product_list(candidates) if len(candidates) else []
    except Exception as exc:
        raise ArchiveError(f"MAST spectral time-series search failed for {target!r}: {exc}") from exc

    observations_by_id = {str(row["obsid"]): row for row in candidates}
    results = []
    for product in products:
        try:
            filename = str(product["productFilename"])
            if not filename.lower().endswith("_x1dints.fits"):
                continue
            observation = observations_by_id.get(str(product["obsID"]))
            if observation is None:
                continue
            results.append({
                "obs_id": str(observation["obs_id"]),
                "obsid": str(observation["obsid"]),
                "target_name": str(observation["target_name"]),
                "instrument_name": str(observation["instrument_name"]),
                "proposal_id": str(observation["proposal_id"]),
                "t_min": float(observation["t_min"]),
                "t_exptime": float(observation["t_exptime"]),
                "filename": filename,
                "data_uri": str(product["dataURI"]),
                "size": int(product["size"]),
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchiveError(f"MAST returned incomplete spectral product metadata for {target!r}: {exc}") from exc
    if not results:
        raise ArchiveError(f"MAST has no JWST x1dints spectral time-series products for {target!r}")
    return sorted(results, key=lambda row: (row["t_min"], row["filename"]))


def list_lightcurve_products(obs_id: str, observations: Any = None):
    """List FITS light-curve products (rather than target pixel files) for a TESS observation."""
    obs = _observations(observations)
    try:
        table = obs.query_criteria(obs_collection="TESS", obs_id=obs_id)
        products = obs.get_product_list(table)
    except Exception as exc:
        raise ArchiveError(f"could not list light-curve products for {obs_id}: {exc}") from exc
    def value(product, key):
        try:
            return product[key]
        except (KeyError, IndexError, TypeError):
            return ""

    keep = [str(value(p, "productSubGroupDescription")).upper().endswith("LC")
            and str(value(p, "productFilename")).lower().endswith("lc.fits") for p in products]
    return products[keep] if hasattr(products, "colnames") else [p for p, include in zip(products, keep) if include]


def download_mast_lightcurve(obs_id: str, dest: Path, observations: Any = None) -> Path:
    """Download and return the first available TESS light-curve FITS product for ``obs_id``."""
    products = list_lightcurve_products(obs_id, observations)
    if not products:
        raise ArchiveError(f"MAST has no TESS light-curve FITS product for observation {obs_id}")
    cached = _find_cached_product(products, Path(dest), (obs_id,))
    if cached is not None:
        return cached
    paths = download_products(products, dest, observations)
    downloaded = next((path for path in paths if path.is_file()), None)
    if downloaded is None:
        raise ArchiveError(f"MAST did not provide a readable local file for observation {obs_id}")
    return downloaded


def download_mast_spectral_timeseries(product: Dict[str, Any], dest: Path,
                                      observations: Any = None) -> Path:
    """Download one selected JWST x1dints product by its MAST data URI."""
    obs = _observations(observations)
    filename = Path(str(product.get("filename", ""))).name
    data_uri = str(product.get("data_uri", ""))
    if not filename.lower().endswith("_x1dints.fits") or not data_uri.startswith("mast:JWST/product/"):
        raise ArchiveError("selected MAST product is not a valid JWST x1dints file")
    Path(dest).mkdir(parents=True, exist_ok=True)
    local_path = Path(dest) / filename
    if local_path.is_file() and local_path.stat().st_size > 0:
        return local_path
    try:
        status, message, _ = obs.download_file(data_uri, local_path=str(local_path), verbose=False)
    except Exception as exc:
        raise ArchiveError(f"JWST spectrum download failed for {filename}: {exc}") from exc
    if status != "COMPLETE" or not local_path.is_file():
        raise ArchiveError(f"MAST did not complete the download for {filename}: {message or status}")
    return local_path


def _find_cached_product(products: Sequence[Any], dest: Path,
                         preferred_path_parts: Sequence[str] = ()) -> Optional[Path]:
    """Reuse one unambiguous, non-empty downloaded product already under the archive cache."""
    if not dest.is_dir():
        return None
    filenames = set()
    for product in products:
        try:
            filename = str(product["productFilename"])
        except (KeyError, IndexError, TypeError):
            continue
        if filename:
            filenames.add(Path(filename).name)
    candidates = [path for filename in filenames for path in dest.rglob(filename)
                  if path.is_file() and path.stat().st_size > 0]
    if not candidates:
        return None
    preferred = tuple(str(part).lower() for part in preferred_path_parts if str(part))
    matching = [path for path in candidates
                if any(part in str(path).lower() for part in preferred)]
    if len(matching) == 1:
        return matching[0]
    return candidates[0] if len(candidates) == 1 else None


def list_spectral_products(obs_id: str, observations: Any = None):
    """Spectral/time-series products of one observation (x1dints, x1d, whtlt)."""
    obs = _observations(observations)
    try:
        table = obs.query_criteria(obs_collection="JWST", obs_id=obs_id)
        products = obs.get_product_list(table)
    except Exception as exc:
        raise ArchiveError(f"could not list products for {obs_id}: {exc}") from exc
    keep = [str(p["productFilename"]).lower().endswith(SPECTRAL_SUFFIXES) for p in products]
    return products[keep] if hasattr(products, "colnames") else [p for p, include in zip(products, keep) if include]


def download_products(products: Sequence[Any], dest: Path, observations: Any = None) -> List[Path]:
    """Download selected products (one or a few at a time, as MAST recommends)."""
    obs = _observations(observations)
    Path(dest).mkdir(parents=True, exist_ok=True)
    try:
        manifest = obs.download_products(products, download_dir=str(dest), verbose=False)
    except Exception as exc:
        raise ArchiveError(f"download failed: {exc}") from exc
    return [Path(str(p)) for p in manifest["Local Path"]]


# ================================================================ fit vs archive
def compare_with_archive(result, record: PlanetRecord, tolerance: float = 0.05) -> List[str]:
    """
    Human-readable comparison of a :class:`~app.core.transit.TransitResult` with the archive's
    default parameters.  A value is flagged when it differs by more than ``tolerance`` (fractional)
    *and* by more than 3x the fit's own 1-sigma error where one exists - so a flag means a real
    disagreement, not just a different noise realisation.
    """
    p, e = result.params, result.param_errors
    rows = [("Period (d)", p.period, record.get("pl_orbper"), e.get("period")),
            ("Rp/R★", p.rp_rs, record.get("pl_ratror"), e.get("rp_rs")),
            ("a/R★", p.a_rs, record.get("pl_ratdor"), e.get("a_rs")),
            ("Impact b", p.b, record.get("pl_imppar"), e.get("b")),
            ("T14 (h)", result.duration_hours, record.get("pl_trandur"), None)]
    lines = []
    for label, fit, arch, err in rows:
        if arch is None:
            lines.append(f"{label:<11} fit {fit:<10.5g} archive –")
            continue
        diff = fit - arch
        rel = abs(diff) / abs(arch) if arch else float("inf")
        real = rel > tolerance and (err is None or abs(diff) > 3.0 * err)
        lines.append(f"{label:<11} fit {fit:<10.5g} archive {arch:<10.5g} Δ {100 * diff / arch if arch else 0:+6.1f} %"
                     + ("   ⚠ differs" if real else "   ✓ consistent"))
    lines.append(f"Archive source: {record.refs.get('planet') or '–'} (retrieved {record.retrieved})")
    return lines


def transit_params_from_archive(record: PlanetRecord, t_mid_bjd: float, time_offset_bjd: float) -> TransitParams:
    """
    Transit geometry from the archive's ephemeris for a data set whose midpoint is ``t_mid_bjd`` (BJD)
    and whose time axis is ``BJD - time_offset_bjd``.  Used for single-transit observations (e.g. one JWST
    visit) where the period cannot be searched for.  Needs period, mid-transit time, Rp/R★ and a/R★.
    """
    need = {"pl_orbper": "period", "pl_tranmid": "mid-transit time", "pl_ratror": "Rp/R★", "pl_ratdor": "a/R★"}
    missing = [label for col, label in need.items() if record.get(col) is None]
    if missing:
        raise ArchiveError(f"the archive has no {', '.join(missing)} for {record.name}; analyse the light "
                           "curve yourself or enter the ephemeris")
    period, tranmid = float(record.get("pl_orbper")), float(record.get("pl_tranmid"))
    epoch = round((t_mid_bjd - tranmid) / period)
    return TransitParams(tranmid + epoch * period - time_offset_bjd, period, float(record.get("pl_ratror")),
                         float(record.get("pl_ratdor")), float(record.get("pl_imppar") or 0.0))
