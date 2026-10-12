"""
ECAS table reader - turns "whatever the user has" into a :class:`RawTable`.

Handles, without the caller caring which:

* delimited text: comma / semicolon / tab / whitespace, ``#`` ``%`` ``//`` comments,
  free-text metadata lines, one or more header lines, decimal commas, ``D`` exponents,
  UTF-8 (with BOM) or Latin-1, ragged rows (the dominant column count wins);
* FITS binary/ASCII tables (e.g. JWST ``x1d``, TESS light curves) via astropy;
* JCAMP-DX ``XYDATA`` / ``XYPOINTS`` (uncompressed AFFN only).

Qt-free.  Column *interpretation* (units, normalisation) lives in
:mod:`app.core.fitting` and :mod:`app.core.transit`.
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

PathLike = Union[str, Path]


class TableError(ValueError):
    """Raised when a file cannot be read as numeric columns."""


@dataclass
class RawTable:
    """Numeric columns plus whatever descriptive text the file offered."""

    names: List[str]
    data: NDArray[np.float64]                 # (rows, columns)
    header: str = ""                          # free text used for unit hints
    kind: str = "text"                        # text | fits | jcamp
    source: str = ""
    notes: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)     # JCAMP-DX header tags (TITLE, CASREGISTRYNO, ...)

    @property
    def n_rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.data.shape[1])

    def column(self, index: int) -> NDArray[np.float64]:
        return self.data[:, index]


_COMMENT = ("#", "%", "//", "!")
_SPLIT_ANY = re.compile(r"[,;\t ]+")
_FLOAT_TOKEN = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?$|^[-+]?(?:nan|inf)$", re.I)


# ------------------------------------------------------------------------ dispatch
def read_table(path: PathLike) -> RawTable:
    """Read ``path`` as numeric columns; raises :class:`TableError` if impossible."""
    path = Path(path)
    try:
        head = path.read_bytes()[:2880]
    except OSError as exc:
        raise TableError(f"cannot read {path}: {exc}") from exc
    if head.startswith(b"SIMPLE") or path.suffix.lower() in (".fits", ".fit", ".fts"):
        return _read_fits(path)
    text = _decode(path.read_bytes())
    if "##XYDATA" in text.upper() or "##XYPOINTS" in text.upper():
        return _read_jcamp(text, str(path))
    return _read_text(text, str(path))


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise TableError("unsupported text encoding")


# --------------------------------------------------------------------------- text
def _numeric(tokens: Sequence[str]) -> bool:
    return bool(tokens) and all(_FLOAT_TOKEN.match(t) for t in tokens)


def _read_text(text: str, source: str) -> RawTable:
    structured = _read_structured_csv(text, source)
    if structured is not None:
        return structured
    lines = [ln.strip() for ln in text.split("\n")]
    notes: List[str] = []

    # decimal comma?  (numbers like 1234,56 in a file that separates with ; or tab)
    body = [ln for ln in lines if ln and not ln.startswith(_COMMENT)]
    sample = "\n".join(body[:200])
    decimal_comma = bool(re.search(r"\d,\d", sample)) and not re.search(r"\d\.\d", sample) \
        and bool(re.search(r"[;\t]", sample) or re.search(r"\d,\d+\s+[-+]?\d", sample))
    if decimal_comma:
        notes.append("decimal comma detected")

    rows, meta_lines, header_line = [], [], ""
    for ln in lines:
        if not ln:
            continue
        if ln.startswith(_COMMENT):
            meta_lines.append(ln.lstrip("#%/! ").strip())
            continue
        probe = ln.replace(",", ".") if decimal_comma else ln
        tokens = [t for t in _SPLIT_ANY.split(probe.replace("\"", "")) if t]
        if _numeric(tokens):
            rows.append([float(t.replace("D", "E").replace("d", "e")) for t in tokens])
        elif not rows:
            header_line = ln                                   # last text line before data
            meta_lines.append(ln)
    if not rows:
        raise TableError("no numeric rows found")

    width = Counter(len(r) for r in rows).most_common(1)[0][0]
    ragged = sum(len(r) != width for r in rows)
    if ragged:
        notes.append(f"{ragged} rows with a different column count ignored")
    data = np.array([r for r in rows if len(r) == width], dtype=np.float64)
    if width < 2:
        raise TableError("need at least two numeric columns")

    names = _names_from_header(header_line, width) or [f"col{i}" for i in range(width)]
    return RawTable(names, data, "\n".join(meta_lines[-6:]), "text", source, notes)


def _read_structured_csv(text: str, source: str) -> Optional[RawTable]:
    """Read a headed CSV while retaining empty and non-numeric fields as NaN."""
    if "," not in text[:4096]:
        return None
    try:
        records = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        return None
    for header_index, header in enumerate(records):
        names = [cell.strip() for cell in header]
        if len(names) < 2 or not any(names):
            continue
        if any(name and _FLOAT_TOKEN.match(name) for name in names):
            continue
        width = len(names)
        data: List[List[float]] = []
        for row in records[header_index + 1:]:
            if len(row) != width:
                continue
            converted: List[float] = []
            numeric_count = 0
            for cell in row:
                value = cell.strip()
                try:
                    number = float(value.replace("D", "E").replace("d", "e")) if value else np.nan
                except ValueError:
                    number = np.nan
                else:
                    numeric_count += 1
                converted.append(number)
            if np.isfinite(converted[0]) and numeric_count >= 2:
                data.append(converted)
        if len(data) >= 8:
            notes = ["empty and non-numeric CSV cells preserved as missing values"]
            return RawTable(names, np.asarray(data, dtype=np.float64), ", ".join(names),
                            "text", source, notes)
    return None


def _names_from_header(line: str, width: int) -> Optional[List[str]]:
    if not line:
        return None
    for splitter in (r"\t", r";", r",", r"\s{2,}", r"\s+"):
        parts = [p.strip().strip("\"'") for p in re.split(splitter, line.lstrip("#% ")) if p.strip()]
        if len(parts) == width:
            return parts
    return None


# ---------------------------------------------------------------------------- FITS
def _read_fits(path: Path) -> RawTable:
    try:
        from astropy.io import fits
    except ImportError as exc:                                  # pragma: no cover
        raise TableError("reading FITS needs astropy (pip install astropy)") from exc
    try:
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                cols = getattr(hdu, "columns", None)
                if cols is None or hdu.data is None:
                    continue
                names, arrays, header = [], [], []
                for col in cols:
                    arr = np.asarray(hdu.data[col.name])
                    arr = arr.reshape(-1) if arr.ndim > 1 and arr.shape[0] == 1 else arr
                    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.number):
                        continue
                    unit = f" [{col.unit}]" if col.unit else ""
                    names.append(f"{col.name}{unit}")
                    arrays.append(arr.astype(np.float64))
                    header.append(names[-1])
                if len(arrays) >= 2 and len({a.size for a in arrays}) == 1:
                    extname = hdu.header.get("EXTNAME", "")
                    return RawTable(names, np.column_stack(arrays), " ".join(header), "fits", str(path),
                                    [f"FITS table {extname!r}" if extname else "FITS table"])
    except OSError as exc:
        raise TableError(f"cannot read FITS file: {exc}") from exc
    raise TableError("no numeric table with >= 2 columns found in the FITS file")


# -------------------------------------------------------------------------- JCAMP
def _read_jcamp(text: str, source: str) -> RawTable:
    tags: Dict[str, str] = {}
    xs: List[float] = []
    ys: List[float] = []
    mode = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if line.startswith("##"):
            key, _, val = line[2:].partition("=")
            key = re.sub(r"[\s_\-]", "", key).upper()
            tags[key] = val.strip()
            mode = key if key in ("XYDATA", "XYPOINTS") else ("" if key != "END" else "")
            continue
        if mode not in ("XYDATA", "XYPOINTS") or not line:
            continue
        tokens = [t for t in re.split(r"[,;\s]+", line) if t]
        if not _numeric(tokens):
            raise TableError("compressed JCAMP-DX (SQZ/DIF) is not supported; export as AFFN/CSV")
        vals = [float(t) for t in tokens]
        if mode == "XYPOINTS":
            xs += vals[0::2]
            ys += vals[1::2][: len(vals[0::2])]
        else:                                                    # X, Y1, Y2, ... with constant DELTAX
            x0, rest = vals[0], vals[1:]
            xfac = float(tags.get("XFACTOR", "1") or 1.0)
            dx = float(tags.get("DELTAX", "0") or 0.0)
            if not dx:                                           # derive from FIRSTX/LASTX/NPOINTS
                try:
                    dx = (float(tags["LASTX"]) - float(tags["FIRSTX"])) / (float(tags["NPOINTS"]) - 1.0)
                except (KeyError, ValueError, ZeroDivisionError):
                    dx = 0.0
            for i, v in enumerate(rest):
                xs.append(x0 * xfac + i * dx)                   # DELTAX is already in real X units
                ys.append(v)
    if len(xs) < 2:
        raise TableError("JCAMP-DX file holds no XY data")
    yfac = float(tags.get("YFACTOR", "1") or 1.0)
    xu, yu = tags.get("XUNITS", ""), tags.get("YUNITS", "")
    return RawTable([f"x [{xu}]", f"y [{yu}]"], np.column_stack([xs, np.asarray(ys) * yfac]),
                    f"{tags.get('TITLE', '')} XUNITS={xu} YUNITS={yu}", "jcamp", source, tags=dict(tags))


# --------------------------------------------------------------- column suggestion
def _first_match(names: Sequence[str], pattern: str, skip: Sequence[int] = ()) -> Optional[int]:
    rx = re.compile(pattern, re.I)
    for i, n in enumerate(names):
        if i not in skip and rx.search(n):
            return i
    return None


def _first_finite_match(table: RawTable, pattern: str, skip: Sequence[int] = ()) -> Optional[int]:
    for index in range(table.n_cols):
        if index not in skip and re.search(pattern, table.names[index], re.I):
            if np.isfinite(table.column(index)).any():
                return index
    return None


def _first_not_none(*values: Optional[int]) -> Optional[int]:
    """First value that is not None (index 0 is a valid answer, so no ``or`` chains)."""
    return next((v for v in values if v is not None), None)


def suggest_spectrum_columns(table: RawTable) -> Dict[str, Optional[int]]:
    """Best guess for ``{"x": i, "y": j, "sigma": k or None}`` of a spectrum table."""
    x = _first_not_none(
        _first_finite_match(table, r"wave|lambda|λ|freq|nu\b|wn\b|^x\b|micron|angstrom"), 0)
    sigma = _first_finite_match(table, r"err|sigma|unc|noise|std|σ", skip=(x,))
    if sigma is not None and not np.any(np.isfinite(table.column(sigma)) & (table.column(sigma) > 0.0)):
        sigma = None
    skipped = tuple(i for i in (x, sigma) if i is not None)
    y = _first_not_none(
        _first_finite_match(table, r"^f(?:lam|nu)(?:$|[\s\[])", skip=skipped),
        _first_finite_match(
            table, r"transm|flux|intens|reflect|absorb|depth|signal|^y\b|^t\b|^t\s*[%(\[]",
            skip=skipped),
    )
    if y is None:
        y = next(i for i in range(table.n_cols)
                 if i not in (x, sigma) and np.isfinite(table.column(i)).any())
    return {"x": x, "y": y, "sigma": sigma}


def suggest_lightcurve_columns(table: RawTable) -> Dict[str, Optional[int]]:
    """Best guess for a light curve; prefers PDCSAP over SAP flux (TESS/Kepler files)."""
    n = table.names
    t = _first_not_none(_first_match(n, r"^time|bjd|btjd|mjd|\bjd\b|^t\b|^days?\b|epoch"), 0)
    y = _first_not_none(_first_match(n, r"pdcsap_flux(?!_err)", skip=(t,)),
                        _first_match(n, r"^sap_flux(?!_err)", skip=(t,)),
                        _first_match(n, r"flux|^f\b|mag|rel|norm", skip=(t,)))
    if y is None:
        y = next(i for i in range(table.n_cols) if i != t)
    err = _first_match(n, r"err|sigma|unc", skip=(t, y))
    return {"t": t, "y": y, "sigma": err}
