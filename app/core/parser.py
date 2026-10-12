"""
ECAS ORCA output parser.

Harvests the fundamental vibrational modes (wavenumber in cm^-1 and IR intensity
in km/mol) from the ``IR SPECTRUM`` block of an ORCA ``.out`` file.

Robustness strategy
-------------------
1. The block is located dynamically: the *last* line that is exactly
   ``IR SPECTRUM`` which actually yields rows (geometry-optimisation + frequency
   jobs and numerical-frequency jobs can print several).
2. Its end is the first ``"\\n*"`` (the asterisk footnote / banner lines ORCA prints
   after the table), or end-of-file if there is none.
3. Inside the block, columns are mapped from the header text, so ORCA 4 style
   (``T**2 (km/mol)``) and ORCA 5/6 style (``eps  Int  T**2``) both work.
4. A soft guard stops at the first non-table line after rows have started, so a
   following block that lacks an asterisk terminator cannot leak in.
5. Windows line endings, scientific notation, imaginary (negative) frequencies and
   the zero-frequency translation/rotation modes are all handled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

__all__ = ["OrcaParseError", "IRPeakTable", "OrcaIRParser", "load_peaks"]

MOCK_SCHEME = "mock://"

_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?"
_NUM_RE = re.compile(_FLOAT)
_TITLE_RE = re.compile(r"^[ \t]*IR SPECTRUM[ \t]*$", re.MULTILINE)
_TERMINATOR_RE = re.compile(r"\n\*")                       # "\n*" boundary
_ROW_RE = re.compile(r"^[ \t]*(?P<mode>\d+)[ \t]*:(?P<rest>.*)$")
_COLUMN_RE = re.compile(r"(?<![\w*])(freq|eps|Int|T\*\*2|TX|TY|TZ)(?![\w*])")
_RULE_CHARS = set("-= ")


class OrcaParseError(ValueError):
    """Raised when no usable IR SPECTRUM table can be found."""


@dataclass
class IRPeakTable:
    """
    Clean, array-based result of one parse.

    Attributes
    ----------
    wavenumber_cm1, intensity_km_mol, mode_index
        Equal-length 1-D arrays sorted by ascending wavenumber.
    meta
        Parse bookkeeping (source, column layout, counts of filtered modes).
    """

    wavenumber_cm1: NDArray[np.float64]
    intensity_km_mol: NDArray[np.float64]
    mode_index: NDArray[np.int64]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.wavenumber_cm1.size)

    def as_dict(self) -> Dict[str, Any]:
        """Nested-dictionary view: ``{"peaks": {...arrays...}, "meta": {...}}``."""
        return {
            "peaks": {
                "wavenumber_cm1": self.wavenumber_cm1,
                "intensity_km_mol": self.intensity_km_mol,
                "mode_index": self.mode_index,
            },
            "meta": dict(self.meta),
        }


class OrcaIRParser:
    """
    Parser for ORCA ``IR SPECTRUM`` tables.

    Parameters
    ----------
    min_wavenumber_cm1
        Modes at or below this value (translations, rotations, imaginary modes)
        are discarded.
    drop_inactive
        Discard IR-silent modes (intensity <= 0); they cannot contribute to the
        optical depth and only inflate the line matrix.
    """

    def __init__(self, min_wavenumber_cm1: float = 1.0, drop_inactive: bool = True):
        self.min_wavenumber_cm1 = float(min_wavenumber_cm1)
        self.drop_inactive = bool(drop_inactive)

    # ------------------------------------------------------------------ public
    def parse_file(self, path: Union[str, Path]) -> IRPeakTable:
        """Read an ORCA ``.out`` file from disk and parse it."""
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        table = self.parse_text(text)
        table.meta["source"] = str(path)
        return table

    def parse_text(self, text: str) -> IRPeakTable:
        """Parse the raw text of an ORCA output."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        titles = list(_TITLE_RE.finditer(text))
        if not titles:
            raise OrcaParseError("No 'IR SPECTRUM' block found.")

        for title in reversed(titles):                 # last block wins
            block = self.extract_block(text, title.end())
            rows, layout = self._parse_block(block)
            if rows:
                return self._build_table(rows, layout)
        raise OrcaParseError("'IR SPECTRUM' block found, but it contains no mode rows.")

    @staticmethod
    def extract_block(text: str, start: int) -> str:
        """Slice from ``start`` to the first ``"\\n*"`` boundary (or EOF)."""
        end = _TERMINATOR_RE.search(text, start)
        return text[start : end.start() if end else len(text)]

    # ----------------------------------------------------------------- private
    @staticmethod
    def _detect_layout(block: str) -> Dict[str, Any]:
        """Read the header lines (before the first row) to map column names."""
        header_lines: List[str] = []
        for line in block.split("\n"):
            if _ROW_RE.match(line):
                break
            header_lines.append(line)
        header = "\n".join(header_lines)
        names = _COLUMN_RE.findall(header)             # e.g. freq eps Int T**2 TX TY TZ
        intensity_idx: Optional[int] = None
        if "Int" in names:
            intensity_idx = names.index("Int")
        elif "T**2" in names and "km/mol" in header:
            intensity_idx = names.index("T**2")
        return {"columns": names, "intensity_idx": intensity_idx}

    def _parse_block(self, block: str):
        layout = self._detect_layout(block)
        rows: List[tuple] = []
        started = False
        for line in block.split("\n"):
            m = _ROW_RE.match(line)
            if m:
                started = True
                nums = [float(v.replace("D", "E").replace("d", "e"))
                        for v in _NUM_RE.findall(m.group("rest"))]
                idx = layout["intensity_idx"]
                if idx is None:                         # header missing: fall back by width
                    idx = 2 if len(nums) >= 7 else 1
                if len(nums) > max(idx, 0):
                    rows.append((int(m.group("mode")), nums[0], nums[idx]))
            elif started and line.strip() and not set(line.strip()) <= _RULE_CHARS:
                break                                   # soft guard: table is over
        return rows, layout

    def _build_table(self, rows, layout) -> IRPeakTable:
        arr = np.asarray(rows, dtype=np.float64)        # columns: mode, freq, intensity
        mode, freq, inten = arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2]

        keep = freq > self.min_wavenumber_cm1
        n_low = int(np.count_nonzero(~keep))
        n_imag = int(np.count_nonzero(freq < 0.0))
        if self.drop_inactive:
            keep &= inten > 0.0
        n_inactive = int(np.count_nonzero(~keep)) - n_low
        if not keep.any():
            raise OrcaParseError("All modes were filtered out (no IR-active modes).")

        order = np.argsort(freq[keep], kind="stable")
        return IRPeakTable(
            wavenumber_cm1=freq[keep][order],
            intensity_km_mol=inten[keep][order],
            mode_index=mode[keep][order],
            meta={
                "columns": layout["columns"],
                "n_rows": int(arr.shape[0]),
                "n_kept": int(keep.sum()),
                "n_low_or_imaginary": n_low,
                "n_imaginary": n_imag,
                "n_inactive": n_inactive,
            },
        )


def load_peaks(source: Union[str, Path], parser: Optional[OrcaIRParser] = None) -> IRPeakTable:
    """
    Resolve a gas ``source`` string to a peak table.

    * ``mock://H2O`` - synthesise an ORCA-style text with the mock generator and
      push it through the *real* regex parser (so startup exercises the same code
      path as a genuine file).
    * anything else - treated as a filesystem path to an ORCA ``.out`` file.
    """
    parser = parser or OrcaIRParser()
    src = str(source)
    if src.startswith(MOCK_SCHEME):
        from . import mock_data                         # lazy: test-only dependency

        key = src[len(MOCK_SCHEME):]
        table = parser.parse_text(mock_data.render_mock_orca_output(key))
        table.meta["source"] = src
        return table
    return parser.parse_file(src)
