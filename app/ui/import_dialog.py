"""
Import dialog: shows what was read from a file and lets the user correct the guesses.

Everything the loader assumes - which column is x/y/sigma, the x unit, how y becomes
transmittance - is a visible control here, with a live summary of the result.  Nothing is
silently guessed: the auto-detection notes are displayed.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHeaderView, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from ..core.fitting import FitError, ObservedSpectrum, XUnit, YMode, build_observation
from ..core.tables import RawTable, suggest_spectrum_columns


class ImportDialog(QDialog):
    """Modal column/unit/scale picker.  After ``exec()`` read :attr:`observation`."""

    PREVIEW_ROWS = 8

    def __init__(self, table: RawTable, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import spectrum")
        self.resize(720, 520)
        self._table = table
        self._obs: Optional[ObservedSpectrum] = None

        lay = QVBoxLayout(self)
        notes = f"  ({'; '.join(table.notes)})" if table.notes else ""
        lay.addWidget(QLabel(f"<b>{table.source or 'data'}</b><br>{table.kind} file, "
                             f"{table.n_rows} rows × {table.n_cols} columns{notes}"))

        preview = QTableWidget(min(self.PREVIEW_ROWS, table.n_rows), table.n_cols)
        preview.setHorizontalHeaderLabels(table.names)
        preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for r in range(preview.rowCount()):
            for c in range(table.n_cols):
                preview.setItem(r, c, QTableWidgetItem(f"{table.data[r, c]:.6g}"))
        preview.setMaximumHeight(220)
        lay.addWidget(preview)

        guess = suggest_spectrum_columns(table)
        self._x = self._column_combo(guess["x"])
        self._y = self._column_combo(guess["y"])
        self._s = self._column_combo(guess["sigma"], allow_none=True)
        self._xu = QComboBox()
        for u in XUnit:
            self._xu.addItem("auto-detect" if u is XUnit.AUTO else u.value, u)
        self._ym = QComboBox()
        for m in YMode:
            self._ym.addItem("auto-detect" if m is YMode.AUTO else m.value, m)

        form = QFormLayout()
        form.addRow("X column", self._x)
        form.addRow("Y column", self._y)
        form.addRow("Uncertainty column", self._s)
        form.addRow("X unit", self._xu)
        form.addRow("Y values are", self._ym)
        lay.addLayout(form)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary, 1)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        lay.addWidget(self._buttons)

        for combo in (self._x, self._y, self._s, self._xu, self._ym):
            combo.currentIndexChanged.connect(self._rebuild)
        self._rebuild()

    def _column_combo(self, selected: Optional[int], allow_none: bool = False) -> QComboBox:
        combo = QComboBox()
        if allow_none:
            combo.addItem("(none)", None)
        for i, name in enumerate(self._table.names):
            combo.addItem(f"{i}: {name}", i)
        combo.setCurrentIndex(0 if selected is None else combo.findData(selected))
        return combo

    @property
    def observation(self) -> Optional[ObservedSpectrum]:
        return self._obs

    def _rebuild(self) -> None:
        """Re-interpret the file with the current choices and summarise the outcome."""
        try:
            self._obs = build_observation(
                self._table, self._x.currentData(), self._y.currentData(), self._s.currentData(),
                self._xu.currentData(), self._ym.currentData())
        except (FitError, ValueError) as exc:
            self._obs = None
            self._summary.setText(f"<span style='color:#d9534f'>Cannot use these settings: {exc}</span>")
        else:
            o = self._obs
            unit = "cm⁻¹" if o.unit.value == "wavenumber" else "µm"
            notes = "<br>".join(f"• {n}" for n in o.meta.get("import_notes", []))
            self._summary.setText(
                f"<b>{o.x.size} points</b>, x = {o.x.min():.5g} – {o.x.max():.5g} {unit}, "
                f"transmittance {o.y.min():.4g} – {o.y.max():.4g}<br>{notes}")
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self._obs is not None)
