"""
ECAS "Identify" workspace: explain the active observation as a combination of reference spectra.

Left:  reference library (simulator gases + imported spectra), candidate ranking, and run controls.
Right: observed vs model with each detected molecule's own contribution, residuals, and tabs for
       the per-molecule table, per-feature attribution (who explains this peak, do they overlap?),
       warnings, and the references behind every spectrum used.

All numbers come from :mod:`app.core.identify`; this module only draws and wires.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
    QScrollArea,
    QListWidgetItem, QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget, QInputDialog, QSizePolicy,
)

from ..core.engine import SpectrumEngine
from ..core.fitting import FitError, ObservedSpectrum, find_features
from ..core.identify import DecompositionResult, IdentifyError, decompose, rank_references
from ..core.library import (DEFAULT_DIR, LibraryError, ReferenceLibrary, hapi_cross_section_reference,
                            import_exomol_transitions, import_jcamp, import_nist_jcamp)
from ..core.sources import SOURCES, CitationLog
from .theme import ACCENT, ERR, MUTED, OK, WARN, CollapsibleSection, muted, primary

_COLORS = ("#4FC3F7", "#FF8A65", "#81C784", "#BA68C8", "#FFD54F", "#4DB6AC", "#F06292", "#A1887F",
           "#90A4AE", "#DCE775")
_SOURCE_TAG = {"mock": "MOCK", "orca": "ORCA", "nist_webbook": "NIST", "nist_quantitative_ir": "NIST-QIR",
               "hitran2020": "HITRAN", "hapi": "HAPI", "exomol": "ExoMol"}


class IdentifyPanel(QWidget):
    """Controls + charts + tables for multi-molecule identification."""

    def __init__(self, engine: SpectrumEngine, get_observation: Callable[[], Optional[ObservedSpectrum]],
                 citations: CitationLog, library_dir: Optional[Path] = DEFAULT_DIR, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._get_obs = get_observation
        self._citations = citations
        self._library_dir = Path(library_dir) if library_dir else None
        self._user = ReferenceLibrary()
        if self._library_dir is not None:
            try:
                self._user = ReferenceLibrary.load(self._library_dir)
            except LibraryError:
                pass
        self.result: Optional[DecompositionResult] = None
        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._colors: Dict[str, str] = {}
        self._region: Optional[pg.LinearRegionItem] = None

        # ---------------- charts
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Wavenumber (cm⁻¹)")
        self._plot.setLabel("left", "Transmittance")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.addLegend(offset=(-10, 10), labelTextColor=MUTED)
        self._plot.getViewBox().setMouseEnabled(x=True, y=False)
        self._data_item = self._plot.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None,
                                          symbolBrush=(255, 210, 77, 170), name="observed")
        self._model_item = self._plot.plot(pen=pg.mkPen("#ffffff", width=2), name="model (total)")
        self._resid_plot = pg.PlotWidget()
        self._resid_plot.setLabel("left", "Residual (σ)")
        self._resid_plot.showGrid(x=True, y=True, alpha=0.2)
        self._resid_plot.setMaximumHeight(140)
        self._resid_plot.setXLink(self._plot)
        self._resid_item = self._resid_plot.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None,
                                                 symbolBrush=(255, 200, 0, 150))
        for y in (3.0, -3.0):
            self._resid_plot.addLine(y=y, pen=pg.mkPen(ERR, style=Qt.PenStyle.DashLine))
        self._resid_plot.setYRange(-8, 8, padding=0.05)

        # ---------------- result tabs
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(["Molecule", "Formula", "Amplitude", "± 1σ", "Significance", "Status", "Source"])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.currentCellChanged.connect(lambda r, *_: self._emphasise_row(r))
        self._features = QListWidget()
        self._features.currentRowChanged.connect(self._on_feature_selected)
        self._notes = QPlainTextEdit()
        self._notes.setReadOnly(True)
        self._refs = QPlainTextEdit()
        self._refs.setReadOnly(True)
        self._tabs = QTabWidget()
        for widget, title in ((self._table, "Molecules"), (self._features, "Feature attribution"),
                              (self._notes, "Warnings && notes"), (self._refs, "References")):
            self._tabs.addTab(widget, title)

        right = QSplitter(Qt.Orientation.Vertical)
        charts = QWidget()
        c_lay = QVBoxLayout(charts)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.addWidget(self._plot, 4)
        c_lay.addWidget(self._resid_plot, 1)
        right.addWidget(charts)
        right.addWidget(self._tabs)
        right.setSizes([520, 300])

        root = QHBoxLayout(self)
        root.addWidget(self._build_controls())
        root.addWidget(right, 1)
        self._rebuild_list()

    # ------------------------------------------------------------------- controls
    def _build_controls(self) -> QWidget:
        box = QWidget()
        box.setMinimumWidth(360)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Identify molecules")
        title.setProperty("heading", True)
        sub = QLabel("Explain the active observation as a combination of reference spectra. "
                     "Overlapping molecules are separated where the data allow, and flagged where they cannot be.")
        sub.setWordWrap(True)
        muted(sub)
        lay.addWidget(title)
        lay.addWidget(sub)

        self._dataset_label = QLabel("")
        self._dataset_label.setWordWrap(True)
        lay.addWidget(self._dataset_label)

        lib = CollapsibleSection("Reference spectra")
        self._ref_list = QListWidget()
        self._ref_list.setMinimumHeight(190)
        self._ref_list.setToolTip("Tick the molecules to consider. Fewer, relevant candidates give a more reliable answer.")
        self._ref_list.itemChanged.connect(lambda _i: self._update_summary())
        row = QHBoxLayout()
        b_imp, b_rm = QPushButton("Import JCAMP-DX…"), QPushButton("Remove")
        b_imp.clicked.connect(self._on_import)
        b_rm.clicked.connect(self._on_remove)
        row.addWidget(b_imp, 1)
        row.addWidget(b_rm)
        row_sources = QHBoxLayout()
        b_nist = QPushButton("Browse NIST…")
        b_nist.setToolTip("Fetch an evaluated IR or UV/Vis JCAMP spectrum from the NIST Chemistry WebBook by CAS/WebBook ID.")
        b_nist.clicked.connect(self._on_nist)
        b_exomol = QPushButton("Import ExoMol…")
        b_exomol.setToolTip("Import an ExoMol .states/.trans pair as a documented line-strength shape.")
        b_exomol.clicked.connect(self._on_exomol)
        b_hapi = QPushButton("Import HAPI CSV…")
        b_hapi.setToolTip("Import sampled HAPI wavenumber/cross-section output; columns: cm-1, cm2 molecule-1.")
        b_hapi.clicked.connect(self._on_hapi)
        for button in (b_nist, b_exomol, b_hapi):
            row_sources.addWidget(button)
        row2 = QHBoxLayout()
        b_all, b_none = QPushButton("All"), QPushButton("None")
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none.clicked.connect(lambda: self._set_all(False))
        row2.addWidget(b_all)
        row2.addWidget(b_none)
        b_rank = QPushButton("Auto-rank candidates")
        b_rank.setToolTip("Screen the checked library against this observation, then keep the strongest templates for the joint fit.")
        b_rank.clicked.connect(self._rank_candidates)
        row2.addWidget(b_rank, 1)
        self._lib_label = QLabel("")
        muted(self._lib_label)
        self._lib_label.setWordWrap(True)
        for w in (self._ref_list,):
            lib.body_layout().addWidget(w)
        lib.body_layout().addLayout(row)
        lib.body_layout().addLayout(row_sources)
        lib.body_layout().addLayout(row2)
        lib.body_layout().addWidget(self._lib_label)
        lay.addWidget(lib)

        run = CollapsibleSection("Run")
        form = QHBoxLayout()
        form.addWidget(QLabel("Detected if ≥"))
        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(1.0, 20.0)
        self._sigma.setValue(3.0)
        self._sigma.setSuffix(" σ")
        form.addWidget(self._sigma)
        self._chk_cont = QCheckBox("Fit continuum (level + tilt)")
        self._chk_cont.setChecked(True)
        self._chk_feat = QCheckBox("Attribute each spectral feature")
        self._chk_feat.setChecked(True)
        self._btn_run = QPushButton("Identify molecules")
        primary(self._btn_run)
        self._btn_run.clicked.connect(self.run)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(64)
        self._status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        run.body_layout().addLayout(form)
        for w in (self._chk_cont, self._chk_feat, self._btn_run, self._status):
            run.body_layout().addWidget(w)
        lay.addWidget(run)
        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(box)
        scroll.setMinimumWidth(380)
        return scroll

    # ----------------------------------------------------------------- library
    def _all_refs(self) -> ReferenceLibrary:
        """Simulator gases at the *current* conditions, plus imported references."""
        return ReferenceLibrary.from_engine(self._engine).merged(self._user)

    def _rebuild_list(self) -> None:
        previous = {self._ref_list.item(i).data(Qt.ItemDataRole.UserRole): self._ref_list.item(i).checkState()
                    for i in range(self._ref_list.count())}
        self._ref_list.blockSignals(True)
        self._ref_list.clear()
        for ref in self._all_refs():
            tag = _SOURCE_TAG.get(ref.source_key, ref.source_key)
            item = QListWidgetItem(f"{ref.formula or ref.ref_id:<8} {ref.name}   [{tag}]"
                                   + ("" if ref.quantitative else "  · relative"))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(previous.get(ref.ref_id, Qt.CheckState.Checked))
            item.setData(Qt.ItemDataRole.UserRole, ref.ref_id)
            item.setToolTip(f"{ref.name}\nsource: {SOURCES[ref.source_key].title}\n"
                            f"coverage: {ref.coverage_cm1[0]:.0f}-{ref.coverage_cm1[1]:.0f} cm⁻¹\n"
                            f"{'quantitative (units known)' if ref.quantitative else 'shape only: amplitudes are relative'}")
            self._ref_list.addItem(item)
        self._ref_list.blockSignals(False)
        self._update_summary()

    def _checked_ids(self) -> List[str]:
        return [self._ref_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self._ref_list.count())
                if self._ref_list.item(i).checkState() == Qt.CheckState.Checked]

    def _set_all(self, on: bool) -> None:
        self._ref_list.blockSignals(True)
        for i in range(self._ref_list.count()):
            self._ref_list.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._ref_list.blockSignals(False)
        self._update_summary()

    def _rank_candidates(self) -> None:
        obs = self._get_obs()
        if obs is None:
            self._status.setText(f"<span style='color:{WARN}'>Load an observation first.</span>")
            return
        lib = self._all_refs()
        ids = [i for i in self._checked_ids() if i in lib]
        try:
            ranked = rank_references(obs, lib, ids)
        except IdentifyError as exc:
            self._status.setText(f"<span style='color:{ERR}'>Candidate ranking failed: {exc}</span>")
            return
        keep = {item.ref_id for item in ranked[:min(12, max(4, int(np.sqrt(len(ranked)) * 3)))]}
        self._ref_list.blockSignals(True)
        for i in range(self._ref_list.count()):
            item = self._ref_list.item(i)
            rid = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if rid in keep else Qt.CheckState.Unchecked)
        self._ref_list.blockSignals(False)
        if ranked:
            preview = ", ".join(f"{r.name} ({r.score:.1f}σ)" for r in ranked[:5])
            self._status.setText(f"<span style='color:{OK}'>Kept {len(keep)} candidates.</span> Strongest: {preview}")
        else:
            self._status.setText(f"<span style='color:{WARN}'>No references covered enough usable points.</span>")
        self._update_summary()

    def _update_summary(self) -> None:
        ids = self._checked_ids()
        obs = self._get_obs()
        dataset_text = (f"<b>Dataset:</b> {obs.meta.get('name', 'observation')}  ({obs.x.size} points)"
                        if obs is not None else
                        "<span style='color:%s'>No dataset loaded - load or fetch one on the Spectroscopy tab.</span>"
                        % WARN)
        is_emission = obs is not None and obs.meta.get("spectrum_kind") == "emission_flux"
        if is_emission:
            dataset_text += ("<br><span style='color:%s'><b>Emission spectrum:</b> "
                             "displayed below; molecular line identification is not yet enabled for emitted flux.</span>") % WARN
        if obs is not None and obs.meta.get("source") == "transmission-spectrum":
            dataset_text += ("<br><span style='color:%s'><b>Exploratory comparison only:</b> "
                             "measured transit depths were mapped to relative pseudo-transmittance; "
                             "gas amplitudes are not atmospheric abundances.</span>") % WARN
        self._dataset_label.setText(
            dataset_text)
        n_user = len([i for i in ids if i in self._user])
        self._lib_label.setText(f"{len(ids)} selected  ·  {n_user} imported.  Imported spectra are saved to "
                                f"{self._library_dir if self._library_dir else '(session only)'}.")
        self._btn_run.setEnabled(obs is not None and not is_emission and bool(ids))
        if is_emission:
            self._show_emission_observation(obs)
        elif self.result is None and obs is not None:
            self._plot.setLabel("left", "Transmittance / normalized spectrum")
            self._model_item.setVisible(True)
            self._data_item.setData(obs.wavenumber_cm1, obs.y)

    def refresh(self) -> None:
        """Call when the dataset or simulator conditions changed."""
        self._rebuild_list()

    def _show_emission_observation(self, obs: ObservedSpectrum) -> None:
        """Display emission data without applying absorption/transmission physics."""
        self.result = None
        nu = obs.wavenumber_cm1
        order = np.argsort(nu)
        x, y = nu[order], obs.y[order]
        self._data_item.setData(x, y)
        self._model_item.clear()
        self._model_item.setVisible(False)
        self._resid_item.clear()
        for item in self._curves.values():
            self._plot.removeItem(item)
        self._curves.clear()
        self._table.setRowCount(0)
        self._features.clear()
        self._plot.setLabel("bottom", "Wavenumber (cm⁻¹)")
        self._plot.setLabel("left", f"Relative {obs.meta.get('flux_density_kind', 'emission').upper()}")
        pad = max(float(np.ptp(x)) * 0.02, 1.0)
        y_lo, y_hi = float(np.nanmin(y)), float(np.nanmax(y))
        y_pad = max((y_hi - y_lo) * 0.08, 1e-6)
        self._plot.setRange(xRange=(float(x.min()) - pad, float(x.max()) + pad),
                            yRange=(y_lo - y_pad, y_hi + y_pad), padding=0.0)
        self._notes.setPlainText(
            "Emission spectrum displayed as relative flux. The current reference library and "
            "mixture fitter model absorption/transmission, so no molecular emission identification "
            "is claimed yet. Fit a blackbody continuum on the Spectroscopy tab for an exploratory baseline.")

    def _on_import(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Import reference spectra", "",
                                                "JCAMP-DX (*.jdx *.dx *.jcm *.jcamp);;All files (*)")
        for path in paths:
            self.import_reference(path)

    def import_reference(self, path: str) -> bool:
        try:
            ref = import_jcamp(path)
        except LibraryError as exc:
            self._status.setText(f"<span style='color:{ERR}'>Could not import {Path(path).name}: {exc}</span>")
            return False
        self._user.add(ref)
        self._citations.record(ref.source_key, f"{ref.name} ({ref.meta.get('cas') or ref.ref_id}) from {Path(path).name}")
        if self._library_dir is not None:
            try:
                self._user.save(self._library_dir)
            except OSError:
                pass
        self._rebuild_list()
        self._status.setText(f"Imported {ref.name} ({'quantitative' if ref.quantitative else 'shape only'}).")
        return True

    def _add_reference(self, ref, detail: str, query: str = "") -> bool:
        self._user.add(ref)
        self._citations.record(ref.source_key, detail, query=query)
        if self._library_dir is not None:
            try:
                self._user.save(self._library_dir)
            except OSError:
                pass
        self._rebuild_list()
        self._status.setText(f"Imported {ref.name} ({'quantitative' if ref.quantitative else 'shape only'}).")
        return True

    def _on_nist(self) -> None:
        identifier, accepted = QInputDialog.getText(self, "NIST WebBook", "CAS/WebBook ID:")
        if not accepted or not identifier.strip():
            return
        spectrum, accepted = QInputDialog.getItem(
            self, "NIST WebBook", "Spectrum collection:", ["IR", "UV/Vis"], 0, False)
        if not accepted:
            return
        index, accepted = QInputDialog.getInt(self, "NIST WebBook", "Spectrum index:", 0, 0, 999)
        if not accepted:
            return
        try:
            ref = import_nist_jcamp(identifier, index, spectrum=spectrum.lower().replace("/", "-"))
        except LibraryError as exc:
            self._status.setText(f"<span style='color:{ERR}'>NIST import failed: {exc}</span>")
            return
        self._add_reference(ref, f"{ref.name} from NIST WebBook", ref.meta.get("query", ""))

    def _on_exomol(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select ExoMol states and transitions", "",
                                                "ExoMol files (*.states *.trans *.bz2 *.gz);;All files (*)")
        states = next((p for p in paths if ".states" in Path(p).name), None)
        transitions = next((p for p in paths if ".trans" in Path(p).name), None)
        if not states or not transitions:
            self._status.setText("Select both an ExoMol .states file and its .trans file.")
            return
        molecule, accepted = QInputDialog.getText(self, "ExoMol line list", "Molecule/formula:")
        if not accepted or not molecule.strip():
            return
        temperature, accepted = QInputDialog.getDouble(self, "ExoMol line list", "Reference temperature (K):", 1500.0, 1.0, 100000.0, 1)
        if not accepted:
            return
        try:
            ref = import_exomol_transitions(transitions, states, molecule.strip(), temperature)
        except LibraryError as exc:
            self._status.setText(f"<span style='color:{ERR}'>ExoMol import failed: {exc}</span>")
            return
        self._add_reference(ref, f"{ref.name} ExoMol line list", ref.meta.get("transitions_file", ""))

    def _on_hapi(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import HAPI cross-section CSV", "", "CSV (*.csv *.txt);;All files (*)")
        if not path:
            return
        molecule, accepted = QInputDialog.getText(self, "HAPI cross-section", "Molecule/formula:")
        if not accepted or not molecule.strip():
            return
        temperature, accepted = QInputDialog.getDouble(self, "HAPI cross-section", "Temperature (K):", 296.0, 1.0, 100000.0, 1)
        if not accepted:
            return
        try:
            table = np.genfromtxt(path, delimiter=",", comments="#", names=True)
            if table.dtype.names and len(table.dtype.names) >= 2:
                nu, cross_section = (np.asarray(table[name], dtype=float) for name in table.dtype.names[:2])
            else:
                raw = np.loadtxt(path, delimiter=",")
                nu, cross_section = raw[:, 0], raw[:, 1]
            ref = hapi_cross_section_reference(molecule.strip(), nu, cross_section, temperature)
        except (OSError, ValueError, IndexError, LibraryError) as exc:
            self._status.setText(f"<span style='color:{ERR}'>HAPI import failed: {exc}</span>")
            return
        self._add_reference(ref, f"{ref.name} HAPI cross-section", ref.meta.get("query", ""))

    def _on_remove(self) -> None:
        item = self._ref_list.currentItem()
        if item is None:
            return
        rid = item.data(Qt.ItemDataRole.UserRole)
        if rid in self._user:
            self._user.remove(rid)
            if self._library_dir is not None:
                try:
                    self._user.save(self._library_dir)
                except OSError:
                    pass
            self._rebuild_list()
        else:
            self._status.setText("Simulator gases cannot be removed; untick them instead.")

    # --------------------------------------------------------------------- run
    def run(self) -> Optional[DecompositionResult]:
        obs = self._get_obs()
        if obs is None:
            self._status.setText(f"<span style='color:{WARN}'>Load an observation first.</span>")
            return None
        if obs.meta.get("spectrum_kind") == "emission_flux":
            self._status.setText(
                f"<span style='color:{WARN}'>Molecular identification currently supports "
                "transmission/absorption data only.</span>")
            return None
        lib = self._all_refs()
        ids = [i for i in self._checked_ids() if i in lib]
        try:
            feats = find_features(obs, min_snr=5.0) if self._chk_feat.isChecked() else []
        except FitError:
            feats = []
        try:
            res = decompose(obs, lib, ids, fit_continuum=self._chk_cont.isChecked(),
                            detection_sigma=self._sigma.value(), features=feats)
        except IdentifyError as exc:
            self._status.setText(f"<span style='color:{ERR}'>{exc}</span>")
            return None
        self.result = res
        for c in res.components:                                      # provenance for everything that was used
            src = lib.get(c.ref_id)
            if src.source_key in ("mock", "orca"):
                self._citations.record(src.source_key, "simulator band model (shapes, not measurements)")
        self._show(res, obs)
        det = res.detected()
        self._status.setText(
            f"<span style='color:{OK if det else WARN}'><b>{len(det)} molecule(s) detected</b></span> "
            f"({', '.join(c.formula or c.name for c in det) or 'none'}).  RMS residual (T) = {res.rms_t:.4f}  ·  reduced χ² = {res.reduced_chi2:.2f}.")
        return res

    # --------------------------------------------------------------- rendering
    def _show(self, res: DecompositionResult, obs: ObservedSpectrum) -> None:
        for item in self._curves.values():
            self._plot.removeItem(item)
            try:
                self._plot.getPlotItem().legend.removeItem(item)
            except Exception:
                pass
        self._curves.clear()
        if self._region is not None:
            self._plot.removeItem(self._region)
            self._region = None
        self._data_item.setData(res.nu, np.exp(-res.tau_obs))
        self._model_item.setData(res.nu, np.exp(-res.tau_model))
        for i, c in enumerate(res.components):
            if not c.detected:
                continue
            color = self._colors.setdefault(c.ref_id, _COLORS[len(self._colors) % len(_COLORS)])
            idx = res.ref_ids.index(c.ref_id)
            item = self._plot.plot(res.nu, np.exp(-res.tau_by_ref[:, idx]), pen=pg.mkPen(color, width=1.4),
                                   name=c.formula or c.name)
            self._curves[c.ref_id] = item
        self._plot.getViewBox().setLimits(xMin=float(res.nu.min()) - 50, xMax=float(res.nu.max()) + 50,
                                          yMin=-0.05, yMax=1.1)
        self._plot.setRange(xRange=(float(res.nu.min()), float(res.nu.max())), yRange=(0.0, 1.05), padding=0.02)
        self._resid_item.setData(res.nu, np.clip((res.tau_obs - res.tau_model) / res.sigma_tau, -8, 8))

        comps = sorted(res.components, key=lambda c: -c.significance)
        self._table.setRowCount(len(comps))
        for r, c in enumerate(comps):
            cells = [c.name, c.formula, f"{c.amplitude:.4g}" + ("" if c.quantitative else " (rel.)"),
                     f"{c.uncertainty:.2g}", f"{c.significance:.1f} σ", "DETECTED" if c.detected else "not significant",
                     _SOURCE_TAG.get(c.source_key, c.source_key)]
            for col, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setData(Qt.ItemDataRole.UserRole, c.ref_id)
                if col == 5:
                    it.setForeground(QColor(OK if c.detected else MUTED))
                if col == 0 and c.detected:
                    it.setForeground(QColor(self._colors.get(c.ref_id, ACCENT)))
                self._table.setItem(r, col, it)
        self._table.resizeColumnsToContents()

        self._features.clear()
        for a in res.attributions:
            who = ", ".join(f"{s:.0%} {k}" for k, s in sorted(a.shares.items(), key=lambda kv: -kv[1]) if s >= 0.05)
            line = f"{a.nu_peak_cm1:7.1f} cm⁻¹ ({1e4 / a.nu_peak_cm1:5.2f} µm)  →  {who or 'unassigned'}"
            if a.overlapped:
                line += "   ⚠ overlap"
            if a.unexplained > 0.15:
                line += f"   ({a.unexplained:.0%} unexplained)"
            self._features.addItem(line)
        self._notes.setPlainText("\n".join(res.summary_lines()[:1] + [f"• {w}" for w in res.warnings]
                                           + ([""] + [f"Degeneracy: condition number {res.condition_number:.1f}"]
                                              if res.condition_number > 30 else [])))
        self._refs.setPlainText(self._citations.references_markdown())

    def _emphasise_row(self, row: int) -> None:
        if row < 0 or self._table.item(row, 0) is None:
            return
        chosen = self._table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        for rid, item in self._curves.items():
            item.setPen(pg.mkPen(self._colors[rid], width=3.5 if rid == chosen else 1.0))

    def _on_feature_selected(self, row: int) -> None:
        if self.result is None or not (0 <= row < len(self.result.attributions)):
            return
        a = self.result.attributions[row]
        if self._region is None:
            self._region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(255, 235, 59, 70), pen=pg.mkPen(None))
            self._region.setZValue(-5)
            self._plot.addItem(self._region)
        self._region.setRegion((a.nu_lo_cm1, a.nu_hi_cm1))
        pad = max(4 * (a.nu_hi_cm1 - a.nu_lo_cm1), 40.0)
        self._plot.setXRange(a.nu_peak_cm1 - pad, a.nu_peak_cm1 + pad, padding=0)
