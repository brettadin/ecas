"""
ECAS main window: slider pane (left) + pyqtgraph canvas with unit tabs (right).

This module is a thin view/controller.  It owns no physics: every number comes from
:class:`app.core.engine.SpectrumEngine`.  The hot path (``_refresh``) is

    engine.update_conditions(...)   # O(1)
    engine.transmittance(unit, m)   # cached (N,G) @ (G,) mat-vec + exp
    curve.setData(x, y)             # hand the array to pyqtgraph

Slider events only mutate state and *schedule* a refresh through a zero-delay
single-shot timer, so a burst of events during a drag collapses into one redraw per
event-loop turn.
"""
from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QListWidgetItem, QFileDialog, QMessageBox, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QMainWindow, QPushButton, QScrollArea, QSlider, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from ..core.config import GasSpec, PhysicsDefaults
from ..core.engine import AtmosphericConditions, SpectrumEngine, Unit
from ..core import export as ecas_export
from ..core.project import ProjectError, ProjectState, load_project, save_project
from ..core.fitting import (
    Anomaly, CandidateLibrary, FitError, FitResult, ObservedSpectrum, build_observation,
    find_anomalies, find_features, fit_multipliers,
)
from ..core.emission import BlackbodyContinuumFit, EmissionFitError, fit_blackbody_continuum
from ..core.sources import CitationLog
from ..core.tables import TableError, read_table, suggest_spectrum_columns
from .import_dialog import ImportDialog
from ..core.physics import LineShape
from .mappings import LinearMapping, LogMapping
from .identify_panel import IdentifyPanel
from .sources_dialog import SourcesDialog
from .theme import CollapsibleSection, primary
from .transit_panel import TransitPanel



class LabeledSlider(QWidget):
    """Title + float-valued slider + live value read-out.  Emits ``value_changed(float)``."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        title: str,
        mapping,
        value: float,
        suffix: str = "",
        decimals: int = 2,
        color: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._mapping, self._suffix, self._decimals = mapping, suffix, decimals

        self._title = QLabel(title)
        if color:
            self._title.setStyleSheet(f"color: {color}; font-weight: 600;")
        self._readout = QLabel()
        self._readout.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._readout.setMinimumWidth(92)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, mapping.steps)
        self._slider.setValue(mapping.to_pos(value))
        self._slider.valueChanged.connect(self._on_slider)

        top = QHBoxLayout()
        top.addWidget(self._title, 1)
        top.addWidget(self._readout)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(1)
        lay.addLayout(top)
        lay.addWidget(self._slider)
        self._update_readout(self.value())

    def value(self) -> float:
        return self._mapping.to_value(self._slider.value())

    def set_value(self, value: float) -> None:
        """Move the handle (emits ``value_changed`` through the slider signal)."""
        self._slider.setValue(self._mapping.to_pos(value))

    def _on_slider(self, pos: int) -> None:
        v = self._mapping.to_value(pos)
        self._update_readout(v)
        self.value_changed.emit(v)

    def _update_readout(self, v: float) -> None:
        text = f"{v:.{self._decimals}f}" if abs(v) >= 0.01 or v == 0 else f"{v:.2e}"
        self._readout.setText(f"{text} {self._suffix}".strip())


@dataclass
class _Canvas:
    """Everything the updater needs for one unit tab (all pre-created, never rebuilt)."""

    plot: pg.PlotWidget
    x: NDArray[np.float64]
    master: pg.PlotDataItem
    emission_continuum: pg.PlotDataItem
    layers: Tuple[pg.PlotDataItem, ...]
    data_items: list = field(default_factory=list)          # one curve per visible dataset
    regions: list = field(default_factory=list)             # red anomaly bars


@dataclass
class _Dataset:
    """One loaded observation shown in the dataset list."""

    uid: int
    name: str
    obs: ObservedSpectrum
    visible: bool = True


class MainWindow(QMainWindow):
    """Multi-gas, unit-switching transmittance dashboard."""

    SLIDER_PRESSURE = LogMapping(1e-4, 1e2, 1000)       # atm
    SLIDER_TEMPERATURE = LinearMapping(50.0, 1500.0, 1450)   # K
    SLIDER_ENVELOPE = LinearMapping(0.0, 30.0, 300)     # cm^-1 HWHM

    _RESID_CLIP = 15.0                                  # residual panel shows +-15 sigma
    _TAB_UNITS: Tuple[Unit, ...] = (Unit.WAVENUMBER, Unit.WAVELENGTH)
    _TAB_TITLES = {Unit.WAVENUMBER: "Wavenumber (cm⁻¹)", Unit.WAVELENGTH: "Wavelength (µm)"}

    def __init__(
        self,
        engine: SpectrumEngine,
        defaults: PhysicsDefaults,
        notes: Sequence[str] = (),
        enable_3d: bool = True,
        library_dir=None,
        planet_fetch=None,
    ):
        super().__init__()
        self.setWindowTitle("ECAS - Exoplanetary Characterization & Astrochemistry Suite")
        self.resize(1360, 780)

        self._engine = engine
        self._defaults = defaults
        self._conditions = engine.conditions
        self._multipliers = engine.default_multipliers()     # mutated in place by sliders
        self._show_layers = False
        self._gas_sliders: Dict[str, LabeledSlider] = {}
        self._datasets: List[_Dataset] = []
        self._active_uid: Optional[int] = None
        self._next_uid = 1
        self._syncing = False
        self._show_model = True
        self._feature_mode = False
        self._fit: Optional[FitResult] = None
        self._emission_fit: Optional[BlackbodyContinuumFit] = None
        self._emission_fit_uid: Optional[int] = None
        self._anomalies: list = []
        self._candidates: list = []
        self._project_path: Optional[str] = None
        try:
            self._library: Optional[CandidateLibrary] = CandidateLibrary.from_engine(engine)
        except (OSError, ValueError, KeyError):
            self._library = None

        # Coalesces bursts of slider events into one refresh per event-loop turn.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._refresh)

        self._canvases: Dict[Unit, _Canvas] = {u: self._make_canvas(u) for u in self._TAB_UNITS}
        self._tabs = QTabWidget()
        for u in self._TAB_UNITS:
            self._tabs.addTab(self._canvases[u].plot, self._TAB_TITLES[u])
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._resid_plot = pg.PlotWidget()
        self._resid_plot.setLabel("left", "Residual (σ)")
        self._resid_plot.showGrid(x=True, y=True, alpha=0.25)
        self._resid_curve = self._resid_plot.plot(pen=None, symbol="o", symbolSize=3,
                                                  symbolPen=None, symbolBrush=(255, 200, 0, 170))
        self._thr_lines = tuple(
            self._resid_plot.addLine(y=y, pen=pg.mkPen("#ff4d4d", style=Qt.PenStyle.DashLine))
            for y in (3.0, -3.0))
        self._resid_plot.setMouseEnabled(x=True, y=False)
        self._resid_plot.getViewBox().setLimits(yMin=-self._RESID_CLIP - 1, yMax=self._RESID_CLIP + 1)
        self._resid_plot.setYRange(-self._RESID_CLIP, self._RESID_CLIP, padding=0.05)
        self._resid_plot.hide()
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._tabs)
        right.addWidget(self._resid_plot)
        right.setStretchFactor(0, 4)
        right.setStretchFactor(1, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 960])

        # Top-level workspaces: spectroscopy (Phases 1-4) and transit photometry + 3-D (Phase 5).
        self._citations = CitationLog()
        self._transit = TransitPanel(engine, enable_3d=enable_3d, citations=self._citations, planet_fetch=planet_fetch)
        self._transit.observation_ready.connect(self._on_transit_observation)
        self._identify = IdentifyPanel(engine, lambda: self._obs, self._citations, library_dir=library_dir)
        self._workspaces = QTabWidget()
        self._workspaces.addTab(splitter, "Spectroscopy")
        self._workspaces.addTab(self._identify, "Identify")
        self._workspaces.addTab(self._transit, "Transit && 3D")
        self._workspaces.currentChanged.connect(lambda _i: self._identify.refresh()
                                                if self._workspaces.currentWidget() is self._identify else None)
        self.setCentralWidget(self._workspaces)
        self._build_menu()

        self._status = QLabel()
        self.statusBar().addWidget(self._status, 1)
        if notes:
            self.statusBar().showMessage("  |  ".join(notes), 15000)
        self._refresh()

    def _on_fit_blackbody(self) -> None:
        obs = self._obs
        if (obs is None or obs.meta.get("spectrum_kind") != "emission_flux"
                or obs.unit is not Unit.WAVELENGTH):
            return
        density_kind = obs.meta.get("flux_density_kind")
        if density_kind not in ("flam", "fnu"):
            self._warn("Blackbody fitting requires archive Fλ or Fν data.")
            return
        try:
            result = fit_blackbody_continuum(
                obs.x, obs.y, obs.sigma, density_kind=density_kind)
        except EmissionFitError as exc:
            self._warn(f"Emission continuum fit failed: {exc}")
            return
        self._emission_fit = result
        self._emission_fit_uid = self._active_uid
        for canvas in self._canvases.values():
            x = (result.wavelength_um if canvas is self._canvases[Unit.WAVELENGTH]
                 else 1e4 / result.wavelength_um)
            order = np.argsort(x)
            canvas.emission_continuum.setData(x[order], result.model[order])
        self._sync_observation_mode()
        self._fit_label.setText(
            f"Blackbody color temperature: {result.temperature_k:.0f} K; "
            f"normalized-flux RMS residual: {result.rms_residual:.4g}. "
            "Exploratory continuum only; molecular features and systematics are not modeled.")

    # ----------------------------------------------------------- construction
    def _make_canvas(self, unit: Unit) -> _Canvas:
        """Create the plot, the master curve and one hidden curve per gas (once)."""
        grid = self._engine.grid
        x = grid.axis(unit)
        lo, hi = grid.limits(unit)

        plot = pg.PlotWidget()
        plot.setLabel("bottom", unit.axis_label)
        plot.setLabel("left", "Transmittance / normalized spectrum")
        plot.showGrid(x=True, y=True, alpha=0.25)
        vb = plot.getViewBox()
        # Lock the viewport to the physical range; negative space is unreachable.
        vb.setLimits(xMin=lo, xMax=hi, yMin=0.0, yMax=1.05)
        vb.setRange(xRange=(lo, hi), yRange=(0.0, 1.05), padding=0.0)
        vb.setMouseEnabled(x=True, y=False)

        # NOTE: x is the same read-only array object every frame; skipFiniteCheck
        # avoids an O(N) NaN scan on every setData call.
        layers = tuple(
            plot.plot(pen=pg.mkPen(QColor(g.color), width=1.2), skipFiniteCheck=True)
            for g in self._engine.gases
        )
        for item in layers:
            item.setZValue(10)                  # overlay sits above the white master curve
            item.setVisible(False)
        master = plot.plot(pen=pg.mkPen("#ffffff", width=2), skipFiniteCheck=True)
        emission_continuum = plot.plot(
            pen=pg.mkPen("#4fc3f7", width=2), name="Blackbody continuum")
        emission_continuum.setVisible(False)
        plot.scene().sigMouseClicked.connect(partial(self._on_plot_clicked, unit))
        return _Canvas(plot, x, master, emission_continuum, layers)

    def _build_sidebar(self) -> QWidget:
        """Left pane: atmosphere controls + one dynamically generated slider per gas."""
        d = self._defaults
        atmos = CollapsibleSection("Atmosphere  ·  pressure, temperature, line shape", expanded=False)
        a_lay = atmos.body_layout()

        self._s_pressure = LabeledSlider("Pressure P", self.SLIDER_PRESSURE, d.pressure_atm, "atm", 3)
        self._s_temperature = LabeledSlider("Temperature T", self.SLIDER_TEMPERATURE, d.temperature_k, "K", 0)
        self._s_envelope = LabeledSlider("Band envelope HWHM", self.SLIDER_ENVELOPE, d.envelope_hwhm_cm1, "cm⁻¹", 1)
        self._s_pressure.value_changed.connect(self._on_pressure)
        self._s_temperature.value_changed.connect(self._on_temperature)
        self._s_envelope.value_changed.connect(self._on_envelope)

        self._shape = QComboBox()
        for s in LineShape:
            self._shape.addItem(s.value.capitalize(), s)
        self._shape.setCurrentIndex(list(LineShape).index(d.line_shape))
        self._shape.currentIndexChanged.connect(self._on_shape)

        for w in (self._s_pressure, self._s_temperature, self._s_envelope):
            a_lay.addWidget(w)
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Line shape"))
        shape_row.addWidget(self._shape, 1)
        a_lay.addLayout(shape_row)

        gases = CollapsibleSection("Gas layers  ·  column multipliers")
        g_lay = gases.body_layout()
        for g in self._engine.gases:                       # setup-time loop, not the updater
            idx = self._engine.gas_index[g.gas_id]
            s = LabeledSlider(
                f"{g.formula}  {g.name}",
                LinearMapping(0.0, g.multiplier_max, 2000),
                float(self._multipliers[idx]), "x", 2, color=g.color,
            )
            s.value_changed.connect(partial(self._on_gas, idx))
            self._gas_sliders[g.gas_id] = s
            g_lay.addWidget(s)

        self._layers_box = QCheckBox("Show individual gas layers")
        self._layers_box.toggled.connect(self._on_layers_toggled)
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self._reset)

        inverse = self._build_inverse_group()

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.addWidget(inverse)
        lay.addWidget(gases)
        lay.addWidget(atmos)
        lay.addWidget(self._layers_box)
        lay.addWidget(reset)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(body)
        scroll.setMinimumWidth(300)
        return scroll

    # ---------------------------------------------------------------- slots
    def _schedule(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _on_pressure(self, v: float) -> None:
        self._conditions = replace(self._conditions, pressure_atm=v)
        self._schedule()

    def _on_temperature(self, v: float) -> None:
        self._conditions = replace(self._conditions, temperature_k=v)
        self._schedule()

    def _on_envelope(self, v: float) -> None:
        self._conditions = replace(self._conditions, envelope_hwhm_cm1=v)
        self._schedule()

    def _on_shape(self, _index: int) -> None:
        self._conditions = replace(self._conditions, line_shape=self._shape.currentData())
        self._schedule()

    def _on_gas(self, idx: int, v: float) -> None:
        self._multipliers[idx] = v                           # in-place O(1) write
        self._schedule()

    def _on_layers_toggled(self, on: bool) -> None:
        self._show_layers = on
        for canvas in self._canvases.values():
            for item in canvas.layers:                       # GUI handles, not numerics
                item.setVisible(on)
        self._refresh()

    def _reset(self) -> None:
        d = self._defaults
        self._s_pressure.set_value(d.pressure_atm)
        self._s_temperature.set_value(d.temperature_k)
        self._s_envelope.set_value(d.envelope_hwhm_cm1)
        self._shape.setCurrentIndex(list(LineShape).index(d.line_shape))
        for g in self._engine.gases:
            self._gas_sliders[g.gas_id].set_value(g.default_multiplier)

    # --------------------------------------------------------------- updater
    def active_unit(self) -> Unit:
        return self._TAB_UNITS[self._tabs.currentIndex()]

    def _refresh(self) -> None:
        """
        The data updater.  Numerically loop-free: the only work is the cached
        mat-vec in the engine and one ``setData`` per visible curve.
        """
        t0 = time.perf_counter()
        unit = self.active_unit()
        canvas = self._canvases[unit]
        self._engine.update_conditions(self._conditions)
        canvas.master.setData(canvas.x, self._engine.transmittance(unit, self._multipliers))
        if self._show_layers:
            self._push_layers(unit, canvas)
        dt = (time.perf_counter() - t0) * 1e3
        self._status.setText(
            f"model + setData: {dt:5.2f} ms   |   N = {self._engine.grid.n_points}   "
            f"lines = {self._engine.n_lines}   gases = {self._engine.n_gases}   "
            f"profile rebuilds = {self._engine.profile_evaluations}"
        )

    def _push_layers(self, unit: Unit, canvas: _Canvas) -> None:
        """
        Optional overlay.  The (N, G) layer matrix is one broadcast ``exp``; the
        short loop below only hands each column to its own graphics item (one
        curve per colour), i.e. GUI-handle dispatch over <= G objects.
        """
        layers = self._engine.layer_transmittances(unit, self._multipliers)
        for col, item in enumerate(canvas.layers):
            item.setData(canvas.x, layers[:, col])

    # ===================================== Observations, identification and inverse fit
    _DATA_COLORS = ("#ffd24d", "#66d9ef", "#a6e22e", "#f92672", "#ae81ff", "#fd971f")

    def _build_inverse_group(self) -> QGroupBox:
        """Sidebar: dataset list, view/model options, identification, fit, anomaly list."""
        box = CollapsibleSection("Observations  ·  identify & fit")
        lay = box.body_layout()

        self._data_list = QListWidget()
        self._data_list.setMaximumHeight(96)
        self._data_list.setToolTip("Tick to show/hide a file.  The selected row is the one that is fitted.")
        self._data_list.currentRowChanged.connect(self._on_dataset_selected)
        self._data_list.itemChanged.connect(self._on_dataset_checked)
        lay.addWidget(self._data_list)

        row = QHBoxLayout()
        for text, slot in (("Load…", self._on_load_csv), ("Mock", self._on_mock_observation),
                           ("Remove", self._on_clear_observation)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)

        row_v = QHBoxLayout()
        b_zoom, b_reset = QPushButton("Zoom to data"), QPushButton("Reset view")
        b_zoom.clicked.connect(self._zoom_to_data)
        b_reset.clicked.connect(self._reset_view)
        row_v.addWidget(b_zoom)
        row_v.addWidget(b_reset)
        lay.addLayout(row_v)

        self._chk_model = QCheckBox("Show model curve")
        self._chk_model.setChecked(True)
        self._chk_model.toggled.connect(self._on_show_model)
        self._chk_baseline = QCheckBox("Fit continuum (level + tilt)")
        self._chk_baseline.setToolTip("Let the fit scale/tilt the continuum instead of assuming it is exactly 1.")
        self._chk_window = QCheckBox("Restrict fit to model range")
        self._chk_window.setChecked(True)
        self._chk_window.setToolTip("Only fit where the model has bands; avoids flagging unmodelled "
                                    "overtone/electronic features as unknown molecules.")
        for w in (self._chk_model, self._chk_baseline, self._chk_window):
            lay.addWidget(w)

        row2 = QHBoxLayout()
        self._btn_identify = QPushButton("Identify peaks (no model)")
        self._btn_identify.setToolTip("Find absorption peaks directly in the data and match them to known bands.")
        self._btn_identify.setEnabled(False)
        self._btn_identify.clicked.connect(self._on_identify)
        self._btn_fit = QPushButton("Fit gas multipliers")
        self._btn_fit.setEnabled(False)
        self._btn_fit.clicked.connect(self._on_fit)
        primary(self._btn_fit)
        row2.addWidget(self._btn_identify)
        row2.addWidget(self._btn_fit)
        lay.addLayout(row2)
        self._btn_blackbody = QPushButton("Fit blackbody emission continuum")
        self._btn_blackbody.setEnabled(False)
        self._btn_blackbody.setToolTip(
            "Fit color temperature and arbitrary scale to an archive Fλ/Fν spectrum; no molecular retrieval.")
        self._btn_blackbody.clicked.connect(self._on_fit_blackbody)
        lay.addWidget(self._btn_blackbody)

        form = QHBoxLayout()
        form.addWidget(QLabel("Anomaly"))
        self._thr_spin = QDoubleSpinBox()
        self._thr_spin.setRange(1.0, 10.0)
        self._thr_spin.setSingleStep(0.5)
        self._thr_spin.setValue(3.0)
        self._thr_spin.setSuffix(" σ")
        self._thr_spin.valueChanged.connect(lambda _v: self._analyse())
        form.addWidget(self._thr_spin)
        form.addWidget(QLabel("Peak SNR"))
        self._snr_spin = QDoubleSpinBox()
        self._snr_spin.setRange(2.0, 50.0)
        self._snr_spin.setSingleStep(1.0)
        self._snr_spin.setValue(5.0)
        form.addWidget(self._snr_spin)
        lay.addLayout(form)

        self._fit_label = QLabel("No observation loaded.")
        self._fit_label.setWordWrap(True)
        self._anom_header = QLabel("")
        self._anom_header.setStyleSheet("color: #ff5c5c; font-weight: 700;")
        self._anom_list = QListWidget()
        self._anom_list.setMaximumHeight(150)
        self._anom_list.setToolTip("Click an entry to highlight and zoom to it (you can also click the chart).")
        self._anom_list.currentRowChanged.connect(lambda row: self._highlight(row, zoom=True))
        self._cand_label = QLabel("")
        self._cand_label.setWordWrap(True)
        for w in (self._fit_label, self._anom_header, self._anom_list, self._cand_label):
            lay.addWidget(w)
        return box

    # ------------------------------------------------------------------ datasets
    @property
    def _obs(self) -> Optional[ObservedSpectrum]:
        ds = self._active_dataset()
        return ds.obs if ds else None

    def _active_dataset(self) -> Optional[_Dataset]:
        return next((d for d in self._datasets if d.uid == self._active_uid), None)

    def _on_load_csv(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load observed spectra", "",
            "Spectra (*.csv *.dat *.txt *.tsv *.fits *.fit *.jdx *.dx *.jcamp);;All files (*)")
        for path in paths:
            self.load_observation_file(path, interactive=True)

    def load_observation_file(self, path: str, interactive: bool = False) -> Optional[ObservedSpectrum]:
        """
        Read ``path`` and add it as a dataset.  Interactive mode shows the import dialog so
        columns, units and scaling can be corrected; otherwise the detected settings are used.
        """
        try:
            table = read_table(path)
            if interactive:
                dialog = ImportDialog(table, self)
                if dialog.exec() != dialog.DialogCode.Accepted or dialog.observation is None:
                    return None
                obs = dialog.observation
            else:
                cols = suggest_spectrum_columns(table)
                obs = build_observation(table, cols["x"], cols["y"], cols["sigma"])
        except (TableError, FitError, OSError) as exc:
            self._warn(f"Could not load {path}: {exc}")
            return None
        self.set_observation(obs)
        return obs

    def _on_mock_observation(self) -> None:
        """Synthesise a noisy observation with an injected, unmodelled SO2 absorber."""
        from ..core.mock_data import make_mock_observation

        truth = {"H2O": 3.0, "CO2": 2.0, "CH4": 0.8, "O3": 1.5, "N2O": 1.0}
        truth = {k: v for k, v in truth.items() if k in self._engine.gas_index}
        self._engine.update_conditions(self._conditions)
        self._citations.record("mock", "mock observation")
        self.set_observation(make_mock_observation(self._engine, self.active_unit(), truth),
                             name="MOCK observation")

    def set_observation(self, obs: ObservedSpectrum, name: Optional[str] = None) -> None:
        """Add a dataset, make it active and show it immediately (no fit required)."""
        uid, self._next_uid = self._next_uid, self._next_uid + 1
        label = name or obs.meta.get("name") or f"observation {uid}"
        obs.meta["name"] = str(label)                      # one name everywhere (list, Identify tab, reports)
        self._datasets.append(_Dataset(uid, str(label), obs))
        self._active_uid = uid
        self._after_dataset_change()
        self._zoom_to_data()
        notes = "; ".join(obs.meta.get("import_notes", []))
        self._fit_label.setText(f"{obs.x.size} points loaded. {notes}  "
                                "Use 'Identify peaks' or 'Fit gas multipliers'.")

    def _after_dataset_change(self) -> None:
        """Common refresh after the dataset list or the active dataset changed."""
        self._reset_analysis()
        self._emission_fit = None
        self._emission_fit_uid = None
        self._sync_dataset_list()
        self._redraw_datasets()
        self._sync_observation_mode()
        self._identify.refresh()
        has = self._obs is not None
        is_emission = bool(has and self._obs.meta.get("spectrum_kind") == "emission_flux")
        self._btn_fit.setEnabled(has and not is_emission)
        self._btn_identify.setEnabled(has and not is_emission)
        self._btn_blackbody.setEnabled(
            bool(is_emission and self._obs.unit is Unit.WAVELENGTH))
        if not has:
            self._fit_label.setText("No observation loaded.")
        elif is_emission:
            self._fit_label.setText(
                "Archive emission-flux spectrum. Transmission gas fitting/peak identification are disabled; "
                "fit a blackbody continuum as an exploratory baseline.")

    def _sync_observation_mode(self) -> None:
        """Keep the transmission model separate from archive emission-flux observations."""
        obs = self._obs
        is_emission = bool(obs is not None and obs.meta.get("spectrum_kind") == "emission_flux")
        for canvas in self._canvases.values():
            canvas.plot.setLabel(
                "left", f"Normalized {obs.meta.get('flux_density_kind', '').upper()} (relative emission)"
                if is_emission and obs is not None else "Transmittance / normalized spectrum")
            canvas.master.setVisible(self._show_model and not is_emission)
            for item in canvas.layers:
                item.setVisible(self._show_model and self._show_layers and not is_emission)
            show_continuum = bool(
                is_emission and self._emission_fit is not None
                and self._emission_fit_uid == self._active_uid)
            canvas.emission_continuum.setVisible(show_continuum)

    def _sync_dataset_list(self) -> None:
        self._syncing = True
        self._data_list.clear()
        for i, ds in enumerate(self._datasets):
            item = QListWidgetItem(f"{ds.name}   [{ds.obs.x.size} pts]")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if ds.visible else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, ds.uid)
            item.setForeground(QColor(self._DATA_COLORS[0] if ds.uid == self._active_uid
                                      else self._DATA_COLORS[1 + i % (len(self._DATA_COLORS) - 1)]))
            self._data_list.addItem(item)
            if ds.uid == self._active_uid:
                self._data_list.setCurrentItem(item)
        self._syncing = False

    def _on_dataset_selected(self, row: int) -> None:
        if self._syncing or row < 0:
            return
        uid = self._data_list.item(row).data(Qt.ItemDataRole.UserRole)
        if uid != self._active_uid:
            self._active_uid = uid
            self._after_dataset_change()
            ds = self._active_dataset()
            if ds:
                self._fit_label.setText(f"{ds.obs.x.size} points. {'; '.join(ds.obs.meta.get('import_notes', []))}")

    def _on_dataset_checked(self, item: QListWidgetItem) -> None:
        if self._syncing:
            return
        uid = item.data(Qt.ItemDataRole.UserRole)
        for ds in self._datasets:
            if ds.uid == uid:
                ds.visible = item.checkState() == Qt.CheckState.Checked
        self._redraw_datasets()

    def _on_clear_observation(self) -> None:
        """Remove the active dataset (the next one, if any, becomes active)."""
        self._datasets = [d for d in self._datasets if d.uid != self._active_uid]
        self._active_uid = self._datasets[-1].uid if self._datasets else None
        self._after_dataset_change()

    def _clear_all_datasets(self) -> None:
        self._datasets, self._active_uid = [], None
        self._after_dataset_change()

    # ------------------------------------------------------------- drawing & view
    def _redraw_datasets(self) -> None:
        """Rebuild every dataset curve on every tab (active one is gold and on top)."""
        fit = self._fit
        for unit, canvas in self._canvases.items():
            for item in canvas.data_items:
                canvas.plot.removeItem(item)
            canvas.data_items.clear()
            for i, ds in enumerate(self._datasets):
                if not ds.visible:
                    continue
                active = ds.uid == self._active_uid
                x, y = ds.obs.x_in(unit), ds.obs.y
                if active and fit is not None and fit.baseline_coeffs.size and fit.normalised_y is not None:
                    x, y = fit.used.x_in(unit), fit.normalised_y          # continuum-normalised view
                color = self._DATA_COLORS[0 if active else 1 + i % (len(self._DATA_COLORS) - 1)]
                dots = x.size <= 4000
                item = canvas.plot.plot(
                    x, y, pen=pg.mkPen(color, width=1.5 if active else 1),
                    symbol="o" if dots else None, symbolSize=3, symbolPen=None,
                    symbolBrush=pg.mkBrush(color))
                item.setZValue(6 if active else 5)
                canvas.data_items.append(item)
        self._sync_observation_mode()

    def _data_extent(self, unit: Unit):
        visible = [d.obs for d in self._datasets if d.visible]
        if not visible:
            return None
        x = np.concatenate([o.x_in(unit) for o in visible])
        y = np.concatenate([o.y for o in visible])
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())

    def _zoom_to_data(self) -> None:
        """Widen the locked view limits to include the data, then frame it."""
        for unit, canvas in self._canvases.items():
            ext = self._data_extent(unit)
            if ext is None:
                continue
            lo, hi = self._engine.grid.limits(unit)
            xmin, xmax, ymin, ymax = ext
            pad = 0.02 * max(xmax - xmin, 1e-9)
            y_lo, y_hi = min(0.0, ymin - 0.02), max(1.05, ymax * 1.05)
            vb = canvas.plot.getViewBox()
            vb.setLimits(xMin=min(lo, xmin) - pad, xMax=max(hi, xmax) + pad, yMin=y_lo, yMax=y_hi)
            vb.setRange(xRange=(xmin - pad, xmax + pad), yRange=(y_lo, y_hi), padding=0.0)

    def _reset_view(self) -> None:
        """Frame the model's own range (data stays reachable by panning)."""
        for unit, canvas in self._canvases.items():
            lo, hi = self._engine.grid.limits(unit)
            vb = canvas.plot.getViewBox()
            vb.setRange(xRange=(lo, hi), yRange=(0.0, 1.05), padding=0.0)

    def _on_show_model(self, on: bool) -> None:
        self._show_model = on
        self._sync_observation_mode()

    # -------------------------------------------------------------- tabs / residuals
    def _on_tab_changed(self, _i: int) -> None:
        self._link_residual_axis()
        self._update_residual_plot()
        self._refresh()

    def _link_residual_axis(self) -> None:
        """Keep the residual panel's x-axis locked to whichever unit tab is visible."""
        unit = self.active_unit()
        self._resid_plot.setXLink(None)
        self._resid_plot.setLabel("bottom", unit.axis_label)
        self._resid_plot.setXLink(self._canvases[unit].plot)

    def _reset_analysis(self) -> None:
        self._fit, self._anomalies, self._feature_mode = None, [], False
        self._clear_analysis_widgets()
        self._resid_plot.hide()

    # --------------------------------------------------------------------- fitting
    def _on_fit(self) -> None:
        """Bounded least-squares fit; sliders follow; residual analysis + anomaly flagging."""
        obs = self._obs
        if obs is None:
            return
        if obs.meta.get("spectrum_kind") == "emission_flux":
            self._warn("Transmission gas fitting is not valid for an emitted-flux spectrum.")
            return
        self._engine.update_conditions(self._conditions)
        window = self._engine.grid.spec.wavenumber_range_cm1 if self._chk_window.isChecked() else None
        try:
            fit = fit_multipliers(self._engine, obs, x0=self._multipliers.copy(),
                                  baseline=2 if self._chk_baseline.isChecked() else 0, window_cm1=window)
        except FitError as exc:
            self._warn(f"Fit failed: {exc}")
            return
        self._fit, self._feature_mode = fit, False
        for gas in self._engine.gases:          # moving the sliders re-renders the model curve
            self._gas_sliders[gas.gas_id].set_value(float(fit.multipliers[self._engine.gas_index[gas.gas_id]]))
        self._redraw_datasets()
        self._analyse()
        self._refresh()

    def _analyse(self) -> None:
        """Residual grid + k-sigma anomaly flagging + reverse identification (after a fit)."""
        if self._obs is None or self._fit is None:
            return
        fit = self._fit
        self._anomalies = find_anomalies(self._obs, fit, float(self._thr_spin.value()))
        sig = "estimated" if fit.sigma_estimated else "from file"
        extra = f"\n{fit.n_excluded} points outside the model range ignored." if fit.n_excluded else ""
        base = (f"\nContinuum: {fit.baseline_coeffs[0]:.3f}"
                + (f" {fit.baseline_coeffs[1]:+.3f}·u" if fit.baseline_coeffs.size > 1 else "")
                if fit.baseline_coeffs.size else "")
        self._fit_label.setText(
            f"χ² = {fit.chi2:.1f}   reduced χ² = {fit.reduced_chi2:.2f}   ({fit.n_evaluations} evals, σ {sig})"
            + base + extra + ("" if fit.success else f"\nSolver: {fit.message}"))
        self._clear_analysis_widgets()
        self._show_anomalies(self._anomalies)
        self._update_residual_plot()
        self._link_residual_axis()
        self._resid_plot.show()

    def _on_identify(self) -> None:
        """Model-free: find absorption peaks in the data and match them to known bands."""
        obs = self._obs
        if obs is None:
            return
        if obs.meta.get("spectrum_kind") == "emission_flux":
            self._warn("Transmission-spectrum peak identification is not valid for emitted flux.")
            return
        try:
            feats = find_features(obs, min_snr=float(self._snr_spin.value()))
        except FitError as exc:
            self._warn(str(exc))
            return
        self._fit, self._feature_mode = None, True
        self._resid_plot.hide()
        self._clear_analysis_widgets()
        self._anomalies = feats
        tol = max(25.0, 2.0 * self._conditions.envelope_hwhm_cm1)
        matches = (self._library.match_features([f.nu_peak_cm1 for f in feats], tol)
                   if feats and self._library is not None else [[] for _ in feats])
        self._show_anomalies(feats, matches, tol)
        self._fit_label.setText(f"Peak finding (no model): {len(feats)} features at SNR ≥ "
                                f"{self._snr_spin.value():g}; matching tolerance ±{tol:.0f} cm⁻¹.")

    # ------------------------------------------------------------ anomaly display
    def _clear_analysis_widgets(self) -> None:
        self._anom_list.clear()
        self._anom_header.setText("")
        self._cand_label.setText("")
        self._candidates = []
        for canvas in self._canvases.values():
            for region in canvas.regions:
                canvas.plot.removeItem(region)
            canvas.regions.clear()

    def _show_anomalies(self, found: Sequence[Anomaly], matches=None, tolerance: float = 25.0) -> None:
        """List features, draw bars on every tab, and rank candidate molecules."""
        if not found:
            self._anom_header.setStyleSheet("color: #81c784; font-weight: 700;")
            self._anom_header.setText("No features beyond threshold." if self._feature_mode
                                      else "No residual beyond threshold.")
            return
        feature_mode = self._feature_mode
        self._anom_header.setStyleSheet(f"color: {'#ffb74d' if feature_mode else '#ff5c5c'}; font-weight: 700;")
        self._anom_header.setText(f"{len(found)} absorption features found" if feature_mode
                                  else "Unidentified Molecular Signature Detected")
        for i, a in enumerate(found):
            kind = "" if feature_mode else ("  extra absorption" if a.extra_absorption else "  excess flux")
            depth = f"  depth {abs(a.peak_residual):.0%}" if feature_mode else ""
            ids = ""
            if matches and matches[i]:
                ids = "  → " + ", ".join(f"{m.molecule} ({m.band_cm1:.0f})" for m in matches[i])
            text = (f"{a.nu_peak_cm1:7.1f} cm⁻¹ ({1e4 / a.nu_peak_cm1:5.2f} µm)  "
                    f"{a.significance_sigma:5.1f}σ{depth}{kind}{ids}")
            entry = QListWidgetItem(text)
            entry.setToolTip(text)                         # the sidebar is narrow: show the full line on hover
            self._anom_list.addItem(entry)
            for unit, canvas in self._canvases.items():
                lo, hi = a.nu_lo_cm1, a.nu_hi_cm1
                if unit is Unit.WAVELENGTH:
                    lo, hi = 1.0e4 / hi, 1.0e4 / lo
                bar = pg.LinearRegionItem(values=(lo, hi), movable=False,
                                          brush=pg.mkBrush(255, 40, 40, 70), pen=pg.mkPen(None))
                bar.setZValue(-5)
                canvas.plot.addItem(bar)
                canvas.regions.append(bar)
        if self._library is not None:
            tol = max(tolerance, 2.0 * self._conditions.envelope_hwhm_cm1)
            exclude = () if feature_mode else tuple(self._engine.gas_index)
            ranked = self._library.rank([a.nu_peak_cm1 for a in found], tolerance_cm1=tol, exclude=exclude)
            self._candidates = list(ranked)
            if ranked:
                self._cand_label.setText("Candidates: " + ",  ".join(
                    f"{c.molecule} ({c.score:.0%})" for c in ranked))

    def _highlight(self, row: int, zoom: bool = False) -> None:
        """Emphasise one anomaly on every chart; optionally zoom the active chart onto it."""
        for canvas in self._canvases.values():
            for i, region in enumerate(canvas.regions):
                chosen = i == row
                region.setBrush(pg.mkBrush(255, 235, 59, 150) if chosen else pg.mkBrush(255, 40, 40, 70))
                region.setZValue(-4 if chosen else -5)
        if zoom and 0 <= row < len(self._anomalies):
            a = self._anomalies[row]
            unit = self.active_unit()
            lo, hi = a.nu_lo_cm1, a.nu_hi_cm1
            if unit is Unit.WAVELENGTH:
                lo, hi = 1.0e4 / hi, 1.0e4 / lo
            glo, ghi = self._engine.grid.limits(unit)
            pad = max(4.0 * (hi - lo), 0.03 * (ghi - glo))
            self._canvases[unit].plot.getViewBox().setXRange((lo + hi) / 2 - pad, (lo + hi) / 2 + pad, padding=0)

    def _on_plot_clicked(self, unit: Unit, ev) -> None:
        """Clicking near a flagged feature on a chart selects it in the list."""
        if not self._anomalies or ev.button() != Qt.MouseButton.LeftButton:
            return
        vb = self._canvases[unit].plot.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = vb.mapSceneToView(ev.scenePos()).x()
        nu = np.array([a.nu_peak_cm1 for a in self._anomalies])
        axis = nu if unit is Unit.WAVENUMBER else 1.0e4 / nu
        view = vb.viewRange()[0]
        i = int(np.argmin(np.abs(axis - x)))
        if abs(axis[i] - x) < 0.03 * (view[1] - view[0]):
            self._anom_list.setCurrentRow(i)

    def _update_residual_plot(self) -> None:
        """Draw (observed - model)/sigma on the active unit's axis (as of the last fit)."""
        if self._fit is None:
            return
        unit = self.active_unit()
        z = np.clip(self._fit.residual / self._fit.sigma, -self._RESID_CLIP, self._RESID_CLIP)
        self._resid_curve.setData(self._fit.used.x_in(unit), z)   # clipped: one huge band cannot crush the scale
        thr = float(self._thr_spin.value())
        self._thr_lines[0].setValue(thr)
        self._thr_lines[1].setValue(-thr)

    # ========================================================= Phase 5 integration
    def _on_transit_observation(self, obs: ObservedSpectrum) -> None:
        """A transmission spectrum extracted from photometry: show it on the fit workspace."""
        self.set_observation(obs)
        self._workspaces.setCurrentIndex(0)
        self.statusBar().showMessage(
            "Transmission spectrum loaded (pseudo-transmittance). Press 'Fit gas multipliers'.", 12000)

    def closeEvent(self, event) -> None:          # noqa: N802 (Qt naming)
        self._transit.shutdown()                    # releases the VTK render window cleanly
        super().closeEvent(event)

    # ================================================================ Phase 6: files
    def _build_menu(self) -> None:
        """File menu: projects, exports."""
        menu = self.menuBar().addMenu("&File")
        for text, slot, shortcut in (
            ("&Open project…", self._on_open_project, "Ctrl+O"),
            ("&Save project…", self._on_save_project, "Ctrl+S"),
            (None, None, None),
            ("Export model spectrum (CSV)…", self._on_export_model, None),
            ("Export fit residuals (CSV)…", self._on_export_fit, None),
            ("Export anomalies (CSV)…", self._on_export_anomalies, None),
            ("Export report (Markdown)…", self._on_export_report, None),
            ("Export current plot (PNG)…", self._on_export_png, None),
        ):
            if text is None:
                menu.addSeparator()
                continue
            action = menu.addAction(text)
            action.triggered.connect(slot)
            if shortcut:
                action.setShortcut(shortcut)
        helpm = self.menuBar().addMenu("&Help")
        helpm.addAction("Sources && citations…").triggered.connect(self._on_sources)

    def _ask_save(self, title: str, name: str, filt: str) -> Optional[str]:
        path, _ = QFileDialog.getSaveFileName(self, title, name, filt)
        return path or None

    def _warn(self, text: str) -> None:
        self.statusBar().showMessage(text, 10000)

    # ---- project state
    def capture_state(self) -> ProjectState:
        c = self._conditions
        lc, stellar, window, spectral_series = self._transit.capture()
        return ProjectState(
            conditions={"pressure_atm": c.pressure_atm, "temperature_k": c.temperature_k,
                        "envelope_hwhm_cm1": c.envelope_hwhm_cm1, "line_shape": c.line_shape.value},
            multipliers={g.gas_id: float(self._multipliers[self._engine.gas_index[g.gas_id]])
                         for g in self._engine.gases},
            anomaly_threshold=float(self._thr_spin.value()), stellar=stellar,
            detrend_window_days=window, observation=self._obs, lightcurve=lc,
            spectral_series=spectral_series)

    def restore_state(self, st: ProjectState) -> None:
        """Re-apply a project: sliders first, then data, then re-run fit analysis/transit."""
        c = st.conditions
        if c:
            self._s_pressure.set_value(c.get("pressure_atm", self._defaults.pressure_atm))
            self._s_temperature.set_value(c.get("temperature_k", self._defaults.temperature_k))
            self._s_envelope.set_value(c.get("envelope_hwhm_cm1", self._defaults.envelope_hwhm_cm1))
            shape = LineShape(c.get("line_shape", self._defaults.line_shape.value))
            self._shape.setCurrentIndex(list(LineShape).index(shape))
        for gas_id, value in st.multipliers.items():          # gases absent from this config are skipped
            if gas_id in self._gas_sliders:
                self._gas_sliders[gas_id].set_value(value)
        self._thr_spin.setValue(st.anomaly_threshold)
        self._clear_all_datasets()
        if st.observation is not None:
            self.set_observation(st.observation)
        self._transit.restore(st.lightcurve, st.stellar, st.detrend_window_days,
                              spectral_series=st.spectral_series)
        self._refresh()

    def _on_save_project(self) -> None:
        path = self._ask_save("Save project", self._project_path or "session.ecas", "ECAS project (*.ecas)")
        if path:
            try:
                self._project_path = str(save_project(path, self.capture_state()))
                self.statusBar().showMessage(f"Saved {self._project_path}", 8000)
            except OSError as exc:
                self._warn(f"Save failed: {exc}")

    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "ECAS project (*.ecas)")
        if not path:
            return
        try:
            self.restore_state(load_project(path))
            self._project_path = path
            self.statusBar().showMessage(f"Opened {path}", 8000)
        except ProjectError as exc:
            QMessageBox.warning(self, "Cannot open project", str(exc))

    # ---- exports
    def _on_export_model(self) -> None:
        path = self._ask_save("Export model spectrum", "model.csv", "CSV (*.csv)")
        if path:
            ecas_export.export_model_csv(path, self._engine, self.active_unit(), self._multipliers)

    def _on_export_fit(self) -> None:
        if self._obs is None or self._fit is None:
            return self._warn("Run a fit first.")
        path = self._ask_save("Export fit residuals", "fit.csv", "CSV (*.csv)")
        if path:
            ecas_export.export_fit_csv(path, self._obs, self._fit)

    def _on_export_anomalies(self) -> None:
        if not self._anomalies:
            return self._warn("No anomalies to export.")
        path = self._ask_save("Export anomalies", "anomalies.csv", "CSV (*.csv)")
        if path:
            ecas_export.export_anomalies_csv(path, self._anomalies)

    def _on_export_report(self) -> None:
        path = self._ask_save("Export report", "ecas_report.md", "Markdown (*.md)")
        if path:
            text = ecas_export.build_report_markdown(
                self._engine, self._conditions, self._multipliers, self._fit, self._anomalies,
                self._candidates, self._transit.report_lines,
                identification_lines=self._identify.result.summary_lines() if self._identify.result else (),
                citations=self._citations)
            Path(path).write_text(text, encoding="utf-8")

    def _on_export_png(self) -> None:
        """Save whichever chart is on screen (spectrum tab, or the transit light curve)."""
        from pyqtgraph.exporters import ImageExporter

        on_transit = self._workspaces.currentWidget() is self._transit
        item = (self._transit._lc_plot if on_transit else self._canvases[self.active_unit()].plot).plotItem
        path = self._ask_save("Export plot", "plot.png", "PNG (*.png)")
        if path:
            exporter = ImageExporter(item)
            exporter.parameters()["width"] = 1920
            exporter.export(path)

    def _on_sources(self) -> None:
        SourcesDialog(self._citations, self).exec()
