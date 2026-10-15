"""
ECAS "Transit & 3D" panel: photometry analysis + synchronized orbit viewers.

One clock drives everything.  The time slider / play button / speed slider call
:meth:`TransitPanel.set_time`, which moves in a single call:

* the tracker bar on the light-curve chart,
* the planet on the observer-view "sky" chart,
* the planet in the orbit viewer: PyVista 3-D when available, otherwise an equivalent
  2-D projected orbit view (so planet-size and camera controls always work).

Analysis runs on a worker thread (the UI never blocks).  All numbers come from
:mod:`app.core.transit`; this module only draws.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPen
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QScrollArea, QFrame, QFileDialog, QGraphicsEllipseItem, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QPlainTextEdit, QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..core.archives import ArchiveError, PlanetRecord, compare_with_archive, transit_params_from_archive
from ..core.atmosphere import (
    AbundanceScanResult, PetitRadtransParameters, TransmissionAtmosphereModel,
    compare_model_to_depth_spectrum, compute_petit_radtrans_transmission,
    depth_spectrum_channel_edges, scan_single_gas_abundance,
)
from ..core.engine import SpectrumEngine
from ..core.sources import CitationLog
from ..core.timeseries import SpectralTimeSeries, TimeSeriesError, load_matrix_csv, load_x1dints
from ..core.transit import (
    R_JUP_KM, R_SUN_KM, DepthSpectrum, LightCurve, StellarParams, TransitError, TransitParams, TransitResult,
    analyze_lightcurve, bin_depth_spectrum, depth_spectrum_to_observation, extract_transmission_spectrum,
    load_lightcurve_csv, orbit_positions, transit_model,
)
from .tasks import run_async
from .mappings import LogMapping
from .theme import mono_font, primary, muted

PLANET_COLOR = "#4fc3f7"


def teff_to_hex(teff_k: float) -> str:
    """Approximate blackbody colour of a star (Tanner Helland's fit), as ``#rrggbb``."""
    t = float(np.clip(teff_k, 1000.0, 40000.0)) / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
        b = 0.0 if t <= 19.0 else 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60.0) ** -0.1332047592
        g = 288.1221695283 * (t - 60.0) ** -0.0755148492
        b = 255.0
    r, g, b = (int(np.clip(v, 0, 255)) for v in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _caption(text: str) -> pg.TextItem:
    item = pg.TextItem(text, color="#9aa7b8", anchor=(0.0, 1.0))
    item.setFont(mono_font())
    return item


class SkyView(pg.PlotWidget):
    """Observer's view: star disc and planet in stellar-radius units (true relative size)."""

    HALF_WIDTH = 2.5

    def __init__(self) -> None:
        super().__init__()
        self.setBackground("#0b0e14")
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.setMouseEnabled(False, False)
        self.setRange(xRange=(-self.HALF_WIDTH, self.HALF_WIDTH),
                      yRange=(-self.HALF_WIDTH, self.HALF_WIDTH), padding=0)
        self._star = QGraphicsEllipseItem(-1.0, -1.0, 2.0, 2.0)
        self._star.setPen(QPen(Qt.PenStyle.NoPen))
        self._planet = QGraphicsEllipseItem(0.0, 0.0, 0.2, 0.2)
        self._planet.setBrush(QBrush(QColor("#10141c")))
        self._planet.setPen(QPen(QColor(PLANET_COLOR), 0))
        self._planet.setZValue(10)
        self._caption = _caption("")
        self._caption.setPos(-self.HALF_WIDTH + 0.05, -self.HALF_WIDTH + 0.05)
        self._caption.setZValue(20)
        self._k = 0.1
        for item in (self._star, self._planet, self._caption):
            self.addItem(item)
        self.set_star_color("#fff2e0")
        self._planet.setVisible(False)

    def set_star_color(self, color: str) -> None:
        self._star.setBrush(QBrush(QColor(color)))

    def set_caption(self, text: str) -> None:
        self._caption.setText(text)

    def configure(self, k: float) -> None:
        self._k = float(k)
        self._planet.setVisible(False)

    def set_position(self, x: float, y: float, z: float) -> None:
        """Planet is hidden while behind the star (z < 0 and inside the disc)."""
        behind = z < 0.0 and np.hypot(x, y) < 1.0 + self._k
        k = self._k
        self._planet.setRect(x - k, y - k, 2 * k, 2 * k)
        self._planet.setVisible(not behind and abs(x) < self.HALF_WIDTH + k)


class OrbitView2D(pg.PlotWidget):
    """
    Projected orbit view that needs nothing but pyqtgraph.  Rotates the 3-D positions into
    a camera frame (observer or oblique), draws the ring, and orders star/planet by depth.
    Same interface as the PyVista viewport, so the panel treats them interchangeably.
    """

    CAMERAS = {"observer": (0.0, 0.0, 1.0), "oblique": (1.3, 0.9, 1.1)}

    def __init__(self) -> None:
        super().__init__()
        self.setBackground("#0b0e14")
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.setMouseEnabled(False, False)
        self._ring = self.plot(pen=pg.mkPen("#8895a7", width=1))
        self._star = QGraphicsEllipseItem(-1.0, -1.0, 2.0, 2.0)
        self._star.setPen(QPen(Qt.PenStyle.NoPen))
        self._star.setZValue(0)
        self._planet = QGraphicsEllipseItem(0.0, 0.0, 0.2, 0.2)
        self._planet.setBrush(QBrush(QColor(PLANET_COLOR)))
        self._planet.setPen(QPen(Qt.PenStyle.NoPen))
        self._caption = _caption("")
        self._caption.setZValue(20)
        for item in (self._star, self._planet, self._caption):
            self.addItem(item)
        self._params: Optional[TransitParams] = None
        self._scale = 1.0
        self._camera = np.array(self.CAMERAS["oblique"])
        self._right = self._up = np.zeros(3)
        self._c = np.array([0.0, 0.0, 1.0])
        self.set_star_color("#fff2e0")
        self._update_basis()

    def set_star_color(self, color: str) -> None:
        self._star.setBrush(QBrush(QColor(color)))

    def set_caption(self, text: str) -> None:
        self._caption.setText(text)

    def _update_basis(self) -> None:
        c = self._camera / np.linalg.norm(self._camera)
        right = np.cross((0.0, 1.0, 0.0), c)
        right /= np.linalg.norm(right)
        self._c, self._right, self._up = c, right, np.cross(c, right)

    def _project(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(N,3) world points -> screen x, screen y, depth (positive = towards the viewer)."""
        return pts @ self._right, pts @ self._up, pts @ self._c

    def configure(self, params: TransitParams, planet_scale: float = 1.0, star_color: Optional[str] = None) -> None:
        self._params, self._scale = params, float(planet_scale)
        if star_color:
            self.set_star_color(star_color)
        phi = np.linspace(0.0, 2.0 * np.pi, 361)
        ring = np.column_stack(orbit_positions(params.t0 + params.period * phi / (2 * np.pi), params))
        sx, sy, _ = self._project(ring)
        self._ring.setData(sx, sy)
        half = 1.15 * max(float(np.abs(sx).max()), float(np.abs(sy).max()), 1.5)
        self.setRange(xRange=(-half, half), yRange=(-half, half), padding=0)
        self._caption.setPos(-half * 0.97, -half * 0.97)

    def set_position(self, x: float, y: float, z: float) -> None:
        if self._params is None:
            return
        sx, sy, depth = self._project(np.array([[x, y, z]]))
        r = self._params.rp_rs * self._scale
        self._planet.setRect(float(sx[0]) - r, float(sy[0]) - r, 2 * r, 2 * r)
        self._planet.setZValue(10.0 if depth[0] > 0 else -1.0)         # behind the star -> hidden by it

    def set_view(self, name: str) -> None:
        self._camera = np.array(self.CAMERAS.get(name, self.CAMERAS["oblique"]))
        self._update_basis()
        if self._params is not None:
            self.configure(self._params, self._scale)


def build_scene(plotter, params: TransitParams, planet_scale: float = 1.0, star_color: str = "#fff2e0"):
    """
    Populate any PyVista plotter with the system at true scale (units: stellar radii):
    emissive star, orbit ring, planet sphere, and a positional light at the star.
    Returns the planet actor, whose ``position`` the clock drives.
    """
    import pyvista as pv

    plotter.clear()
    plotter.add_mesh(pv.Sphere(radius=1.0, theta_resolution=48, phi_resolution=48),
                     color=star_color, lighting=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 361)
    ring = np.column_stack(orbit_positions(params.t0 + params.period * phi / (2 * np.pi), params))
    plotter.add_mesh(pv.lines_from_points(ring, close=True), color="#8895a7", line_width=1)
    planet = plotter.add_mesh(
        pv.Sphere(radius=params.rp_rs * planet_scale, theta_resolution=32, phi_resolution=32),
        color=PLANET_COLOR, smooth_shading=True, ambient=0.4)
    light = pv.Light(position=(0, 0, 0), light_type="scene light", intensity=1.0)
    light.positional = True
    light.cone_angle = 180.0               # >= 90 degrees: omnidirectional point light at the star
    plotter.add_light(light)
    return planet


def camera_for(name: str, a_rs: float):
    """Camera triple for ``"observer"`` (down the line of sight) or ``"oblique"`` views."""
    distance = 3.8 * a_rs                  # whole orbit fits PyVista's default 30-degree view
    if name == "observer":
        return [(0.0, 0.0, distance), (0, 0, 0), (0, 1, 0)]
    direction = np.array([1.3, 0.9, 1.1])
    return [tuple(direction / np.linalg.norm(direction) * distance), (0, 0, 0), (0, 1, 0)]


class Viewport3D(QWidget):
    """
    PyVista scene when available; otherwise the 2-D :class:`OrbitView2D`.  Either way the
    interface is the same.  ``available`` is True only for the real 3-D viewport and
    ``reason`` explains why not (shown to the user with the install hint).
    """

    def __init__(self, parent: Optional[QWidget] = None, enabled: bool = True):
        super().__init__(parent)
        self.available = False
        self.fallback: Optional[OrbitView2D] = None
        self.reason = ""
        self._plotter = None
        self._planet = None
        self._params: Optional[TransitParams] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        from PyQt6.QtWidgets import QApplication

        platform = QApplication.platformName() if QApplication.instance() else ""
        if enabled and platform in ("offscreen", "minimal"):
            # These Qt platforms have no OpenGL context; embedding a VTK render window there is
            # undefined behaviour (it can corrupt the process), so refuse up front.
            enabled = False
            self.reason = f"3-D viewport needs OpenGL, which the '{platform}' Qt platform lacks - showing the 2-D orbit view."
        if enabled:
            try:
                from pyvistaqt import QtInteractor

                self._plotter = QtInteractor(self)
                self._plotter.set_background("#0b0e14")
                lay.addWidget(self._plotter.interactor)
                self.available = True
            except Exception as exc:                         # missing package, no GL, ...
                missing = getattr(exc, "name", None) or type(exc).__name__
                self.reason = (f"3-D viewport unavailable ({type(exc).__name__}: {missing}). "
                               "Install with:  pip install pyvista pyvistaqt   - showing the 2-D orbit view.")
        elif not self.reason:
            self.reason = "3-D viewport disabled (--no-3d) - showing the 2-D orbit view."
        if not self.available:
            note = QLabel(self.reason)
            note.setWordWrap(True)
            note.setStyleSheet("color: #9aa7b8;")
            self.fallback = OrbitView2D()
            lay.addWidget(note)
            lay.addWidget(self.fallback, 1)

    def configure(self, params: TransitParams, planet_scale: float = 1.0, star_color: str = "#fff2e0") -> None:
        """(Re)build star, orbit ring and planet.  Star radius = 1; planet = k x scale."""
        self._params = params
        if self.fallback is not None:
            self.fallback.configure(params, planet_scale, star_color)
            return
        self._planet = build_scene(self._plotter, params, planet_scale, star_color)
        self.set_view("oblique")

    def set_caption(self, text: str) -> None:
        if self.fallback is not None:
            self.fallback.set_caption(text)

    def set_position(self, x: float, y: float, z: float) -> None:
        if self.fallback is not None:
            self.fallback.set_position(x, y, z)
        elif self._planet is not None:
            self._planet.position = (x, y, z)
            self._plotter.render()

    def set_view(self, name: str) -> None:
        """``"observer"`` looks down the line of sight; ``"oblique"`` shows the inclined orbit."""
        if self.fallback is not None:
            self.fallback.set_view(name)
        elif self._params:
            self._plotter.camera_position = camera_for(name, self._params.a_rs)
            self._plotter.render()

    def shutdown(self) -> None:
        if self._plotter is not None:
            self._plotter.close()


class _AnalysisWorker(QObject):
    """Runs :func:`analyze_lightcurve` off the UI thread."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, lc: LightCurve, stellar: StellarParams, window: Optional[float],
                 known_period: Optional[float] = None, min_snr: float = 10.0):
        super().__init__()
        self._args = (lc, stellar, window, known_period, min_snr)

    def run(self) -> None:
        lc, stellar, window, known, min_snr = self._args
        try:
            self.done.emit(analyze_lightcurve(lc, stellar, detrend_window_days=window, known_period=known,
                                              min_snr=min_snr))
        except TransitError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:                              # never let a worker die silently
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class TransitPanel(QWidget):
    """Controls + charts + viewers for transit photometry."""

    observation_ready = pyqtSignal(object)            # ObservedSpectrum -> inverse fitter
    SLIDER_STEPS = 20000
    SPEED = LogMapping(0.05, 10.0, 200)               # multiples of the base animation rate

    def __init__(self, engine: SpectrumEngine, enable_3d: bool = True, parent=None,
                 citations: Optional[CitationLog] = None, planet_fetch=None):
        super().__init__(parent)
        self._engine = engine
        self._citations = citations
        self._planet_fetch = planet_fetch
        self._planet: Optional[PlanetRecord] = None
        self._series: Optional[SpectralTimeSeries] = None
        self._transmission_spectrum: Optional[DepthSpectrum] = None
        self._lc: Optional[LightCurve] = None
        self._truth: Optional[TransitParams] = None
        self._result: Optional[TransitResult] = None
        self._params: Optional[TransitParams] = None
        self._t_min, self._t_max, self._t = 0.0, 1.0, 0.0
        self._thread: Optional[QThread] = None
        self._worker: Optional[_AnalysisWorker] = None
        self._last_tick = time.perf_counter()
        self._star_hex = teff_to_hex(5772.0)

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)
        self._play_timer.timeout.connect(self._advance)

        # ---- charts -----------------------------------------------------------------
        self._lc_plot = pg.PlotWidget()
        self._lc_plot.setLabel("bottom", "Time (days)")
        self._lc_plot.setLabel("left", "Normalised flux")
        self._lc_plot.showGrid(x=True, y=True, alpha=0.25)
        self._lc_points = self._lc_plot.plot(pen=None, symbol="o", symbolSize=2, symbolPen=None,
                                             symbolBrush=(255, 255, 255, 110))
        self._lc_model = self._lc_plot.plot(pen=pg.mkPen("#4fc3f7", width=2))
        self._tracker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ff5c5c", width=2))
        self._lc_plot.addItem(self._tracker)

        self._fold_plot = pg.PlotWidget()
        self._fold_plot.setLabel("bottom", "Hours from mid-transit")
        self._fold_plot.setLabel("left", "Flux")
        self._fold_plot.showGrid(x=True, y=True, alpha=0.25)
        self._fold_points = self._fold_plot.plot(pen=None, symbol="o", symbolSize=2, symbolPen=None,
                                                 symbolBrush=(255, 200, 0, 110))
        self._fold_model = self._fold_plot.plot(pen=pg.mkPen("#4fc3f7", width=2))
        self._fold_caption = pg.TextItem("Analyze the loaded light curve to populate the phase-folded plot.",
                                         color="#8593a6", anchor=(0, 0))
        self._fold_plot.addItem(self._fold_caption)

        self._spec_plot = pg.PlotWidget()
        self._spec_plot.setLabel("bottom", "Wavelength (µm)")
        self._spec_plot.setLabel("left", "F(t) / F_out-of-transit")
        self._spec_plot.showGrid(x=True, y=True, alpha=0.25)
        self._spec_plot.addLine(y=1.0, pen=pg.mkPen("#556070", style=Qt.PenStyle.DashLine))
        self._spec_curve = self._spec_plot.plot(pen=pg.mkPen("#ffd24d", width=1), symbol="o", symbolSize=3,
                                                symbolPen=None, symbolBrush=(255, 210, 77, 170))
        self._spec_caption = pg.TextItem("Load or mock a spectral time series to see the spectrum at the "
                                         "scrubber time.", color="#8593a6", anchor=(0, 0))
        self._spec_plot.addItem(self._spec_caption)
        self._transmission_plot = pg.PlotWidget()
        self._transmission_plot.setLabel("bottom", "Wavelength (µm)")
        self._transmission_plot.setLabel("left", "Transit depth (ppm)")
        self._transmission_plot.showGrid(x=True, y=True, alpha=0.25)
        self._transmission_points = self._transmission_plot.plot(
            pen=None, symbol="o", symbolSize=4, symbolPen=None, symbolBrush=(255, 210, 77, 190),
            name="measured depth")
        self._transmission_errors = pg.ErrorBarItem(beam=0.0, pen=pg.mkPen("#ffd24d", width=1))
        self._transmission_plot.addItem(self._transmission_errors)
        self._atmosphere_model_curve = self._transmission_plot.plot(
            pen=pg.mkPen("#4fc3f7", width=2), name="petitRADTRANS forward model")
        self._transmission_plot.addLegend(offset=(-10, 10), labelTextColor="#8593a6")
        self._transmission_caption = pg.TextItem(
            "Extract a spectrum from a wavelength-resolved transit time series.",
            color="#8593a6", anchor=(0, 0))
        self._transmission_plot.addItem(self._transmission_caption)
        self._residual_plot = pg.PlotWidget()
        self._residual_plot.setLabel("bottom", "Wavelength (µm)")
        self._residual_plot.setLabel("left", "Measured − model (ppm)")
        self._residual_plot.showGrid(x=True, y=True, alpha=0.25)
        self._residual_plot.addLine(y=0.0, pen=pg.mkPen("#8593a6", style=Qt.PenStyle.DashLine))
        self._residual_curve = self._residual_plot.plot(
            pen=None, symbol="o", symbolSize=5, symbolPen=None, symbolBrush=(255, 200, 0, 190))
        self._residual_caption = pg.TextItem(
            "Generate a forward model to compare channel-integrated residuals.",
            color="#8593a6", anchor=(0, 0))
        self._residual_plot.addItem(self._residual_caption)
        self._noise_plot = pg.PlotWidget()
        self._noise_plot.setLabel("bottom", "Consecutive integrations per bin")
        self._noise_plot.setLabel("left", "Robust β-style noise ratio")
        self._noise_plot.showGrid(x=True, y=True, alpha=0.25)
        self._noise_plot.addLine(y=1.0, pen=pg.mkPen("#8593a6", style=Qt.PenStyle.DashLine))
        self._noise_curve = self._noise_plot.plot(
            pen=pg.mkPen("#4fc3f7", width=2), symbol="o", symbolSize=5,
            symbolPen=None, symbolBrush=(79, 195, 247, 210))
        self._noise_error = pg.ErrorBarItem(beam=0.15, pen=pg.mkPen("#4fc3f7", width=1))
        self._noise_plot.addItem(self._noise_error)
        self._noise_caption = pg.TextItem(
            "Characterize the spectral residuals to view robust noise growth with bin size.",
            color="#8593a6", anchor=(0, 0))
        self._noise_plot.addItem(self._noise_caption)
        self._abundance_plot = pg.PlotWidget()
        self._abundance_plot.setLabel("bottom", "Gas volume mixing ratio")
        self._abundance_plot.setLabel("left", "Δχ² from scan minimum")
        self._abundance_plot.setLogMode(x=True, y=False)
        self._abundance_plot.showGrid(x=True, y=True, alpha=0.25)
        self._abundance_plot.addLine(y=1.0, pen=pg.mkPen("#8593a6", style=Qt.PenStyle.DashLine))
        self._abundance_curve = self._abundance_plot.plot(
            pen=pg.mkPen("#b39ddb", width=2), symbol="o", symbolSize=5,
            symbolPen=None, symbolBrush=(179, 157, 219, 210))
        self._abundance_caption = pg.TextItem(
            "Scan one configured gas to see conditional Δχ² versus volume mixing ratio.",
            color="#8593a6", anchor=(0, 0))
        self._abundance_plot.addItem(self._abundance_caption)
        self._chart_tabs = QTabWidget()
        self._chart_tabs.addTab(self._fold_plot, "Phase-folded")
        self._chart_tabs.addTab(self._spec_plot, "Spectrum at t")
        self._chart_tabs.addTab(self._transmission_plot, "Transmission depth")
        self._chart_tabs.addTab(self._residual_plot, "Model residuals")
        self._chart_tabs.addTab(self._noise_plot, "Noise diagnostics")
        self._chart_tabs.addTab(self._abundance_plot, "Abundance profile")

        self._sky = SkyView()
        self._view3d = Viewport3D(enabled=enable_3d)

        # ---- scrubber row -----------------------------------------------------------
        self._play = QPushButton("▶ Play")
        self._play.setCheckable(True)
        self._play.toggled.connect(self._on_play)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self.SLIDER_STEPS)
        self._slider.valueChanged.connect(self._on_scrub)
        self._speed = QSlider(Qt.Orientation.Horizontal)
        self._speed.setRange(0, self.SPEED.steps)
        self._speed.setValue(self.SPEED.to_pos(1.0))
        self._speed.setMaximumWidth(150)
        self._speed.valueChanged.connect(self._on_speed)
        self._speed_label = QLabel()
        self._speed_label.setMinimumWidth(115)
        self._clock = QLabel("t = –")
        self._clock.setMinimumWidth(330)
        self._clock.setFont(mono_font())
        self._scale = QDoubleSpinBox()
        self._scale.setRange(1.0, 40.0)
        self._scale.setValue(1.0)
        self._scale.setPrefix("planet ×")
        self._scale.setToolTip("Exaggerate the planet's radius in the orbit view (1 = true scale)")
        self._scale.valueChanged.connect(self._rebuild_3d)
        b_obs, b_obl = QPushButton("Observer view"), QPushButton("Oblique view")
        b_obs.clicked.connect(lambda: self._view3d.set_view("observer"))
        b_obl.clicked.connect(lambda: self._view3d.set_view("oblique"))

        row = QHBoxLayout()
        row.addWidget(self._play)
        row.addWidget(self._slider, 1)
        row.addWidget(self._clock)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Speed"))
        row2.addWidget(self._speed)
        row2.addWidget(self._speed_label)
        row2.addStretch(1)
        for w in (self._scale, b_obs, b_obl):
            row2.addWidget(w)

        views = QSplitter(Qt.Orientation.Horizontal)
        views.addWidget(self._sky)
        views.addWidget(self._view3d)
        charts = QSplitter(Qt.Orientation.Vertical)
        charts.addWidget(self._lc_plot)
        charts.addWidget(self._chart_tabs)
        charts.addWidget(views)
        charts.setSizes([260, 170, 330])
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.addWidget(charts, 1)
        r_lay.addLayout(row)
        r_lay.addLayout(row2)

        root = QHBoxLayout(self)
        root.addWidget(self._build_controls())
        root.addWidget(right, 1)
        self._on_speed(self._speed.value())
        self._update_captions()

    # ------------------------------------------------------------------- controls
    def _build_controls(self) -> QWidget:
        box = QWidget()
        box.setMinimumWidth(380)
        lay = QVBoxLayout(box)

        phot = QGroupBox("Photometry")
        p_lay = QVBoxLayout(phot)
        row = QHBoxLayout()
        b_load, b_mock = QPushButton("Load CSV/FITS…"), QPushButton("Mock light curve")
        b_load.clicked.connect(self._on_load)
        b_mock.clicked.connect(self._on_mock)
        row.addWidget(b_load)
        row.addWidget(b_mock)
        p_lay.addLayout(row)
        b_planet = QPushButton("Look up planet…")
        b_planet.setToolTip("NASA Exoplanet Archive: fill in R★, M★ and Teff and compare your fit with published values.")
        b_planet.clicked.connect(self._on_lookup)
        p_lay.addWidget(b_planet)
        self._btn_fetch_mast = QPushButton("Fetch TESS light curve…")
        self._btn_fetch_mast.setToolTip("Find and load public TESS photometry for the selected host star via MAST.")
        self._btn_fetch_mast.setEnabled(False)
        self._btn_fetch_mast.clicked.connect(self._on_fetch_photometry)
        p_lay.addWidget(self._btn_fetch_mast)

        self._spin_r = self._spin("R★", 1.0, 0.05, 50.0, " R☉", 3)
        self._spin_m = self._spin("M★", 1.0, 0.05, 50.0, " M☉", 3)
        self._spin_t = self._spin("Teff", 5772.0, 2000.0, 50000.0, " K", 0)
        self._spin_w = self._spin("Detrend window", 1.0, 0.0, 20.0, " d (0 = off)", 2)
        for lbl, spin in (self._spin_r, self._spin_m, self._spin_t):
            spin.valueChanged.connect(self._on_stellar_changed)
        self._spin_snr = self._spin("Detection SNR", 10.0, 3.0, 50.0, "", 1)
        self._spin_snr[1].setToolTip("Minimum signal-to-noise for a transit to be accepted. Lower it only if "
                                     "the light curve plainly shows a dip; the default protects against noise.")
        for lbl, spin in (self._spin_r, self._spin_m, self._spin_t, self._spin_w, self._spin_snr):
            r = QHBoxLayout()
            r.addWidget(lbl)
            r.addWidget(spin)
            p_lay.addLayout(r)
        self._btn_analyze = QPushButton("Analyze transit")
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(lambda: self.analyze(blocking=False))
        p_lay.addWidget(self._btn_analyze)
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._report.setFont(mono_font())
        self._report.setMinimumHeight(250)
        p_lay.addWidget(self._report)

        spec = QGroupBox("Transmission spectroscopy")
        s_lay = QVBoxLayout(spec)
        note = QLabel("A wavelength-resolved transit time series (e.g. JWST x1dints) is needed to derive "
                      "a spectrum; broadband TESS photometry cannot provide one.")
        note.setWordWrap(True)
        muted(note)
        s_lay.addWidget(note)
        b_spec = QPushButton("Mock spectral curves → inverse fit")
        b_spec.setToolTip("Simulate per-wavelength light curves from the current gas mix, "
                          "extract transit depths and send them to the inverse fitter.")
        b_spec.clicked.connect(self._on_spectral)
        note = QLabel("Extracted depths are converted to a pseudo-transmittance (heuristic), "
                      "so fitted multipliers are relative.")
        note.setWordWrap(True)
        s_lay.addWidget(b_spec)
        s_lay.addWidget(note)

        ser = QGroupBox("Spectral time series")
        t_lay = QVBoxLayout(ser)
        row_s = QHBoxLayout()
        b_sload, b_sfetch, b_smock = (
            QPushButton("Load x1dints / matrix…"), QPushButton("Fetch JWST spectra…"), QPushButton("Mock"))
        b_sload.setToolTip("JWST *_x1dints.fits (one spectrum per integration) or a CSV matrix: first row "
                           "wavelengths, first column times.  Get x1dints from MAST.")
        b_sload.clicked.connect(self._on_load_series)
        b_sfetch.setEnabled(False)
        b_sfetch.setToolTip("Look up a planet first, then fetch a JWST wavelength-resolved transit time series from MAST.")
        b_sfetch.clicked.connect(self._on_fetch_spectral_series)
        b_smock.clicked.connect(self._on_mock_series)
        row_s.addWidget(b_sload, 1)
        row_s2 = QHBoxLayout()
        row_s2.addWidget(b_sfetch, 1)
        row_s2.addWidget(b_smock)
        self._btn_fetch_spectral = b_sfetch
        self._series_label = QLabel("No spectral time series loaded.")
        self._series_label.setWordWrap(True)
        muted(self._series_label)
        bin_row = QHBoxLayout()
        bin_row.addWidget(QLabel("Wavelength bin size"))
        self._bin_size = QSpinBox()
        self._bin_size.setRange(1, 1)
        self._bin_size.setToolTip("Combine adjacent wavelength channels using inverse-variance weighting.")
        self._bin_size.valueChanged.connect(self._on_bin_size_changed)
        bin_row.addWidget(self._bin_size)
        self._btn_extract = QPushButton("Extract measured spectrum")
        primary(self._btn_extract)
        self._btn_extract.setEnabled(False)
        self._btn_extract.clicked.connect(self._on_extract)
        self._btn_noise = QPushButton("Characterize time-correlated noise")
        self._btn_noise.setEnabled(False)
        self._btn_noise.setToolTip(
            "Compare robust spectral residual scatter across integration bin sizes; uncertainties are not changed.")
        self._btn_noise.clicked.connect(self._on_correlated_noise)
        self._btn_identify = QPushButton("Send spectrum to Identify →")
        self._btn_identify.setEnabled(False)
        self._btn_identify.clicked.connect(self._on_send_to_identify)
        self._btn_export_depth = QPushButton("Export measured spectrum CSV…")
        self._btn_export_depth.setEnabled(False)
        self._btn_export_depth.clicked.connect(self._on_export_depth_spectrum)
        self._btn_atmosphere_model = QPushButton("Generate petitRADTRANS model…")
        self._btn_atmosphere_model.setEnabled(False)
        self._btn_atmosphere_model.setToolTip(
            "Overlay a clear, isothermal forward model using petitRADTRANS; no abundance fit is performed.")
        self._btn_atmosphere_model.clicked.connect(self._on_generate_atmosphere_model)
        self._btn_abundance_scan = QPushButton("Scan one gas abundance…")
        self._btn_abundance_scan.setEnabled(False)
        self._btn_abundance_scan.setToolTip(
            "Conditional 1D pRT profile scan; other settings and gases are fixed, with a fitted depth offset.")
        self._btn_abundance_scan.clicked.connect(self._on_scan_abundance)
        t_lay.addLayout(row_s)
        t_lay.addLayout(row_s2)
        t_lay.addWidget(self._series_label)
        t_lay.addLayout(bin_row)
        t_lay.addWidget(self._btn_extract)
        t_lay.addWidget(self._btn_noise)
        t_lay.addWidget(self._btn_export_depth)
        t_lay.addWidget(self._btn_atmosphere_model)
        t_lay.addWidget(self._btn_abundance_scan)
        t_lay.addWidget(self._btn_identify)

        lay.addWidget(phot)
        lay.addWidget(ser)
        lay.addWidget(spec)
        lay.addStretch(1)
        scroll = QScrollArea()                      # the sidebar scrolls rather than cutting text off
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(box)
        scroll.setMinimumWidth(400)
        return scroll

    @staticmethod
    def _spin(label: str, value: float, lo: float, hi: float, suffix: str, decimals: int):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setValue(value)
        s.setSuffix(suffix)
        return QLabel(label), s

    def _stellar(self) -> StellarParams:
        return StellarParams(self._spin_r[1].value(), self._spin_m[1].value(), self._spin_t[1].value())

    # ------------------------------------------------------------- live stellar update
    def _on_stellar_changed(self, _value: float = 0.0) -> None:
        """R★, M★ and Teff act immediately: physical numbers, captions, star colour."""
        st = self._stellar()
        hex_color = teff_to_hex(st.teff_k)
        recolor = hex_color != self._star_hex
        self._star_hex = hex_color
        if self._result is not None:
            self._result = self._result.with_stellar(st)
            self._show_report()
        self._update_captions()
        if recolor:
            self._sky.set_star_color(hex_color)
            if self._params is not None:
                self._rebuild_3d()

    def _update_captions(self) -> None:
        st = self._stellar()
        text = f"R★ = {st.radius_rsun:.2f} R☉ ({st.radius_rsun * R_SUN_KM:,.0f} km)   Teff {st.teff_k:.0f} K"
        if self._result is not None:
            r = self._result
            text += f"\nRp = {r.rp_rjup:.2f} R_J   a = {r.a_au:.4f} AU   v = {r.orbital_speed_kms:.0f} km/s"
        self._sky.set_caption(text)
        self._view3d.set_caption(text)
        self._sky.set_star_color(self._star_hex)

    def _show_report(self) -> None:
        if self._result is None:
            return
        st = self._stellar()
        head = f"Host star: R★ {st.radius_rsun:.3g} R☉, M★ {st.mass_msun:.3g} M☉, Teff {st.teff_k:.0f} K\n"
        self._report.setPlainText(head + "\n".join(self._report_text()))

    def _report_text(self) -> list:
        lines = self._result.summary_lines() if self._result else []
        if self._result is not None and self._planet is not None:
            lines += ["", f"Comparison with the NASA Exoplanet Archive ({self._planet.name}):"]
            lines += compare_with_archive(self._result, self._planet)
        return lines

    @property
    def report_lines(self):
        lines = self._report_text()
        if self._transmission_spectrum is not None:
            retrieval = self._report.toPlainText().splitlines()
            if retrieval:
                lines += ["", "Transmission retrieval state:", *retrieval]
        return lines

    # ---------------------------------------------------------------- planet lookup
    def _on_lookup(self) -> None:
        from .archive_dialog import PlanetLookupDialog

        dlg = PlanetLookupDialog(self, fetch=self._planet_fetch)
        if dlg.exec() == dlg.DialogCode.Accepted and dlg.record is not None:
            self.use_planet(dlg.record)

    def use_planet(self, record: PlanetRecord) -> None:
        """Adopt an archive record: host-star inputs, citation, comparison with the fit."""
        self._planet = record
        self._btn_fetch_mast.setEnabled(bool(record.values.get("hostname") or record.name))
        self._btn_fetch_spectral.setEnabled(bool(record.values.get("hostname") or record.name))
        sp, missing = record.to_stellar_params()
        self._spin_r[1].setValue(sp.radius_rsun)
        self._spin_m[1].setValue(sp.mass_msun)
        self._spin_t[1].setValue(sp.teff_k)
        if self._citations is not None:
            record.cite(self._citations)
        if self._result is not None:
            self._show_report()
        else:
            self._report.setPlainText("\n".join(record.summary_lines()) + (
                f"\n\nThe archive has no value for: {', '.join(missing)} (defaults kept)." if missing else "")
                + "\n\nThese archive values are not photometry. Fetch a TESS light curve or load one, then press "
                  "'Analyze transit'.")

    def _on_fetch_photometry(self) -> None:
        if self._planet is None:
            return
        from .archive_dialog import TessLightCurveDialog

        target = str(self._planet.values.get("hostname") or self._planet.name)
        dlg = TessLightCurveDialog(target, self)
        if dlg.exec() != dlg.DialogCode.Accepted or dlg.path is None:
            return
        try:
            lc = load_lightcurve_csv(dlg.path)
        except (TransitError, OSError) as exc:
            self._report.setPlainText(f"Could not load the downloaded TESS light curve {dlg.path}:\n{exc}")
            return
        observation = dlg.observation or {}
        lc.meta.update({
            "mast_target": target,
            "mast_observation": observation.get("obs_id", ""),
            "mast_obsid": observation.get("obsid", ""),
            "mast_instrument": observation.get("instrument_name", "TESS"),
            "planet_record": {
                "values": dict(self._planet.values),
                "query": self._planet.query,
                "url": self._planet.url,
                "retrieved": self._planet.retrieved,
                "refs": dict(self._planet.refs),
            },
        })
        self.set_lightcurve(lc)
        obs_id = observation.get("obs_id", "unknown observation")
        if self._citations is not None:
            self._citations.record("mast", f"{target} TESS light curve (MAST observation {obs_id})",
                                   query=f"https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={target}")

    def _on_fetch_spectral_series(self) -> None:
        if self._planet is None:
            return
        from .archive_dialog import JwstSpectralTimeSeriesDialog

        target = str(self._planet.values.get("hostname") or self._planet.name)
        dlg = JwstSpectralTimeSeriesDialog(target, self)
        if dlg.exec() != dlg.DialogCode.Accepted or dlg.series is None:
            return
        product = dlg.product or {}
        series = dlg.series
        series.meta.update({
            "source": str(dlg.path),
            "mast_target": target,
            "mast_observation": product.get("obs_id", ""),
            "mast_product": product.get("filename", ""),
            "proposal_id": product.get("proposal_id", ""),
            "instrument": product.get("instrument_name", ""),
            "planet_record": {
                "values": dict(self._planet.values),
                "query": self._planet.query,
                "url": self._planet.url,
                "retrieved": self._planet.retrieved,
                "refs": dict(self._planet.refs),
            },
        })
        self.set_series(series)
        obs_id = product.get("obs_id", "unknown observation")
        filename = product.get("filename", "JWST x1dints")
        if self._citations is not None:
            self._citations.record("mast", f"{target} JWST spectral time series (MAST {obs_id}; {filename})",
                                   query=str(product.get("data_uri", "")))

    # ------------------------------------------------------------------- data flow
    def set_lightcurve(self, lc: LightCurve, truth: Optional[TransitParams] = None,
                       keep_series: bool = False) -> None:
        """Install a light curve; resets results and the clock to the data span."""
        if not keep_series:
            self._series = None
            self._btn_extract.setEnabled(False)
            self._btn_noise.setEnabled(False)
            self._btn_abundance_scan.setEnabled(False)
            self._btn_identify.setEnabled(False)
            self._btn_export_depth.setEnabled(False)
            self._btn_atmosphere_model.setEnabled(False)
            self._series_label.setText(
                "Photometry only: Spectrum at t and Transmission depth need a wavelength-resolved time series.")
            self._spec_curve.setData([], [])
            self._transmission_spectrum = None
            self._transmission_points.setData([], [])
            self._transmission_errors.setData(x=[], y=[], top=[], bottom=[])
            self._atmosphere_model_curve.setData([], [])
            self._noise_curve.setData([], [])
            self._noise_error.setData(x=[], y=[], top=[], bottom=[])
            self._abundance_curve.setData([], [])
            self._abundance_caption.setText(
                "Scan one configured gas to see conditional Δχ² versus volume mixing ratio.")
            self._noise_caption.setText(
                "Characterize the spectral residuals to view robust noise growth with bin size.")
            self._spec_caption.setText("No wavelength-resolved series loaded; broadband photometry has no spectrum.")
            self._transmission_caption.setText(
                "Load spectral time-series data and extract a spectrum to populate this plot.")
        self._lc, self._truth, self._result, self._params = lc, truth, None, None
        self._t_min, self._t_max = float(lc.time[0]), float(lc.time[-1])
        self._lc_points.setData(lc.time, lc.flux / np.median(lc.flux))
        self._lc_model.setData([], [])
        self._fold_points.setData([], [])
        self._fold_model.setData([], [])
        self._fold_caption.setText("Analyze the loaded light curve to populate the phase-folded plot.")
        self._fold_caption.setPos(self._t_min, 1.0)
        self._fold_caption.setVisible(True)
        self._report.setPlainText(f"{lc.time.size} points, {self._t_max - self._t_min:.2f} d span.\n"
                                  "Press 'Analyze transit'.")
        self._btn_analyze.setEnabled(True)
        self._set_clock(self._t_min)
        self._update_captions()

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load light curve", "", "Light curves (*.csv *.dat *.txt *.tsv *.fits *.fit);;All files (*)")
        if path:
            try:
                self.set_lightcurve(load_lightcurve_csv(path))
            except (TransitError, OSError) as exc:
                self._report.setPlainText(f"Could not load {path}:\n{exc}")

    def _on_mock(self) -> None:
        from ..core.mock_data import make_mock_lightcurve

        lc, truth = make_mock_lightcurve()
        self.set_lightcurve(lc, truth)

    def analyze(self, blocking: bool = False) -> None:
        """
        Run the transit analysis.  ``blocking=False`` (the button) uses a worker thread and
        keeps the window responsive; ``blocking=True`` runs inline (project restore, tests).
        """
        if self._lc is None or self._thread is not None:
            return
        window = self._spin_w[1].value() or None
        known = self._known_period()
        if blocking:
            try:
                self._apply_result(analyze_lightcurve(self._lc, self._stellar(), detrend_window_days=window,
                                                      known_period=known, min_snr=self._spin_snr[1].value()))
            except TransitError as exc:
                self._on_analysis_failed(str(exc))
            return
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.setText("Analyzing…")
        self._report.setPlainText("Analyzing… (period search runs in the background)")
        self._thread = QThread(self)
        self._worker = _AnalysisWorker(self._lc, self._stellar(), window, known, self._spin_snr[1].value())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_analysis_failed)
        for sig in (self._worker.done, self._worker.failed):
            sig.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _known_period(self) -> Optional[float]:
        """Archive period, used only when the data cover fewer than two orbits (single transit)."""
        if self._planet is None or self._lc is None:
            return None
        period = self._planet.get("pl_orbper")
        span = float(self._lc.time[-1] - self._lc.time[0])
        return float(period) if period and span < 2.0 * float(period) else None

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = self._worker = None
        self._btn_analyze.setText("Analyze transit")
        self._btn_analyze.setEnabled(self._lc is not None)

    def _on_worker_done(self, res: TransitResult) -> None:
        self._apply_result(res)

    def _on_analysis_failed(self, message: str) -> None:
        self._report.setPlainText(f"Analysis failed:\n{message}")

    def _apply_result(self, res: TransitResult) -> None:
        self._result, self._params = res.with_stellar(self._stellar()), res.params
        self._lc_points.setData(res.time, res.flux)
        self._lc_model.setData(res.time, res.model)
        p = res.params
        half = p.period / 2.0
        phase_h = ((res.time - p.t0 + half) % p.period - half) * 24.0
        order = np.argsort(phase_h)
        self._fold_points.setData(phase_h, res.flux)
        self._fold_model.setData(phase_h[order], res.model[order])
        self._fold_caption.setVisible(False)
        self._fold_plot.setXRange(-3.0 * res.duration_hours, 3.0 * res.duration_hours, padding=0)
        self._show_report()
        self._sky.configure(p.rp_rs)
        self._rebuild_3d()
        self._update_captions()
        self._set_clock(self._t)

    def _rebuild_3d(self, _value: float = 0.0) -> None:
        if self._params is not None:
            self._view3d.configure(self._params, self._scale.value(), self._star_hex)
            self._update_captions()
            self.set_time(self._t)

    # ------------------------------------------------------------- spectral time series
    def _on_load_series(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load spectral time series", "",
            "Spectral time series (*x1dints.fits *.fits *.csv *.txt *.tsv);;All files (*)")
        if path:
            self.load_series_file(path)

    def load_series_file(self, path: str, wavelength_unit: str = "auto") -> bool:
        try:
            series = (load_x1dints(path) if path.lower().endswith((".fits", ".fit"))
                      else load_matrix_csv(path, wavelength_unit))
        except (TimeSeriesError, OSError) as exc:
            self._report.setPlainText(f"Could not load {path}:\n{exc}")
            return False
        self.set_series(series)
        return True

    def _on_mock_series(self) -> None:
        from ..core.mock_data import make_mock_lightcurve, make_mock_spectral_timeseries

        truth = make_mock_lightcurve()[1]
        if self._citations is not None:
            self._citations.record("mock", "mock spectral time series")
        self.set_series(make_mock_spectral_timeseries(self._engine, truth, multipliers=self._engine.default_multipliers()),
                        truth)

    def set_series(self, series: SpectralTimeSeries, truth: Optional[TransitParams] = None) -> None:
        """Install a spectral time series: its white-light curve drives the usual transit analysis."""
        self._series = series
        self._transmission_spectrum = None
        self._btn_abundance_scan.setEnabled(False)
        self._abundance_curve.setData([], [])
        self._abundance_caption.setText(
            "Scan one configured gas to see conditional Δχ² versus volume mixing ratio.")
        self._noise_curve.setData([], [])
        self._noise_error.setData(x=[], y=[], top=[], bottom=[])
        self._noise_caption.setText(
            "Characterize the spectral residuals to view robust noise growth with bin size.")
        self._btn_identify.setEnabled(False)
        self._btn_export_depth.setEnabled(False)
        self._btn_atmosphere_model.setEnabled(False)
        self._atmosphere_model_curve.setData([], [])
        self._transmission_points.setData([], [])
        self._transmission_errors.setData(x=[], y=[], top=[], bottom=[])
        self._transmission_caption.setText(
            "Analyze the transit, then extract the wavelength-resolved transit depth.")
        self.set_lightcurve(series.white_light(), truth, keep_series=True)
        self._bin_size.setMaximum(max(1, series.n_wavelengths // 8))
        self._bin_size.setValue(1)
        wl = series.wavelength_um
        instrument = str(series.meta.get("instrument", ""))
        product = str(series.meta.get("mast_product", ""))
        origin = f"{instrument} · {product}.  " if product else ""
        self._series_label.setText(
            f"{origin}{series.n_times} integrations × {series.n_wavelengths} channels, "
            f"{wl.min():.2f}–{wl.max():.2f} µm.  "
            f"Time stamps: {series.meta.get('time_kind', 'unknown')}.")
        self._btn_extract.setEnabled(True)
        self._btn_noise.setEnabled(True)
        self._chart_tabs.setCurrentWidget(self._spec_plot)
        self._report.setPlainText(self._report.toPlainText() + "\n\nSpectral time series loaded: scrub the time "
                                  "slider to see the spectrum at that moment ('Spectrum at t' tab).")
        self._update_snapshot(self._t)

    def _update_snapshot(self, t: float) -> None:
        if self._series is None:
            return
        cadence = float(np.median(np.diff(self._series.time)))
        wl, ratio, n = self._series.relative_spectrum_at(t, self._params, 3.0 * cadence)
        self._spec_curve.setData(wl, ratio)
        phase = ""
        if self._params is not None:
            phase = f"   phase {((t - self._params.t0) / self._params.period + 0.5) % 1.0 - 0.5:+.3f}"
        self._spec_caption.setText(f"t = {t:.4f} d{phase}   ({n} integration{'s' if n != 1 else ''} averaged)")
        vb = self._spec_plot.getViewBox()
        lo, hi = float(np.nanmin(ratio)), float(np.nanmax(ratio))
        pad = max(0.15 * (max(hi, 1.0) - min(lo, 1.0)), 2e-4)               # always keep the 1.0 reference in view
        vb.setRange(xRange=(float(wl.min()), float(wl.max())),
                    yRange=(min(lo, 1.0) - pad, max(hi, 1.0) + pad * 2.5), padding=0.0)
        self._spec_caption.setPos(float(wl.min()), vb.viewRange()[1][1])

    def _extraction_params(self) -> TransitParams:
        """Best available transit geometry: own analysis > archive ephemeris > the mock's truth.

        If no analysis has been run yet, run it now, so the user is not asked to press a second button.
        """
        if self._params is None and self._lc is not None:
            self.analyze(blocking=True)                 # reports its own failure reason if it fails
        if self._params is not None:
            return self._params
        series = self._series
        archive_problem = ""
        if self._planet is not None and series is not None:
            kind, off = str(series.meta.get("time_kind", "")), float(series.meta.get("time_offset", 0.0))
            if kind in ("BJD_TDB", "MJD_UTC"):
                off_bjd = off if kind == "BJD_TDB" else off + 2400000.5
                mid = off_bjd + 0.5 * float(series.time[0] + series.time[-1])
                try:
                    return transit_params_from_archive(self._planet, mid, off_bjd)
                except ArchiveError as exc:
                    archive_problem = str(exc)
        if self._truth is not None:
            return self._truth
        detail = f" The archive also lacks what it needs: {archive_problem}" if archive_problem else ""
        raise TransitError("The transit geometry could not be determined from your data (see the report above "
                           "for why the analysis failed)." + detail)

    def _on_extract(self) -> None:
        if self._series is None:
            return
        try:
            params = self._extraction_params()
            spec = self._series.transmission_spectrum(params)
        except (TransitError, ArchiveError) as exc:
            self._report.setPlainText(f"Cannot extract the transmission spectrum:\n{exc}")
            return
        self._transmission_spectrum = spec
        self._btn_identify.setEnabled(True)
        self._btn_export_depth.setEnabled(True)
        self._btn_atmosphere_model.setEnabled(True)
        self._btn_abundance_scan.setEnabled(True)
        self._render_transmission_spectrum()

    def _on_correlated_noise(self) -> None:
        if self._series is None:
            return
        try:
            diagnostics = self._series.characterize_correlated_noise(self._extraction_params())
        except (TimeSeriesError, TransitError, ArchiveError) as exc:
            self._report.setPlainText(f"Could not characterize time-correlated noise:\n{exc}")
            return
        self._noise_curve.setData(diagnostics.bin_sizes, diagnostics.beta_median)
        self._noise_error.setData(
            x=diagnostics.bin_sizes,
            y=diagnostics.beta_median,
            top=diagnostics.beta_high - diagnostics.beta_median,
            bottom=diagnostics.beta_median - diagnostics.beta_low,
        )
        self._noise_plot.getViewBox().setRange(
            xRange=(1.0, float(diagnostics.bin_sizes.max())), padding=0.05)
        self._noise_caption.setText(
            "Residuals after fixed-geometry transit template; channel spread is 16th–84th percentile. "
            "Diagnostic only, not an uncertainty correction.")
        self._chart_tabs.setCurrentWidget(self._noise_plot)
        self._report.setPlainText(
            "Time-correlated-noise diagnostic (robust MAD residual scatter; beta-style ratio).\n"
            f"White-noise residual scatter: {diagnostics.white_noise_ppm_median:.2f} ppm "
            f"(median across {diagnostics.n_channels} usable channels).\n"
            f"At {int(diagnostics.bin_sizes[-1])} integrations per bin: "
            f"β = {diagnostics.beta_median[-1]:.2f} "
            f"(channel 16th–84th percentile {diagnostics.beta_low[-1]:.2f}–"
            f"{diagnostics.beta_high[-1]:.2f}).\n"
            "Ratios compare robust residual scatter with the white-noise 1/√N expectation. "
            "This is an empirical diagnostic after the fixed-geometry transit template; it does not "
            "change extracted uncertainties or correct systematics.")

    def _on_bin_size_changed(self, _value: int) -> None:
        if self._transmission_spectrum is not None:
            self._render_transmission_spectrum()

    def _render_transmission_spectrum(self) -> None:
        if self._transmission_spectrum is None:
            return
        try:
            spec = bin_depth_spectrum(self._transmission_spectrum, self._bin_size.value())
        except TransitError as exc:
            self._report.setPlainText(f"Could not bin the measured spectrum:\n{exc}")
            return
        self._show_transmission_spectrum(spec)

    def _on_send_to_identify(self) -> None:
        if self._transmission_spectrum is None:
            return
        try:
            spec = bin_depth_spectrum(self._transmission_spectrum, self._bin_size.value())
            obs = depth_spectrum_to_observation(spec)
        except (TransitError, ValueError) as exc:
            self._report.setPlainText(f"Cannot prepare the spectrum for Identify:\n{exc}")
            return
        obs.meta["name"] = "transmission spectrum (from time series)"
        origin = ("your transit analysis" if self._params is not None else
                  "the NASA Exoplanet Archive ephemeris" if self._planet is not None else "the mock's true geometry")
        obs.meta["import_notes"] = [f"depths extracted with the geometry from {origin}",
                                    "measured values are wavelength-dependent transit depths",
                                    "atmospheric fitter input is a continuum-relative pseudo-transmittance; "
                                    "fitted abundances are relative, not absolute"]
        self.observation_ready.emit(obs)

    def _on_export_depth_spectrum(self) -> None:
        if self._transmission_spectrum is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export measured transit-depth spectrum", "",
                                              "CSV (*.csv)")
        if not path:
            return
        try:
            from ..core.export import export_depth_spectrum_csv

            export_depth_spectrum_csv(
                path, bin_depth_spectrum(self._transmission_spectrum, self._bin_size.value()))
        except (OSError, TransitError) as exc:
            QMessageBox.warning(self, "Spectrum export failed", str(exc))
            return
        self._report.setPlainText(self._report.toPlainText() + f"\nMeasured spectrum exported to {path}.")

    def _on_generate_atmosphere_model(self) -> None:
        if self._transmission_spectrum is None or self._series is None:
            return
        from .atmosphere_dialog import PetitRadtransSettingsDialog

        planet_radius_rjup = 1.0
        if self._params is not None:
            planet_radius_rjup = (self._params.rp_rs * self._stellar().radius_rsun
                                  * R_SUN_KM / R_JUP_KM)
        elif self._planet is not None:
            archive_radius = self._planet.get("pl_radj")
            if archive_radius is not None:
                planet_radius_rjup = float(archive_radius)
        dialog = PetitRadtransSettingsDialog(self._stellar().radius_rsun, planet_radius_rjup, self)
        if dialog.exec() != dialog.DialogCode.Accepted or dialog.parameters is None:
            return
        parameters = dialog.parameters
        spectrum = bin_depth_spectrum(self._transmission_spectrum, self._bin_size.value())
        try:
            edges = depth_spectrum_channel_edges(spectrum.wavelength_um)
        except ValueError as exc:
            self._report.setPlainText(f"Cannot compare a model to this spectrum:\n{exc}")
            return
        self._abundance_curve.setData([], [])
        self._abundance_caption.setText(
            "Scan one configured gas to see conditional Δχ² versus volume mixing ratio.")
        self._model_run_source_spectrum = self._transmission_spectrum
        self._model_run_series = self._series
        self._model_run_bin_size = self._bin_size.value()
        self._btn_atmosphere_model.setEnabled(False)
        self._btn_abundance_scan.setEnabled(False)
        self._report.setPlainText(
            "Generating petitRADTRANS forward model… This may take time if opacity files need loading.")
        run_async(
            self,
            lambda: compute_petit_radtrans_transmission(
                parameters, (float(edges[0]), float(edges[-1]))),
            self._on_atmosphere_model_ready,
            self._on_atmosphere_model_failed,
        )

    def _on_atmosphere_model_ready(self, model: TransmissionAtmosphereModel) -> None:
        if (self._series is not getattr(self, "_model_run_series", None)
                or self._transmission_spectrum is not getattr(self, "_model_run_source_spectrum", None)
                or self._bin_size.value() != getattr(self, "_model_run_bin_size", None)):
            self._report.setPlainText("Discarded model output because the source spectrum changed while it ran.")
            self._btn_atmosphere_model.setEnabled(self._transmission_spectrum is not None)
            self._btn_abundance_scan.setEnabled(self._transmission_spectrum is not None)
            return
        try:
            spectrum = bin_depth_spectrum(self._transmission_spectrum, self._bin_size.value())
            comparison = compare_model_to_depth_spectrum(model, spectrum)
        except ValueError as exc:
            self._on_atmosphere_model_failed(str(exc))
            return
        self._atmosphere_model_curve.setData(
            comparison.wavelength_um, comparison.model_depth_ppm,
            pen=None, symbol="o", symbolSize=4, symbolBrush=(79, 195, 247, 210))
        valid = np.isfinite(comparison.residual_sigma)
        self._residual_curve.setData(
            comparison.wavelength_um[valid], comparison.residual_ppm[valid])
        self._residual_caption.setText(
            "Measured − channel-integrated model; points with invalid uncertainties are omitted.")
        if valid.any():
            self._residual_plot.getViewBox().setRange(
                xRange=(float(comparison.wavelength_um[valid].min()),
                        float(comparison.wavelength_um[valid].max())), padding=0.02)
        self._transmission_caption.setText(
            "Measured transit depth ± 1σ and model integrated over inferred channel top-hats; "
            "no instrument line-spread convolution or parameter fit.")
        self._chart_tabs.setCurrentWidget(self._transmission_plot)
        assumptions = "; ".join(model.meta["assumptions"])
        self._report.setPlainText(
            "petitRADTRANS forward model calculated. No parameters were fitted.\n"
            f"Channels compared: {int(valid.sum())}; χ²/N = "
            f"{comparison.chi_squared_per_point:.3g} (not reduced χ²; no fit was performed).\n"
            f"Assumptions: {assumptions}\n"
            "Model values were integrated over inferred top-hat channel edges. Instrument line-spread "
            "convolution, correlated noise, and parameter fitting are not included; treat this as a "
            "diagnostic visual comparison only.")
        self._btn_atmosphere_model.setEnabled(True)
        self._btn_abundance_scan.setEnabled(True)

    def _on_atmosphere_model_failed(self, message: str) -> None:
        self._report.setPlainText(f"petitRADTRANS model failed:\n{message}")
        self._btn_atmosphere_model.setEnabled(self._transmission_spectrum is not None)
        self._btn_abundance_scan.setEnabled(self._transmission_spectrum is not None)

    def _on_scan_abundance(self) -> None:
        if self._transmission_spectrum is None or self._series is None:
            return
        from .atmosphere_dialog import PetitRadtransSettingsDialog

        planet_radius_rjup = 1.0
        if self._params is not None:
            planet_radius_rjup = (self._params.rp_rs * self._stellar().radius_rsun
                                  * R_SUN_KM / R_JUP_KM)
        elif self._planet is not None:
            archive_radius = self._planet.get("pl_radj")
            if archive_radius is not None:
                planet_radius_rjup = float(archive_radius)
        dialog = PetitRadtransSettingsDialog(self._stellar().radius_rsun, planet_radius_rjup, self)
        if dialog.exec() != dialog.DialogCode.Accepted or dialog.parameters is None:
            return
        parameters = dialog.parameters
        species, accepted = QInputDialog.getItem(
            self, "Select gas to scan", "Scan the volume mixing ratio of:", list(parameters.mixing_ratios),
            0, False)
        if not accepted:
            return
        self._abundance_curve.setData([], [])
        measured = bin_depth_spectrum(self._transmission_spectrum, self._bin_size.value())
        fixed_vmr = sum(parameters.mixing_ratios.values()) - parameters.mixing_ratios[species]
        lo = max(1e-12, parameters.mixing_ratios[species] / 1000.0)
        hi = min(parameters.mixing_ratios[species] * 1000.0, (1.0 - fixed_vmr) * 0.99)
        if hi <= lo:
            self._report.setPlainText("Cannot scan this gas: the configured VMR leaves no valid scan interval.")
            return
        grid = np.logspace(np.log10(lo), np.log10(hi), 17)
        self._scan_run_series = self._series
        self._scan_run_spectrum = self._transmission_spectrum
        self._scan_run_bin_size = self._bin_size.value()
        self._btn_abundance_scan.setEnabled(False)
        self._btn_atmosphere_model.setEnabled(False)
        self._report.setPlainText(
            f"Scanning {species} over {lo:.2g}–{hi:.2g} VMR; petitRADTRANS will evaluate 17 models. "
            "Other atmospheric inputs and gas abundances are held fixed.")
        run_async(
            self,
            lambda: scan_single_gas_abundance(
                parameters, measured, species, grid),
            self._on_abundance_scan_ready,
            self._on_abundance_scan_failed,
        )

    def _on_abundance_scan_ready(self, result: AbundanceScanResult) -> None:
        if (self._series is not getattr(self, "_scan_run_series", None)
                or self._transmission_spectrum is not getattr(self, "_scan_run_spectrum", None)
                or self._bin_size.value() != getattr(self, "_scan_run_bin_size", None)):
            self._report.setPlainText("Discarded abundance scan because the source spectrum or binning changed.")
            self._btn_abundance_scan.setEnabled(self._transmission_spectrum is not None)
            self._btn_atmosphere_model.setEnabled(self._transmission_spectrum is not None)
            return
        self._abundance_curve.setData(result.volume_mixing_ratio_grid, result.delta_chi_squared)
        self._atmosphere_model_curve.setData(
            result.wavelength_um, result.best_model_depth_ppm,
            pen=pg.mkPen("#b39ddb", width=2), symbol="o", symbolSize=4,
            symbolBrush=(179, 157, 219, 210))
        measured = bin_depth_spectrum(self._transmission_spectrum, self._bin_size.value())
        residual = measured.depth * 1e6 - result.best_model_depth_ppm
        valid = np.isfinite(residual) & np.isfinite(measured.depth_err) & (measured.depth_err > 0.0)
        self._residual_curve.setData(measured.wavelength_um[valid], residual[valid])
        self._residual_caption.setText(
            f"Measured − best conditional {result.species} model; model includes fitted constant depth offset.")
        self._abundance_caption.setText(
            f"Conditional profile scan for {result.species}; all other gas VMRs and atmospheric inputs fixed.")
        lower = (f"{result.lower_1sigma_vmr:.3g}" if result.lower_1sigma_vmr is not None
                 else f"not established within scan ({result.volume_mixing_ratio_grid[0]:.3g})")
        upper = (f"{result.upper_1sigma_vmr:.3g}" if result.upper_1sigma_vmr is not None
                 else f"not established within scan ({result.volume_mixing_ratio_grid[-1]:.3g})")
        if result.conditional_limit_kind == "upper":
            interval_text = (f"conditional upper limit: VMR < {result.conditional_limit_vmr:.3g} "
                             "(best fit at lower scan edge)")
        elif result.conditional_limit_kind == "lower":
            interval_text = (f"conditional lower limit: VMR > {result.conditional_limit_vmr:.3g} "
                             "(best fit at upper scan edge)")
        else:
            interval_text = f"conditional Δχ²=1 interval: {lower} – {upper}"
        self._chart_tabs.setCurrentWidget(self._abundance_plot)
        self._report.setPlainText(
            f"Conditional single-gas profile scan: {result.species}\n"
            f"Best-fit volume mixing ratio: {result.best_volume_mixing_ratio:.4g}\n"
            f"{interval_text}\n"
            f"Minimum χ²: {result.best_chi_squared:.3g}\n"
            f"Fitted constant transit-depth offset: {result.best_depth_offset_ppm:.2f} ppm\n"
            f"Channels: {result.n_channels}; scan grid: "
            f"{result.volume_mixing_ratio_grid[0]:.3g}–{result.volume_mixing_ratio_grid[-1]:.3g}.\n"
            "This is an exploratory profile, not a posterior or calibrated abundance measurement. "
            "All other gases, temperature, gravity, reference radius/pressure, and model assumptions "
            "were fixed. Instrument response and correlated noise were not modeled; the Δχ²=1 interval "
            "is an asymptotic one-parameter approximation only."
            + ("\nWarnings:\n- " + "\n- ".join(result.warnings) if result.warnings else ""))
        self._btn_abundance_scan.setEnabled(True)
        self._btn_atmosphere_model.setEnabled(True)

    def _on_abundance_scan_failed(self, message: str) -> None:
        self._report.setPlainText(f"Abundance scan failed:\n{message}")
        self._btn_abundance_scan.setEnabled(self._transmission_spectrum is not None)
        self._btn_atmosphere_model.setEnabled(self._transmission_spectrum is not None)

    def _show_transmission_spectrum(self, spec: DepthSpectrum) -> None:
        """Display the measured transit-depth spectrum, keeping it distinct from fitter input."""
        self._atmosphere_model_curve.setData([], [])
        self._residual_curve.setData([], [])
        self._abundance_curve.setData([], [])
        self._abundance_caption.setText(
            "Scan one configured gas to see conditional Δχ² versus volume mixing ratio.")
        self._residual_caption.setText(
            "Generate a forward model to compare channel-integrated residuals.")
        depth_ppm = spec.depth * 1e6
        error_ppm = spec.depth_err * 1e6
        self._transmission_points.setData(spec.wavelength_um, depth_ppm)
        self._transmission_errors.setData(
            x=spec.wavelength_um, y=depth_ppm, top=error_ppm, bottom=error_ppm)
        self._transmission_caption.setText(
            f"Measured transit depth ± 1σ, {self._bin_size.value()} channel(s) per bin; "
            "this is not a molecular-abundance estimate.")
        self._transmission_plot.getViewBox().setRange(
            xRange=(float(spec.wavelength_um.min()), float(spec.wavelength_um.max())), padding=0.02)
        self._chart_tabs.setCurrentWidget(self._transmission_plot)

    def _on_spectral(self) -> None:
        """Simulate spectral light curves for the current gas mix, extract, hand to the fitter."""
        from ..core.mock_data import make_mock_lightcurve, make_mock_spectral_lightcurves

        params = self._params or (self._truth or make_mock_lightcurve()[1])
        t, flux, wl = make_mock_spectral_lightcurves(self._engine, params)
        spec = extract_transmission_spectrum(t, flux, wl, params)
        self.observation_ready.emit(depth_spectrum_to_observation(spec))

    # ------------------------------------------------------------------ the clock
    def _current_time(self) -> float:
        return self._t

    def _set_clock(self, t: float) -> None:
        """Set the clock to ``t`` and move the slider without feedback loops."""
        span = max(self._t_max - self._t_min, 1e-12)
        frac = float(np.clip((t - self._t_min) / span, 0.0, 1.0))
        self._slider.blockSignals(True)
        self._slider.setValue(int(round(frac * self.SLIDER_STEPS)))
        self._slider.blockSignals(False)
        self.set_time(t)

    def _on_scrub(self, pos: int) -> None:
        self.set_time(self._t_min + pos / self.SLIDER_STEPS * (self._t_max - self._t_min))

    def _base_rate(self) -> float:
        """Days of simulated time per real second at 1x: a transit lasts ~5 s."""
        if self._params is not None and self._params.duration_days > 0:
            return self._params.duration_days / 5.0
        return (self._t_max - self._t_min) / 30.0

    def _on_speed(self, pos: int) -> None:
        mult = self.SPEED.to_value(pos)
        self._speed_label.setText(f"{mult:.2f}×  ({self._base_rate() * mult * 24.0:.2f} h/s)")

    @property
    def speed_multiplier(self) -> float:
        return self.SPEED.to_value(self._speed.value())

    def _on_play(self, on: bool) -> None:
        self._play.setText("⏸ Pause" if on else "▶ Play")
        if on:
            self._last_tick = time.perf_counter()
            self._on_speed(self._speed.value())
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _advance(self, dt: Optional[float] = None) -> None:
        """Advance by real elapsed time x speed, so the animation rate is frame-rate independent."""
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.2) if dt is None else dt
        self._last_tick = now
        t = self._t + self._base_rate() * self.speed_multiplier * dt
        if t > self._t_max:                                    # wrap around
            t = self._t_min + (t - self._t_max) % max(self._t_max - self._t_min, 1e-12)
        self._set_clock(t)

    def set_time(self, t: float) -> None:
        """Move the tracker bar, the sky-view planet and the orbit-view planet to ``t`` (days)."""
        self._t = float(t)
        self._tracker.setValue(t)
        self._update_snapshot(t)
        if self._params is None:
            self._clock.setText(f"t = {t:9.4f} d")
            return
        p = self._params
        x, y, z = (float(v[0]) for v in orbit_positions(np.array([t]), p))
        self._sky.set_position(x, y, z)
        self._view3d.set_position(x, y, z)
        flux = float(transit_model(np.array([t]), p)[0])
        phase = ((t - p.t0) / p.period + 0.5) % 1.0 - 0.5
        self._clock.setText(f"t = {t:9.4f} d   phase {phase:+.3f}   flux {flux:.5f}")

    # ------------------------------------------------------------- project state
    def capture(self):
        """Return the transit inputs, including the source series when present."""
        st = self._stellar()
        return (self._lc, {"radius_rsun": st.radius_rsun, "mass_msun": st.mass_msun,
                           "teff_k": st.teff_k}, self._spin_w[1].value(), self._series,
                self._transmission_spectrum)

    def restore(self, lc, stellar: dict, window: float, analyze: bool = True,
                spectral_series: Optional[SpectralTimeSeries] = None,
                depth_spectrum: Optional[DepthSpectrum] = None) -> None:
        """Re-install inputs, then re-derive the analysis (results are never stored)."""
        if stellar:
            self._spin_r[1].setValue(stellar.get("radius_rsun", 1.0))
            self._spin_m[1].setValue(stellar.get("mass_msun", 1.0))
            self._spin_t[1].setValue(stellar.get("teff_k", 5772.0))
        self._spin_w[1].setValue(window)
        if spectral_series is not None:
            record = spectral_series.meta.get("planet_record")
            if isinstance(record, dict) and record.get("values"):
                self.use_planet(PlanetRecord(
                    values=record.get("values", {}),
                    query=str(record.get("query", "")), url=str(record.get("url", "")),
                    retrieved=str(record.get("retrieved", "")),
                    refs=record.get("refs", {})))
            self.set_series(spectral_series)
            if analyze:
                self.analyze(blocking=True)
                try:
                    spec = spectral_series.transmission_spectrum(self._extraction_params())
                except (TransitError, ArchiveError) as exc:
                    self._report.setPlainText(self._report.toPlainText()
                                              + f"\nCould not re-extract saved transmission spectrum:\n{exc}")
                else:
                    self._transmission_spectrum = spec
                    self._btn_identify.setEnabled(True)
                    self._btn_export_depth.setEnabled(True)
                    self._btn_atmosphere_model.setEnabled(True)
                    self._btn_abundance_scan.setEnabled(True)
                    self._render_transmission_spectrum()
        elif lc is not None:
            self.set_lightcurve(lc)
            if analyze:
                self.analyze(blocking=True)
        if depth_spectrum is not None and self._transmission_spectrum is None:
            self._transmission_spectrum = depth_spectrum
            self._btn_export_depth.setEnabled(True)
            self._btn_atmosphere_model.setEnabled(True)
            self._btn_abundance_scan.setEnabled(self._series is not None)
            self._render_transmission_spectrum()

    def shutdown(self) -> None:
        self._play_timer.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._view3d.shutdown()
