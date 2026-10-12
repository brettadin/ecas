"""Planet lookup: NASA Exoplanet Archive parameters with their literature references."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit,
    QPushButton, QVBoxLayout,
)

from ..core.archives import (
    ATMOSPHERES_URL, ArchiveError, PlanetRecord, download_mast_lightcurve,
    download_mast_spectral_timeseries, fetch_planet, search_mast_lightcurves,
    search_mast_spectral_timeseries, search_planet_names,
)
from ..core.timeseries import load_x1dints
from .tasks import run_async
from .theme import ERR, OK, muted, primary

CACHE_DIR = Path.home() / ".ecas" / "cache"


class PlanetLookupDialog(QDialog):
    """Fetch one planet's default parameter set (network access happens off the UI thread)."""

    def __init__(self, parent=None, fetch: Optional[Callable[[str], str]] = None, cache_dir: Optional[Path] = CACHE_DIR):
        super().__init__(parent)
        self.setWindowTitle("Look up planet · NASA Exoplanet Archive")
        self.resize(640, 560)
        self._fetch, self._cache = fetch, cache_dir
        self.record: Optional[PlanetRecord] = None

        head = QLabel("NASA Exoplanet Archive")
        head.setProperty("heading", True)
        note = QLabel("Fetches the archive's default parameter set for one planet, with the literature reference "
                      "it gives for each group of values.  Names are exact, e.g. “WASP-39 b”.")
        note.setWordWrap(True)
        muted(note)
        self._name = QLineEdit()
        self._name.setPlaceholderText("Planet name, e.g. WASP-39 b")
        self._name.returnPressed.connect(self.fetch_now)
        self._btn_fetch = QPushButton("Fetch")
        primary(self._btn_fetch)
        self._btn_fetch.clicked.connect(self.fetch_now)
        b_search = QPushButton("Search names")
        b_search.clicked.connect(self.search_now)
        self._suggest = QComboBox()
        self._suggest.setVisible(False)
        self._suggest.activated.connect(lambda _i: self._name.setText(self._suggest.currentText()))
        row = QHBoxLayout()
        row.addWidget(self._name, 1)
        row.addWidget(b_search)
        row.addWidget(self._btn_fetch)
        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        mono = QFont("monospace")
        mono.setPointSizeF(9.0)
        self._summary.setFont(mono)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._btn_use = QPushButton("Use these values")
        primary(self._btn_use)
        self._btn_use.setEnabled(False)
        self._btn_use.clicked.connect(self.accept)
        b_spec = QPushButton("Published spectra (archive)…")
        b_spec.setToolTip("Opens the archive's Atmospheric Spectroscopy Table page in your browser.")
        b_spec.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(ATMOSPHERES_URL)))
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(b_spec)
        bottom.addStretch(1)
        bottom.addWidget(b_close)
        bottom.addWidget(self._btn_use)
        lay = QVBoxLayout(self)
        for w in (head, note):
            lay.addWidget(w)
        lay.addLayout(row)
        for w in (self._suggest, self._summary, self._status):
            lay.addWidget(w)
        lay.addLayout(bottom)

    def _busy(self, on: bool, msg: str = "") -> None:
        self._btn_fetch.setEnabled(not on)
        self._status.setText(msg)

    def fetch_now(self) -> None:
        name = self._name.text()
        self._busy(True, "Contacting the archive…")
        run_async(self, lambda: fetch_planet(name, self._fetch, self._cache), self._on_record, self._on_error)

    def search_now(self) -> None:
        text = self._name.text()
        self._busy(True, "Searching…")
        run_async(self, lambda: search_planet_names(text, fetch=self._fetch, cache_dir=self._cache),
                  self._on_names, self._on_error)

    def _on_record(self, rec: PlanetRecord) -> None:
        self.record = rec
        self._summary.setPlainText("\n".join(rec.summary_lines()))
        self._btn_use.setEnabled(True)
        self._busy(False, f"<span style='color:{OK}'>Retrieved {rec.name} · {rec.retrieved}. "
                          "Using these values records the citation automatically.</span>")

    def _on_names(self, names) -> None:
        self._suggest.clear()
        self._suggest.addItems(names)
        self._suggest.setVisible(bool(names))
        self._busy(False, f"{len(names)} match(es)." if names else "No matching names.")

    def _on_error(self, msg: str) -> None:
        self.record = None
        self._btn_use.setEnabled(False)
        self._busy(False, f"<span style='color:{ERR}'>{msg}</span>")


class TessLightCurveDialog(QDialog):
    """Find a TESS observation on MAST and download its light-curve FITS product."""

    def __init__(self, target: str, parent=None, cache_dir: Optional[Path] = CACHE_DIR,
                 observations=None):
        super().__init__(parent)
        self.setWindowTitle(f"Fetch TESS photometry · {target}")
        self.resize(700, 460)
        self._target, self._cache, self._observations = target, cache_dir, observations
        self.path: Optional[Path] = None
        self.observation = None

        title = QLabel(f"Public TESS light curves for {target}")
        title.setProperty("heading", True)
        note = QLabel("Select an observation to download its calibrated FITS light curve from MAST. "
                      "The file will be loaded into the Photometry panel.")
        note.setWordWrap(True)
        muted(note)
        self._items = QListWidget()
        self._items.currentItemChanged.connect(self._selection_changed)
        self._status = QLabel("Searching MAST…")
        self._status.setWordWrap(True)
        self._btn_download = QPushButton("Download and load")
        primary(self._btn_download)
        self._btn_download.setEnabled(False)
        self._btn_download.clicked.connect(self._download_selected)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(self._status, 1)
        bottom.addWidget(b_close)
        bottom.addWidget(self._btn_download)
        lay = QVBoxLayout(self)
        lay.addWidget(title)
        lay.addWidget(note)
        lay.addWidget(self._items, 1)
        lay.addLayout(bottom)
        self._search()

    def _search(self) -> None:
        run_async(self, lambda: search_mast_lightcurves(self._target, self._observations),
                  self._on_observations, self._on_error)

    def _on_observations(self, observations) -> None:
        self._items.clear()
        for observation in observations:
            obs_id = str(observation.get("obs_id", ""))
            instrument = str(observation.get("instrument_name", "TESS"))
            target = str(observation.get("target_name", self._target))
            start = observation.get("t_min")
            date = f" · MJD {float(start):.2f}" if start is not None else ""
            distance = observation.get("distance")
            separation = f" · {float(distance):.1f}″" if distance is not None else ""
            cadence = observation.get("t_exptime")
            exposure = f" · {float(cadence):g} s" if cadence is not None else ""
            item = QListWidgetItem(f"{obs_id}  ·  {instrument}  ·  {target}{date}{separation}{exposure}")
            item.setData(Qt.ItemDataRole.UserRole, observation)
            self._items.addItem(item)
        self._status.setText(f"{len(observations)} observation(s) found. Select one to download.")
        if observations:
            self._items.setCurrentRow(0)

    def _selection_changed(self, current, _previous) -> None:
        self._btn_download.setEnabled(current is not None)

    def _download_selected(self) -> None:
        item = self._items.currentItem()
        if item is None:
            return
        obs_id = str(item.data(Qt.ItemDataRole.UserRole).get("obs_id", ""))
        if not obs_id:
            self._on_error("The selected MAST observation has no observation ID.")
            return
        self._btn_download.setEnabled(False)
        self._status.setText("Downloading light curve from MAST…")
        dest = self._cache / "tess-lightcurves" if self._cache is not None else Path.home() / ".ecas" / "cache" / "tess-lightcurves"
        run_async(self, lambda: download_mast_lightcurve(obs_id, dest, self._observations),
                  self._on_downloaded, self._on_error)

    def _on_downloaded(self, path: Path) -> None:
        self.path = path
        item = self._items.currentItem()
        self.observation = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        self.accept()

    def _on_error(self, message: str) -> None:
        self._btn_download.setEnabled(self._items.currentItem() is not None)
        self._status.setText(f"<span style='color:{ERR}'>{message}</span>")


class JwstSpectralTimeSeriesDialog(QDialog):
    """Find, download, and validate a JWST per-integration spectrum from MAST."""

    def __init__(self, target: str, parent=None, cache_dir: Optional[Path] = CACHE_DIR,
                 observations=None):
        super().__init__(parent)
        self.setWindowTitle(f"Fetch JWST spectral time series · {target}")
        self.resize(760, 480)
        self._target, self._cache, self._observations = target, cache_dir, observations
        self.path: Optional[Path] = None
        self.series = None
        self.product = None

        title = QLabel(f"JWST spectra during transit for {target}")
        title.setProperty("heading", True)
        note = QLabel("Select a calibrated x1dints product. MAST results can include non-transit or "
                      "single-spectrum files; downloads are checked for a usable time series before loading.")
        note.setWordWrap(True)
        muted(note)
        self._items = QListWidget()
        self._items.currentItemChanged.connect(self._selection_changed)
        self._status = QLabel("Searching MAST…")
        self._status.setWordWrap(True)
        self._btn_download = QPushButton("Download and load")
        primary(self._btn_download)
        self._btn_download.setEnabled(False)
        self._btn_download.clicked.connect(self._download_selected)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(self._status, 1)
        bottom.addWidget(b_close)
        bottom.addWidget(self._btn_download)
        lay = QVBoxLayout(self)
        lay.addWidget(title)
        lay.addWidget(note)
        lay.addWidget(self._items, 1)
        lay.addLayout(bottom)
        run_async(self, lambda: search_mast_spectral_timeseries(self._target, self._observations),
                  self._on_products, self._on_error)

    def _on_products(self, products) -> None:
        self._items.clear()
        for product in products:
            date = f"MJD {product['t_min']:.2f}"
            size_mb = product["size"] / (1024.0 * 1024.0)
            item = QListWidgetItem(
                f"{date} · {product['instrument_name']} · proposal {product['proposal_id']} · "
                f"{size_mb:.0f} MB · {product['filename']}")
            item.setData(Qt.ItemDataRole.UserRole, product)
            self._items.addItem(item)
        self._status.setText(f"{len(products)} JWST spectrum product(s) found. Select one to download.")
        if products:
            self._items.setCurrentRow(0)

    def _selection_changed(self, current, _previous) -> None:
        self._btn_download.setEnabled(current is not None)

    def _download_selected(self) -> None:
        item = self._items.currentItem()
        if item is None:
            return
        product = item.data(Qt.ItemDataRole.UserRole)
        self._btn_download.setEnabled(False)
        self._status.setText("Downloading and checking the JWST time series…")
        dest = self._cache / "jwst-timeseries" if self._cache is not None else (
            Path.home() / ".ecas" / "cache" / "jwst-timeseries")

        def fetch():
            path = download_mast_spectral_timeseries(product, dest, self._observations)
            return path, load_x1dints(path)

        run_async(self, fetch, self._on_downloaded, self._on_error)

    def _on_downloaded(self, result) -> None:
        self.path, self.series = result
        self.product = self._items.currentItem().data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_error(self, message: str) -> None:
        self._btn_download.setEnabled(self._items.currentItem() is not None)
        self._status.setText(f"<span style='color:{ERR}'>{message}</span>")
