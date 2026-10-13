"""
ECAS visual design: one palette, one stylesheet, a few reusable widgets.

Design intent: a calm, dense, instrument-style dark UI (think observatory control software):
low-contrast chrome so the data is the brightest thing on screen, a single cyan accent for the
active/primary things, warm amber for "needs attention", red only for errors/anomalies.
"""
from __future__ import annotations

from typing import Optional

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication, QFrame, QSizePolicy, QToolButton, QVBoxLayout, QWidget

BG = "#0e1218"          # window
PANEL = "#151b24"       # cards / panels
PANEL_HI = "#1c2430"    # raised / hover
BORDER = "#263041"
TEXT = "#d5dce6"
MUTED = "#8593a6"
ACCENT = "#4fc3f7"      # primary / active
ACCENT_DIM = "#1f5f7d"
WARN = "#ffb74d"
OK = "#81c784"
ERR = "#ff6b6b"
PLOT_BG = "#0b0f15"


def configure_plots(use_opengl: bool = False) -> None:
    """pyqtgraph defaults consistent with the theme."""
    pg.setConfigOptions(antialias=False, background=PLOT_BG, foreground=MUTED, useOpenGL=use_opengl)


BASE_POINT_SIZE = 10.0
TEXT_SCALES = {"Small": 0.9, "Normal": 1.15, "Large": 1.4, "Extra large": 1.7}
DEFAULT_TEXT_SCALE = "Normal"


def mono_font() -> QFont:
    """Monospace font at the current application text size (for numbers and reports)."""
    font = QFont(QApplication.font())
    font.setFamily("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def apply_text_scale(app: QApplication, scale: float) -> None:
    """Scale every text element: application font, and the tick and axis labels of all charts."""
    font = QFont(app.font())
    font.setPointSizeF(BASE_POINT_SIZE * float(scale))
    app.setFont(font)
    for widget in app.allWidgets():
        if isinstance(widget, pg.PlotWidget):
            for name in ("bottom", "left", "top", "right"):
                axis = widget.getAxis(name)
                axis.setStyle(tickFont=font)
                try:
                    axis.label.item.setFont(font)
                except AttributeError:
                    pass


def apply_theme(app: QApplication, text_scale: Optional[float] = None) -> None:
    """Fusion style + dark palette + stylesheet."""
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.ColorRole.Window, BG), (QPalette.ColorRole.WindowText, TEXT),
                        (QPalette.ColorRole.Base, PANEL), (QPalette.ColorRole.AlternateBase, PANEL_HI),
                        (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.Button, PANEL_HI),
                        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT_DIM),
                        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, PANEL_HI),
                        (QPalette.ColorRole.ToolTipText, TEXT), (QPalette.ColorRole.PlaceholderText, MUTED)):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    font = QFont("Inter")
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setPointSizeF(BASE_POINT_SIZE * (text_scale if text_scale is not None
                                          else TEXT_SCALES[DEFAULT_TEXT_SCALE]))
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)


STYLESHEET = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {PANEL_HI}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}

QMenuBar {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 5px 12px; background: transparent; }}
QMenuBar::item:selected {{ background: {PANEL_HI}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QStatusBar {{ background: {BG}; color: {MUTED}; border-top: 1px solid {BORDER}; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; top: -1px; background: {BG}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 18px; margin-right: 2px;
               border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; font-weight: 600; }}

QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px; padding: 12px 10px 8px 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {MUTED}; font-weight: 600; }}

QFrame#sectionBody {{ background: {PANEL}; border: 1px solid {BORDER}; border-top: none;
                     border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; }}
QToolButton#sectionHeader {{ background: {PANEL_HI}; border: 1px solid {BORDER}; border-radius: 8px;
                            padding: 7px 10px; font-weight: 600; text-align: left; color: {TEXT}; }}
QToolButton#sectionHeader:hover {{ border-color: {ACCENT_DIM}; }}
QToolButton#sectionHeader:checked {{ border-bottom-left-radius: 0; border-bottom-right-radius: 0; }}

QPushButton {{ background: {PANEL_HI}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT_DIM}; background: #222c3a; }}
QPushButton:pressed {{ background: {ACCENT_DIM}; }}
QPushButton:disabled {{ color: #566275; background: {PANEL}; border-color: {BORDER}; }}
QPushButton[primary="true"] {{ background: {ACCENT_DIM}; border-color: {ACCENT}; color: #ffffff; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: #277a9d; }}
QPushButton[primary="true"]:disabled {{ background: {PANEL}; border-color: {BORDER}; color: #566275; }}
QPushButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QListWidget, QTableWidget, QTextEdit {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 6px;
    selection-background-color: {ACCENT_DIM}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_DIM}; }}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {ACCENT_DIM}; color: #ffffff; }}
QListWidget::item:hover:!selected {{ background: {PANEL_HI}; }}
QHeaderView::section {{ background: {PANEL_HI}; color: {MUTED}; border: none; border-right: 1px solid {BORDER};
                       padding: 5px 8px; font-weight: 600; }}
QTableWidget {{ gridline-color: {BORDER}; alternate-background-color: #121821; }}
QTableWidget::item:selected {{ background: {ACCENT_DIM}; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid #3a475c; background: {PANEL}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 4px; background: #2a3546; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: #8fdcff; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2f3b4e; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #3d4d66; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #2f3b4e; border-radius: 4px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:hover {{ background: {ACCENT_DIM}; }}
QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel[heading="true"] {{ font-weight: 700; color: {TEXT}; }}
"""


class CollapsibleSection(QWidget):
    """A titled card that folds away: ``section.body_layout()`` is where the content goes."""

    def __init__(self, title: str, expanded: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._toggle = QToolButton()
        self._toggle.setObjectName("sectionHeader")
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._body = QFrame()
        self._body.setObjectName("sectionBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(10, 8, 10, 10)
        self._body_layout.setSpacing(6)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(0)
        outer.addWidget(self._toggle)
        outer.addWidget(self._body)
        self._toggle.toggled.connect(self._on_toggled)
        self._on_toggled(expanded)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def set_expanded(self, on: bool) -> None:
        self._toggle.setChecked(on)

    def _on_toggled(self, on: bool) -> None:
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self._body.setVisible(on)


def primary(button) -> None:
    """Mark a QPushButton as the main action of its panel (accent style)."""
    button.setProperty("primary", True)
    button.style().unpolish(button)
    button.style().polish(button)


def muted(label) -> None:
    label.setProperty("muted", True)
    label.style().unpolish(label)
    label.style().polish(label)
