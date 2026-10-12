"""Settings dialog for the optional petitRADTRANS clear-atmosphere forward model."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout,
)

from ..core.atmosphere import AtmosphereModelError, PetitRadtransParameters
from .theme import muted


class PetitRadtransSettingsDialog(QDialog):
    """Collect explicit model parameters; this creates a spectrum but performs no retrieval."""

    def __init__(self, stellar_radius_rsun: float, planet_radius_rjup: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate petitRADTRANS model")
        self.setMinimumWidth(460)
        self.parameters: PetitRadtransParameters | None = None

        self._temperature = self._number(1200.0, 100.0, 5000.0, 0, " K")
        self._gravity = self._number(1000.0, 10.0, 100000.0, 1, " cm s⁻²")
        self._planet_radius = self._number(planet_radius_rjup, 0.1, 10.0, 4, " R_Jup")
        self._stellar_radius = self._number(stellar_radius_rsun, 0.05, 50.0, 4, " R☉")
        self._reference_pressure = self._number(0.01, 1e-8, 100.0, 8, " bar")
        self._mixing_ratios = QLineEdit("H2O=1e-3, CO2=1e-4, CH4=1e-5")
        self._mixing_ratios.setToolTip(
            "Comma-separated pRT species=volume-mixing-ratio pairs; supported gases are listed in README.md.")

        note = QLabel(
            "Clear, isothermal atmosphere; constant gas mixing ratios with altitude; H₂/He background. "
            "These settings can generate a forward-model overlay or configure a separate, conditional "
            "one-gas profile scan. Neither is a calibrated abundance retrieval. "
            "Required pRT opacity files must already be installed."
        )
        note.setWordWrap(True)
        muted(note)
        form = QFormLayout()
        form.addRow("Isothermal temperature", self._temperature)
        form.addRow("Reference gravity", self._gravity)
        form.addRow("Planet radius at reference pressure", self._planet_radius)
        form.addRow("Stellar radius", self._stellar_radius)
        form.addRow("Reference pressure", self._reference_pressure)
        form.addRow("Gas volume mixing ratios", self._mixing_ratios)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(buttons)

    @staticmethod
    def _number(value: float, minimum: float, maximum: float, decimals: int, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(min(max(value, minimum), maximum))
        spin.setSuffix(suffix)
        return spin

    def _accept(self) -> None:
        try:
            mixing_ratios = {}
            for entry in self._mixing_ratios.text().split(","):
                species, separator, value = entry.partition("=")
                species = species.strip()
                if not separator or not species:
                    raise AtmosphereModelError(
                        "Enter comma-separated gas=VMR pairs, for example H2O=1e-3, CO2=1e-4.")
                if species in mixing_ratios:
                    raise AtmosphereModelError(f"Gas {species} is listed more than once.")
                mixing_ratios[species] = float(value.strip())
            self.parameters = PetitRadtransParameters(
                temperature_k=self._temperature.value(),
                gravity_cgs=self._gravity.value(),
                planet_radius_rjup=self._planet_radius.value(),
                stellar_radius_rsun=self._stellar_radius.value(),
                reference_pressure_bar=self._reference_pressure.value(),
                mixing_ratios=mixing_ratios,
            )
        except (AtmosphereModelError, ValueError) as exc:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Invalid atmosphere settings", str(exc))
            return
        self.accept()
