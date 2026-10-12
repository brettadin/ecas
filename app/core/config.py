"""
ECAS configuration manager (dynamic gas injection).

A single JSON file lists the active molecules and where their ORCA ``.out`` track
lives.  Adding a gas to the dashboard is therefore one new JSON entry - no code
change.  Sources may be real paths (resolved relative to the JSON file) or
``mock://<KEY>`` URIs served by :mod:`app.core.mock_data`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from .parser import MOCK_SCHEME
from .physics import LineShape

# Colours are allocated from this palette in registration order when an entry
# has no explicit "color".
PALETTE: Tuple[str, ...] = (
    "#4FC3F7", "#FF8A65", "#81C784", "#BA68C8", "#FFD54F",
    "#4DB6AC", "#F06292", "#A1887F", "#90A4AE", "#DCE775",
)


class ConfigError(ValueError):
    """Raised for malformed configuration files."""


@dataclass(frozen=True)
class GridSpec:
    """Static tracking-index grid: t in [0, 1] mapped onto either physical range."""

    n_points: int = 8192
    wavenumber_range_cm1: Tuple[float, float] = (400.0, 4200.0)
    wavelength_range_um: Tuple[float, float] = (0.1, 25.0)


@dataclass(frozen=True)
class PhysicsDefaults:
    """
    Start-up atmosphere.

    ``envelope_hwhm_cm1`` is an extra Lorentzian width standing in for the
    unresolved rotational envelope of each ORCA fundamental (ORCA yields band
    centres, not rovibrational lines).  ``tau_scale`` converts
    (km/mol x profile) into a dimensionless optical depth for a unit multiplier.
    """

    pressure_atm: float = 1.0
    temperature_k: float = 296.0
    envelope_hwhm_cm1: float = 4.0
    tau_scale: float = 0.08
    line_shape: LineShape = LineShape.VOIGT


@dataclass(frozen=True)
class GasSpec:
    """One registered molecule."""

    gas_id: str
    name: str
    formula: str
    orca_out: str
    mass_amu: float
    gamma0_cm1_atm: float
    n_exponent: float
    color: str
    default_multiplier: float = 1.0
    multiplier_max: float = 10.0
    enabled: bool = True

    def resolved_source(self, base_dir: Path) -> str:
        """Absolute path string for real files; ``mock://`` URIs pass through."""
        if self.orca_out.startswith(MOCK_SCHEME):
            return self.orca_out
        p = Path(self.orca_out)
        return str(p if p.is_absolute() else (base_dir / p).resolve())


@dataclass(frozen=True)
class EcasConfig:
    """Fully validated configuration."""

    grid: GridSpec
    physics: PhysicsDefaults
    gases: Tuple[GasSpec, ...]
    base_dir: Path = field(default_factory=Path.cwd)

    @property
    def active_gases(self) -> Tuple[GasSpec, ...]:
        return tuple(g for g in self.gases if g.enabled)


class ConfigManager:
    """Loads, validates and saves :class:`EcasConfig` objects."""

    DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "gases.json"

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> EcasConfig:
        path = Path(path) if path else cls.DEFAULT_PATH
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Cannot read config {path}: {exc}") from exc
        return cls.from_dict(raw, base_dir=path.parent)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], base_dir: Union[str, Path] = ".") -> EcasConfig:
        grid_raw = dict(raw.get("grid", {}))
        phys_raw = dict(raw.get("physics", {}))
        try:
            grid = GridSpec(
                n_points=int(grid_raw.get("n_points", GridSpec.n_points)),
                wavenumber_range_cm1=tuple(grid_raw.get("wavenumber_range_cm1", (400.0, 4200.0))),
                wavelength_range_um=tuple(grid_raw.get("wavelength_range_um", (0.1, 25.0))),
            )
            phys = PhysicsDefaults(
                pressure_atm=float(phys_raw.get("pressure_atm", 1.0)),
                temperature_k=float(phys_raw.get("temperature_k", 296.0)),
                envelope_hwhm_cm1=float(phys_raw.get("envelope_hwhm_cm1", 4.0)),
                tau_scale=float(phys_raw.get("tau_scale", 0.08)),
                line_shape=LineShape(phys_raw.get("line_shape", "voigt")),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid grid/physics section: {exc}") from exc
        if grid.n_points < 16:
            raise ConfigError("grid.n_points must be >= 16")
        if not (0 < grid.wavenumber_range_cm1[0] < grid.wavenumber_range_cm1[1]):
            raise ConfigError("wavenumber range must be positive and ascending")
        if not (0 < grid.wavelength_range_um[0] < grid.wavelength_range_um[1]):
            raise ConfigError("wavelength range must be positive and ascending")
        if phys.pressure_atm <= 0 or phys.temperature_k <= 0:
            raise ConfigError("pressure and temperature must be > 0")

        gases = []
        seen = set()
        for i, entry in enumerate(raw.get("gases", [])):
            try:
                gas = GasSpec(
                    gas_id=str(entry["id"]),
                    name=str(entry.get("name", entry["id"])),
                    formula=str(entry.get("formula", entry["id"])),
                    orca_out=str(entry["orca_out"]),
                    mass_amu=float(entry["mass_amu"]),
                    gamma0_cm1_atm=float(entry.get("gamma0_cm1_atm", 0.07)),
                    n_exponent=float(entry.get("n_exponent", 0.7)),
                    color=str(entry.get("color") or PALETTE[i % len(PALETTE)]),
                    default_multiplier=float(entry.get("default_multiplier", 1.0)),
                    multiplier_max=float(entry.get("multiplier_max", 10.0)),
                    enabled=bool(entry.get("enabled", True)),
                )
            except KeyError as exc:
                raise ConfigError(f"gases[{i}] is missing required key {exc}") from exc
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"gases[{i}] has an invalid value: {exc}") from exc
            if gas.gas_id in seen:
                raise ConfigError(f"duplicate gas id {gas.gas_id!r}")
            if gas.mass_amu <= 0 or gas.multiplier_max <= 0:
                raise ConfigError(f"gas {gas.gas_id!r}: mass and multiplier_max must be > 0")
            seen.add(gas.gas_id)
            gases.append(gas)

        cfg = EcasConfig(grid, phys, tuple(gases), Path(base_dir))
        if not cfg.active_gases:
            raise ConfigError("no enabled gases in configuration")
        return cfg

    @classmethod
    def to_dict(cls, cfg: EcasConfig) -> Dict[str, Any]:
        """Serialise back to plain JSON-compatible data (project-state saving)."""
        return {
            "grid": {
                "n_points": cfg.grid.n_points,
                "wavenumber_range_cm1": list(cfg.grid.wavenumber_range_cm1),
                "wavelength_range_um": list(cfg.grid.wavelength_range_um),
            },
            "physics": {
                "pressure_atm": cfg.physics.pressure_atm,
                "temperature_k": cfg.physics.temperature_k,
                "envelope_hwhm_cm1": cfg.physics.envelope_hwhm_cm1,
                "tau_scale": cfg.physics.tau_scale,
                "line_shape": cfg.physics.line_shape.value,
            },
            "gases": [
                {
                    "id": g.gas_id, "name": g.name, "formula": g.formula,
                    "orca_out": g.orca_out, "mass_amu": g.mass_amu,
                    "gamma0_cm1_atm": g.gamma0_cm1_atm, "n_exponent": g.n_exponent,
                    "color": g.color, "default_multiplier": g.default_multiplier,
                    "multiplier_max": g.multiplier_max, "enabled": g.enabled,
                }
                for g in cfg.gases
            ],
        }

    @classmethod
    def save(cls, cfg: EcasConfig, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(cls.to_dict(cfg), indent=2), encoding="utf-8")
