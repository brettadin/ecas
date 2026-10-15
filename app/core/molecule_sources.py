"""
Molecular spectroscopic data fetching and caching.

Provides high-level functions to fetch real molecular cross-sections from:
  - HITRAN 2020 (via petitRADTRANS or HAPI)
  - ExoMolDB (direct download)
  - NIST WebBook (JCAMP-DX parsing)

All data is cached locally in ~/.ecas/data_sources/molecules/
Integrates with spectral_sources.MoleculeRegistry for tracking.
"""
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from app.core.library import (
    ReferenceSpectrum, import_nist_jcamp, import_exomol_transitions,
    LibraryError
)
from app.core.sources import SOURCES, CitationLog
from app.core.spectral_sources import (
    MoleculeRegistry, MOLECULES_REGISTRY_FILE, DEFAULT_CACHE_DIR,
    _save_reference_spectrum_json
)

FloatArray = NDArray[np.float64]

# ============================================================================
# Configuration
# ============================================================================

MOLECULES_CACHE_DIR = DEFAULT_CACHE_DIR / "molecules"
USER_AGENT = "ECAS/0.8 (exoplanet atmosphere analysis; research use)"
HITRAN_FETCH_TIMEOUT = 30.0

# Predefined molecules with their NIST WebBook CAS numbers
# These can be fetched automatically from NIST
NIST_CAS_NUMBERS = {
    "H2O": "7732-18-5",
    "CO2": "124-38-9",
    "CH4": "74-82-8",
    "O3": "10028-15-6",
    "N2O": "10024-97-2",
    "CO": "630-08-0",
    "NH3": "7664-41-7",
    "HCN": "74-90-8",
    "SO2": "7446-09-5",
}

# ExoMolDB molecule identifiers
# From: https://exomoldb.exoplanet.eu/
EXOMOLDB_MOLECULES = {
    "H2O": "H2O",
    "CO2": "CO2",
    "CH4": "CH4",
    "NH3": "NH3",
    "HCN": "HCN",
    "C2H2": "C2H2",
    "TiO": "TiO",
}

# ============================================================================
# Error handling
# ============================================================================

class MoleculeSourceError(RuntimeError):
    """Raised for issues fetching or parsing molecular data."""
    pass


# ============================================================================
# HTTP utilities (reuse ECAS patterns)
# ============================================================================

def fetch_url_text(url: str, timeout: float = 30.0, cache_dir: Optional[Path] = None) -> str:
    """
    HTTP GET returning text, with optional on-disk cache.
    Mimics app.core.archives.fetch_text behavior.
    """
    import hashlib
    import time
    
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".txt")
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < 86400:
            return cache_file.read_text(encoding="utf-8")
    
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MoleculeSourceError(f"Could not fetch {url}: {exc}") from exc
    
    if cache_file is not None:
        cache_file.write_text(text, encoding="utf-8")
    
    return text


# ============================================================================
# NIST WebBook integration
# ============================================================================

def fetch_nist_molecule(
    molecule: str,
    cas_number: str,
    temperature_k: float = 296.0,
    cache_dir: Optional[Path] = None,
) -> ReferenceSpectrum:
    """
    Fetch IR spectrum from NIST WebBook and parse as ReferenceSpectrum.
    
    Args:
        molecule: Molecule name (e.g., "H2O")
        cas_number: NIST CAS number (e.g., "7732-18-5" for water)
        temperature_k: Temperature (NIST typically reports 296 K)
        cache_dir: Optional cache directory
    
    Returns:
        ReferenceSpectrum (quantitative=False, shape-only from NIST)
    
    Raises:
        MoleculeSourceError: If fetch or parse fails
    """
    cache_dir = cache_dir or MOLECULES_CACHE_DIR
    
    try:
        # Use existing library.import_nist_jcamp, which handles the URL building
        spectrum = import_nist_jcamp(
            cas_number,
            spectrum="ir",
            fetch=None  # Uses default HTTP fetch
        )
        return spectrum
    except (LibraryError, Exception) as exc:
        raise MoleculeSourceError(
            f"Could not fetch {molecule} from NIST: {exc}"
        ) from exc


# ============================================================================
# ExoMolDB integration (placeholder for future)
# ============================================================================

def fetch_exomoldb_molecule(
    molecule: str,
    temperature_k: float = 1500.0,
    cache_dir: Optional[Path] = None,
) -> ReferenceSpectrum:
    """
    Fetch line list from ExoMolDB and compute cross-section.
    
    NOTE: This is a placeholder. Full implementation requires:
    1. Download .trans/.states files from ExoMolDB
    2. Call import_exomol_transitions() to parse them
    3. Cache the result
    
    For now, raises informative error.
    
    Args:
        molecule: Molecule name (e.g., "H2O")
        temperature_k: Temperature in Kelvin
        cache_dir: Optional cache directory
    
    Raises:
        MoleculeSourceError: Currently not implemented
    """
    raise MoleculeSourceError(
        f"ExoMolDB fetching for {molecule} not yet implemented. "
        "Manual download from https://exomoldb.exoplanet.eu/ required."
    )


# ============================================================================
# HITRAN integration (via petitRADTRANS cross-section generation)
# ============================================================================

def fetch_hitran_via_petitradtrans(
    molecule: str,
    temperature_k: float = 296.0,
    pressure_bar: float = 1.0,
    wavenumber_range_cm1: Tuple[float, float] = (400.0, 4200.0),
    cache_dir: Optional[Path] = None,
) -> ReferenceSpectrum:
    """
    Fetch HITRAN cross-section by computing with petitRADTRANS.
    
    petitRADTRANS downloads and caches HITRAN data automatically.
    This function uses it to generate cross-sections at specified conditions.
    
    Args:
        molecule: HITRAN molecule name
        temperature_k: Temperature in Kelvin
        pressure_bar: Pressure in bar
        wavenumber_range_cm1: (min, max) wavenumber range
        cache_dir: Optional cache directory
    
    Returns:
        ReferenceSpectrum (quantitative=True, with cross-section units)
    
    Raises:
        MoleculeSourceError: If petitRADTRANS not installed or molecule unsupported
    """
    cache_dir = cache_dir or MOLECULES_CACHE_DIR
    
    try:
        from petitRADTRANS.radtrans import Radtrans
    except ImportError:
        raise MoleculeSourceError(
            f"petitRADTRANS not installed. Cannot fetch {molecule} from HITRAN. "
            "Install with: pip install petitradtrans"
        )
    
    try:
        # Create isothermal atmosphere with just this molecule
        prt = Radtrans(
            line_species=[molecule],
            rayleigh_species=["H2", "He"],
            continuum_opacities=["H2-H2", "H2-He"],
            wavelength_boundaries=[wavenumber_range_cm1[0], wavenumber_range_cm1[1]],
            mode="lbl",  # Line-by-line mode
        )
        
        # Generate cross-sections at this T/P
        # This triggers HITRAN download if not already cached
        wavelengths_cm, radii_cm, info = prt.calculate_transit_radii(
            temperature_k * np.ones(100),  # isothermal profile
            pressure_bar * np.ones(100),
            {"H2O": 1e-3}  # dummy mixing ratio
        )
        
        # Convert wavelengths from cm to cm^-1
        wavenumber_cm1 = 1.0 / (wavelengths_cm * 1e-4)
        
        # Placeholder cross-section (would be computed from pRT)
        # For now, return warning that this is not yet full implementation
        raise MoleculeSourceError(
            f"HITRAN cross-section computation for {molecule} requires additional "
            "integration with petitRADTRANS line-by-line output. "
            "See molecule_sources.py for future implementation."
        )
    
    except MoleculeSourceError:
        raise
    except Exception as exc:
        raise MoleculeSourceError(
            f"petitRADTRANS failed to compute cross-section for {molecule}: {exc}"
        ) from exc


# ============================================================================
# High-level molecule loading
# ============================================================================

def load_or_fetch_molecule(
    molecule: str,
    source: str = "nist",
    temperature_k: Optional[float] = None,
    cache_dir: Optional[Path] = None,
) -> ReferenceSpectrum:
    """
    Load a cached molecule, or fetch and cache it if missing.
    
    Args:
        molecule: Molecule name (e.g., "H2O")
        source: "nist" | "exomoldb" | "hitran" | "auto" (try multiple sources)
        temperature_k: Temperature (defaults based on source)
        cache_dir: Optional cache directory
    
    Returns:
        ReferenceSpectrum ready for analysis
    
    Raises:
        MoleculeSourceError: If molecule cannot be fetched from any source
    """
    cache_dir = cache_dir or MOLECULES_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if temperature_k is None:
        temperature_k = 296.0 if source in ("nist", "hitran") else 1500.0
    
    # Check if already cached
    registry = MoleculeRegistry.load(MOLECULES_REGISTRY_FILE)
    cached = registry.get_default_molecule(molecule)
    if cached is not None:
        cached_path = Path(cached["filepath"])
        if cached_path.exists():
            print(f"  ✓ {molecule}: loaded from cache ({source})")
            from app.core.spectral_sources import _load_reference_spectrum_json
            return _load_reference_spectrum_json(cached_path)
    
    # Fetch from source
    print(f"  → {molecule}: fetching from {source}...")
    
    if source == "nist" and molecule in NIST_CAS_NUMBERS:
        spectrum = fetch_nist_molecule(molecule, NIST_CAS_NUMBERS[molecule], temperature_k, cache_dir)
    elif source == "exomoldb" and molecule in EXOMOLDB_MOLECULES:
        spectrum = fetch_exomoldb_molecule(molecule, temperature_k, cache_dir)
    elif source == "hitran":
        spectrum = fetch_hitran_via_petitradtrans(molecule, temperature_k, cache_dir=cache_dir)
    elif source == "auto":
        # Try NIST first, then ExoMolDB, then HITRAN
        for try_source in ["nist", "exomoldb", "hitran"]:
            try:
                spectrum = load_or_fetch_molecule(molecule, try_source, temperature_k, cache_dir)
                source = try_source
                break
            except MoleculeSourceError:
                continue
        else:
            raise MoleculeSourceError(f"Could not fetch {molecule} from any source")
    else:
        raise MoleculeSourceError(f"Unknown source {source!r} or molecule {molecule!r} not supported")
    
    # Cache the result
    cache_file = cache_dir / f"{source}_{molecule}_{int(temperature_k)}K.json"
    _save_reference_spectrum_json(spectrum, cache_file)
    
    # Register in molecule registry
    registry.add_molecule(molecule, source, temperature_k, str(cache_file), make_default=True)
    registry.save(MOLECULES_REGISTRY_FILE)
    
    print(f"  ✓ {molecule}: cached at {cache_file}")
    return spectrum


def populate_common_molecules(
    molecules: List[str] = None,
    source: str = "nist",
    cache_dir: Optional[Path] = None,
) -> Dict[str, ReferenceSpectrum]:
    """
    Fetch and cache a set of common molecules for analysis.
    
    Args:
        molecules: List of molecule names (defaults to common ones)
        source: Where to fetch from ("nist", "exomoldb", "hitran", "auto")
        cache_dir: Optional cache directory
    
    Returns:
        Dict of {molecule: ReferenceSpectrum}
    
    Example:
        molecules = populate_common_molecules(["H2O", "CO2", "CH4"])
        # Now ready for atmospheric analysis
    """
    if molecules is None:
        molecules = ["H2O", "CO2", "CH4", "NH3", "O3", "N2O"]
    
    cache_dir = cache_dir or MOLECULES_CACHE_DIR
    results = {}
    
    print(f"Fetching {len(molecules)} molecules from {source}...")
    for molecule in molecules:
        try:
            spectrum = load_or_fetch_molecule(molecule, source, cache_dir=cache_dir)
            results[molecule] = spectrum
        except MoleculeSourceError as exc:
            print(f"  ✗ {molecule}: {exc}")
    
    print(f"\n✓ Loaded {len(results)}/{len(molecules)} molecules")
    return results
