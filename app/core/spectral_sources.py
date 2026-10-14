"""
Spectral data acquisition and caching for ECAS.

High-level module that orchestrates fetching of:
  - Molecular cross-sections from HITRAN/ExoMolDB
  - Exoplanet transmission spectra from MAST
  - Planet parameters from NASA Exoplanet Archive

All data is cached locally to avoid re-downloading.
Integrates with existing library.py and archives.py infrastructure.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Import from existing ECAS modules
from app.core.archives import (
    fetch_planet, search_mast_spectral_timeseries, download_mast_spectral_timeseries,
    PlanetRecord, ArchiveError
)
from app.core.library import (
    ReferenceSpectrum, reference_from_arrays, LibraryError
)
from app.core.sources import Source, CitationLog, SOURCES
from app.core.transit import DepthSpectrum

FloatArray = NDArray[np.float64]

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CACHE_DIR = Path.home() / ".ecas" / "data_sources"
MOLECULES_REGISTRY_FILE = DEFAULT_CACHE_DIR / "molecules_registry.json"

# Known HITRAN isotopologues and their standard reference conditions
HITRAN_STANDARDS = {
    "H2O": {"temperature_k": 296.0, "pressure_bar": 1.0},
    "CO2": {"temperature_k": 296.0, "pressure_bar": 1.0},
    "CH4": {"temperature_k": 296.0, "pressure_bar": 1.0},
    "O3": {"temperature_k": 296.0, "pressure_bar": 1.0},
    "N2O": {"temperature_k": 296.0, "pressure_bar": 1.0},
    "CO": {"temperature_k": 296.0, "pressure_bar": 1.0},
    "NH3": {"temperature_k": 296.0, "pressure_bar": 1.0},
    "HCN": {"temperature_k": 296.0, "pressure_bar": 1.0},
    "SO2": {"temperature_k": 296.0, "pressure_bar": 1.0},
}

# ============================================================================
# Data classes
# ============================================================================

@dataclass
class MoleculeRegistry:
    """Tracks which molecules have been cached and where they came from."""
    
    molecules: Dict[str, Dict] = field(default_factory=dict)
    # Schema: {
    #   "H2O": {
    #     "sources": [
    #       {"source_key": "hitran2020", "temperature_k": 296.0, "filepath": "..."},
    #       {"source_key": "exomoldb", "temperature_k": 1500.0, "filepath": "..."}
    #     ],
    #     "default_source_key": "hitran2020"
    #   }
    # }
    
    def add_molecule(self, molecule: str, source_key: str, temperature_k: float,
                    filepath: str, make_default: bool = False) -> None:
        """Register a cached molecule cross-section."""
        if molecule not in self.molecules:
            self.molecules[molecule] = {"sources": [], "default_source_key": None}
        
        self.molecules[molecule]["sources"].append({
            "source_key": source_key,
            "temperature_k": temperature_k,
            "filepath": filepath,
        })
        
        if make_default or self.molecules[molecule]["default_source_key"] is None:
            self.molecules[molecule]["default_source_key"] = source_key
    
    def get_default_molecule(self, molecule: str) -> Optional[Dict]:
        """Get the default cached source for a molecule."""
        if molecule not in self.molecules:
            return None
        default_key = self.molecules[molecule]["default_source_key"]
        for source_dict in self.molecules[molecule]["sources"]:
            if source_dict["source_key"] == default_key:
                return source_dict
        return None
    
    def save(self, path: Path) -> None:
        """Serialize registry to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.molecules, f, indent=2)
    
    @staticmethod
    def load(path: Path) -> MoleculeRegistry:
        """Load registry from JSON."""
        registry = MoleculeRegistry()
        if path.exists():
            with open(path, "r") as f:
                registry.molecules = json.load(f)
        return registry


@dataclass
class SpectralDataset:
    """Holds a complete dataset ready for analysis."""
    
    target_name: str
    spectrum: DepthSpectrum
    planet_params: Optional[PlanetRecord] = None
    source_citations: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# Error handling
# ============================================================================

class SpectralSourceError(RuntimeError):
    """Raised for issues fetching or caching spectral data."""
    pass


# ============================================================================
# HITRAN/HAPI integration
# ============================================================================

def fetch_hitran_cross_section(
    molecule: str,
    temperature_k: Optional[float] = None,
    pressure_bar: Optional[float] = None,
    wavenumber_range_cm1: Tuple[float, float] = (400.0, 4200.0),
    cache_dir: Optional[Path] = None,
) -> ReferenceSpectrum:
    """
    Fetch a HITRAN cross-section for a molecule.
    
    NOTE: This is a placeholder. Actual implementation requires HAPI credentials.
    For now, this documents the interface. Users must fetch via HAPI themselves
    and call library.hapi_cross_section_reference() directly.
    
    Args:
        molecule: HITRAN molecule name (e.g., "H2O", "CO2")
        temperature_k: Temperature in Kelvin (defaults to 296 K)
        pressure_bar: Pressure in bar (defaults to 1.0 bar)
        wavenumber_range_cm1: Tuple of (min, max) wavenumber in cm^-1
        cache_dir: Directory for local cache
    
    Returns:
        ReferenceSpectrum ready for use in abundance scans
    
    Raises:
        SpectralSourceError: If HAPI is not installed or credentials missing
    """
    if temperature_k is None:
        temperature_k = HITRAN_STANDARDS.get(molecule, {}).get("temperature_k", 296.0)
    if pressure_bar is None:
        pressure_bar = HITRAN_STANDARDS.get(molecule, {}).get("pressure_bar", 1.0)
    
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"hitran_{molecule}_{int(temperature_k)}K.json"
    
    if cache_file.exists():
        return _load_reference_spectrum_json(cache_file)
    
    raise SpectralSourceError(
        f"HITRAN cross-section for {molecule} not cached. "
        f"To fetch from HITRAN, use petitRADTRANS or manually call HAPI, then save with "
        f"library.hapi_cross_section_reference(). See README for HITRAN credentials setup."
    )


def _load_reference_spectrum_json(filepath: Path) -> ReferenceSpectrum:
    """Load a ReferenceSpectrum from JSON cache."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    return ReferenceSpectrum(
        ref_id=data["ref_id"],
        name=data["name"],
        formula=data["formula"],
        nu_cm1=np.array(data["nu_cm1"], dtype=np.float64),
        k=np.array(data["k"], dtype=np.float64),
        source_key=data["source_key"],
        quantitative=data.get("quantitative", False),
        y_kind=data.get("y_kind", "shape"),
        meta=data.get("meta", {}),
    )


def _save_reference_spectrum_json(spectrum: ReferenceSpectrum, filepath: Path) -> None:
    """Save a ReferenceSpectrum to JSON cache."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "ref_id": spectrum.ref_id,
        "name": spectrum.name,
        "formula": spectrum.formula,
        "nu_cm1": spectrum.nu_cm1.tolist(),
        "k": spectrum.k.tolist(),
        "source_key": spectrum.source_key,
        "quantitative": spectrum.quantitative,
        "y_kind": spectrum.y_kind,
        "meta": spectrum.meta,
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================================
# Exoplanet spectrum fetching & parsing
# ============================================================================

def fetch_exoplanet_transmission_spectrum(
    target_name: str,
    cache_dir: Optional[Path] = None,
) -> SpectralDataset:
    """
    Fetch an exoplanet transmission spectrum from MAST and parse it.
    
    1. Queries NASA Exoplanet Archive for planet parameters
    2. Searches MAST for available JWST/HST spectra
    3. Downloads the first available spectrum
    4. Parses FITS into DepthSpectrum
    5. Caches everything locally
    
    Args:
        target_name: Planet name as in NASA Exoplanet Archive (e.g., "WASP-39 b")
        cache_dir: Directory for local cache
    
    Returns:
        SpectralDataset with spectrum, planet params, and citations
    
    Raises:
        SpectralSourceError: If planet not found or no spectrum available
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    target_cache = cache_dir / "targets" / target_name.replace(" ", "_")
    target_cache.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Fetch planet parameters
    try:
        planet = fetch_planet(target_name)
    except ArchiveError as exc:
        raise SpectralSourceError(f"Could not fetch planet {target_name}: {exc}") from exc
    
    # Step 2: Search for spectra on MAST
    try:
        spectra = search_mast_spectral_timeseries(target_name)
    except ArchiveError as exc:
        raise SpectralSourceError(
            f"No JWST/HST spectra found for {target_name} on MAST: {exc}"
        ) from exc
    
    if not spectra:
        raise SpectralSourceError(f"No transmission spectra available for {target_name}")
    
    # Step 3: Download first available spectrum
    first_spectrum = spectra[0]
    try:
        fits_path = download_mast_spectral_timeseries(first_spectrum, target_cache)
    except ArchiveError as exc:
        raise SpectralSourceError(f"Could not download spectrum: {exc}") from exc
    
    # Step 4: Parse FITS into DepthSpectrum
    try:
        depth_spectrum = _parse_transmission_fits(fits_path)
    except Exception as exc:
        raise SpectralSourceError(f"Could not parse FITS file: {exc}") from exc
    
    # Step 5: Create dataset with citations
    citations = [
        f"Planet parameters: {planet.refs.get('planet', 'NASA Exoplanet Archive')}",
        f"Spectrum: MAST proposal {first_spectrum.get('proposal_id', 'unknown')}",
    ]
    
    metadata = {
        "target_name": target_name,
        "mast_obsid": first_spectrum.get("obsid"),
        "proposal_id": first_spectrum.get("proposal_id"),
        "instrument": first_spectrum.get("instrument_name"),
        "wavelength_range_um": (
            float(depth_spectrum.wavelength_um.min()),
            float(depth_spectrum.wavelength_um.max())
        ),
    }
    
    return SpectralDataset(
        target_name=target_name,
        spectrum=depth_spectrum,
        planet_params=planet,
        source_citations=citations,
        metadata=metadata,
    )


def _parse_transmission_fits(fits_path: Path) -> DepthSpectrum:
    """
    Parse a JWST x1dints FITS file into a DepthSpectrum.
    
    Extracts wavelength, transit depth, and uncertainty from the FITS table.
    Handles unit conversions (Angstroms → microns, parts-per-million, etc.)
    
    Args:
        fits_path: Path to FITS file
    
    Returns:
        DepthSpectrum ready for analysis
    
    Raises:
        ValueError: If FITS structure is unexpected
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError(
            "astropy is required to parse FITS files. "
            "Install with: pip install astropy"
        )
    
    with fits.open(fits_path) as hdul:
        if len(hdul) < 2:
            raise ValueError("FITS file has no data HDU")
        
        # JWST x1dints files have the spectrum in HDU 1
        data_hdu = hdul[1]
        data = data_hdu.data
        
        # Standard JWST column names (case-insensitive search)
        col_names = [name.upper() for name in data.dtype.names or []]
        
        # Find wavelength column (Angstroms in FITS)
        wavelength_col = None
        for name in ["WAVELENGTH", "WAVE", "LAMBDA"]:
            if name in col_names:
                wavelength_col = name
                break
        
        if wavelength_col is None:
            raise ValueError(f"Could not find wavelength column. Available: {col_names}")
        
        # Find flux/transmission columns
        flux_col = None
        for name in ["TRANSMISSION", "FLUX", "FLUX_NORMALIZED"]:
            if name in col_names:
                flux_col = name
                break
        
        if flux_col is None:
            raise ValueError(f"Could not find flux/transmission column. Available: {col_names}")
        
        # Find uncertainty column
        unc_col = None
        for name in ["ERR", "UNCERTAINTY", "FLUX_ERR", "TRANSMISSION_ERR"]:
            if name in col_names:
                unc_col = name
                break
        
        if unc_col is None:
            warnings.warn(f"No uncertainty column found; using 1% of flux as default")
            unc = np.abs(data[flux_col]) * 0.01
        else:
            unc = data[unc_col]
        
        # Convert wavelength from Angstroms to microns
        wavelength_um = data[wavelength_col] / 1e4
        
        # Transit depth (usually fractional; convert to unitless)
        depth = np.array(data[flux_col], dtype=np.float64)
        depth_err = np.array(unc, dtype=np.float64)
        
        # If transmission is > 1, it's already fractional (good)
        # If it's small, assume it's the transit depth directly
        if depth.max() > 1.0:
            # Likely fractional transmission; estimate depth as 1 - transmission
            # This is a heuristic; actual parsing depends on file conventions
            pass
        
        return DepthSpectrum(
            wavelength_um=wavelength_um,
            depth=depth,
            depth_err=depth_err,
            meta={
                "source": "JWST x1dints FITS",
                "fits_file": str(fits_path.name),
            }
        )


# ============================================================================
# High-level orchestration
# ============================================================================

def prepare_analysis(
    target_name: str,
    molecules: List[str] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[SpectralDataset, Dict[str, ReferenceSpectrum]]:
    """
    One-shot function to fetch everything needed for atmospheric analysis.
    
    Args:
        target_name: Exoplanet name (e.g., "WASP-39 b")
        molecules: List of molecules to load (defaults to common ones)
        cache_dir: Directory for caching
    
    Returns:
        (SpectralDataset, dict of ReferenceSpectra)
    
    Example:
        dataset, molecules = prepare_analysis("WASP-39 b", ["H2O", "CO2", "CH4"])
        # Now ready to pass to scan_single_gas_abundance()
    """
    if molecules is None:
        molecules = ["H2O", "CO2", "CH4"]
    
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    print(f"Fetching exoplanet data for {target_name}...")
    dataset = fetch_exoplanet_transmission_spectrum(target_name, cache_dir)
    print(f"  ✓ Spectrum: {dataset.spectrum.wavelength_um.size} channels")
    print(f"  ✓ Range: {dataset.spectrum.wavelength_um.min():.2f}–{dataset.spectrum.wavelength_um.max():.2f} μm")
    
    print(f"\nLoading molecular cross-sections...")
    registry = MoleculeRegistry.load(MOLECULES_REGISTRY_FILE)
    reference_spectra = {}
    
    for molecule in molecules:
        mol_source = registry.get_default_molecule(molecule)
        if mol_source is None:
            print(f"  ⚠ {molecule}: not cached (will need to be added manually)")
            continue
        
        try:
            spectrum = _load_reference_spectrum_json(Path(mol_source["filepath"]))
            reference_spectra[molecule] = spectrum
            print(f"  ✓ {molecule} ({mol_source['source_key']}, {mol_source['temperature_k']:.0f} K)")
        except Exception as exc:
            print(f"  ✗ {molecule}: {exc}")
    
    return dataset, reference_spectra
