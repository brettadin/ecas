# ECAS PHASE 0 AUDIT REPORT PART 2

## Summary of Findings

__Architecture:__ PyQt6 + pyqtgraph; core physics in `src/app/core`; UI panels in `src/app/ui`. Entry point: `src/main.py`. Data registry: `data/gases.json`. Parser: `app/core/parser.py` (ORCA-based but we are moving away from ORCA).

__Verified issues:__

1. __NIST fetches fail__ → HTTP 403; requires identifier mapping / retry logic.
2. __MAST downloads slow__ despite caching—re-downloads occur when cache misses.
3. __Some JWST NIRSpec files fail to parse:__ mismatched wavelength counts, non-positive out-of-transit flux values.
4. __Playback lag on large data__ → Qt event loop / pyqtgraph rendering bottleneck.
5. __Mock data in registry__ (`mock://` entries) mixed with real ORCA paths—must separate or remove.
6. __petitRADTRANS integration__ had missing-import bug; status unclear whether fully fixed (needs re-check).
7. __HITRAN/ExoMol import__ not yet verified for actual data ingestion.

__Pivot decision confirmed:__ ECAS will stop using ORCA output files and instead ingest real spectra from online databases (NIST, HAPI, ExoMol, etc.). This requires:

- Removing all `mock://` entries and any hard-coded mock spectra.
- Updating parsers to fetch/parse remote spectral databases.
- Ensuring data provenance is clearly logged for every loaded spectrum.

__Scope & limitations:__ Mock/synthetic data are now disallowed unless explicitly in a labeled simulation mode (not the default). All spectra must come from verified external sources or local files that were actually produced.

__Open questions addressed:__

- Mock data will be removed immediately per your instruction.
- NIST issues will be addressed as we transition to real online data.
- Blueprint moved; updated as ECAS Master Construction Prompt—docs will be realigned to current structure.
- README unchanged but project paths have moved; documentation will reflect new locations.

---

## Next Steps (Your Approved Order)

1. __Remove all mock data__ from `data/gases.json` and related modules.
2. __Adjust parsers/sources__ to fetch real spectra from online databases (NIST, HAPI, ExoMol, etc.).
3. __Update documentation__ to match current file layout and remove references to ORCA/mocked data.
4. __Re-run verification checks__ on the modified codebase (no import errors, config loads correctly, no mock entries).

Proceeding now with step 1 (mock data removal) and documentation cleanup in parallel.
