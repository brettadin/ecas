# ECAS Transmission Retrieval Handoff v13

Updated 2026-10-12.

## Work completed this session

Reviewed the workspace for real x1dints, MAST downloads, and NIST JCAMP
fixtures. No real observation or downloaded reference files are present, so
the live/real-file validation milestone was not falsely marked complete.

The available offline validation passed **13 tests** covering:

- JWST x1dints with one EXTRACT1D extension per integration;
- JWST vector-column x1dints layout;
- BJD_TDB and MJD_UTC timing metadata;
- missing INT_TIMES behavior and malformed/empty FITS errors;
- matrix CSV wavelength-unit handling;
- transmission extraction and correlated-noise diagnostics;
- JCAMP and emission-table loading paths.

## Current status

The fixed-parameter mismatch runner now covers temperature, gravity, reference
pressure, planet radius, and stellar radius, with explicit synthetic-versus-scan
condition labels. The small tests consistently show that matched controls can
cover while mismatched fixed inputs produce biased, non-covering intervals.
These remain conditional diagnostics, not calibrated uncertainty results.

## Next work

1. Obtain or safely download one real JWST x1dints file and one real NIST/JCAMP
   reference, then validate them through the same loaders and provenance paths.
2. Add a separate instrument-response experiment without mixing it into fixed
   parameter mismatch.
3. Add correlated/systematic-noise validation separately from measured-error
   likelihood tests.
4. Keep emission deferred and do not generalize the profile-scan coverage.
