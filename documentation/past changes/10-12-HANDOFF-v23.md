# ECAS NIST Name Lookup Handoff v23

Updated 2026-10-12.

## Issue fixed

The NIST Browse dialog said “name or CAS,” but the implementation previously
accepted only numeric CAS/WebBook identifiers. Entering a name therefore
produced:

`nist import failed: nist webbook lookup needs a numeric CAS/WebBook identifier`

## Implementation

- Added NIST chemical-name resolution through the WebBook name-search endpoint.
- A unique name is resolved to its numeric species ID, then the existing IR or
  UV/Vis JCAMP download is requested.
- Ambiguous names produce an explicit error listing candidate IDs and ask the
  user to use a CAS number rather than silently choosing one.
- Empty/invalid identifiers produce a clear input error.
- Updated the dialog prompt to “Chemical name or CAS/WebBook ID”.
- Updated the index prompt to “Spectrum index (0 = first record)”. The index
  selects among multiple spectrum records for one species; `0` is the normal
  starting choice.

## Verification

- `tests/test_identify.py`: **16 passed**.
- Python compilation passed for the modified library and Identify UI modules.
- Added regression coverage for name resolution, resolved-ID provenance, and
  the subsequent JCAMP request.

## Next work

1. Use the user's actual 10,000–19,500 cm⁻¹ file to choose compatible UV/Vis
   references and verify phase/solvent/resolution metadata.
2. Add a visible reference-coverage filter and a list of available NIST
   records before download, so users do not have to guess the index.
3. Keep the no-overlap evidence-gap behavior and do not convert an absent
   reference into a non-detection claim.

