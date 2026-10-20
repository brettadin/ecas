# ECAS NIST Spectrum URL Handoff v25

Updated 2026-10-12.

## User-reported issue

After the HTTP 403 repair, `water` produced a broad-search ambiguity. After
that was addressed, `methane` reached NIST but returned:

`nist import failed: nist webbook request failed: http error 404: not found`

## Work completed

- NIST name resolution now prefers an exact visible species-name anchor. This
  prevents broad name-search links for isotopologues or compounds containing
  the search term from making ordinary names such as `water` ambiguous.
- NIST JCAMP URLs now include `Units=SI`. NIST's regular WebBook pages may
  work without it, but the JCAMP endpoint can return 404 without this query
  parameter for records such as methane.
- Added regression coverage for exact-name preference and the `Units=SI`
  request parameter.

## Verification

- `python -m py_compile app/core/library.py`: passed.
- `python -m pytest -p no:cacheprovider --basetemp C:\Code\ECAS\ecas\.pytest-tmp tests/test_identify.py`: **17 passed**.

## Next work

1. Retry `water` and `methane`, IR, index `0` in the UI.
2. Add a visible list of available NIST spectrum records so users do not need
   to guess the index; methane has multiple records with different indices.
3. Continue validating high-range UV/Vis references for the user's
   10,000–19,500 cm⁻¹ data and preserve explicit no-overlap reporting.
