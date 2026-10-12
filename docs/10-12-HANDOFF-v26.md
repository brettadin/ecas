# ECAS NIST IR Download Handoff v26

Updated 2026-10-12.

## Clarification

The user's errors were different:

- `water`: name resolution reported multiple species because broad NIST search
  results included related compounds; exact-name preference was added.
- `methane`: after resolving the name, the JCAMP request returned HTTP 404.

## Work completed

- Confirmed from NIST's actual “Download spectrum in JCAMP-DX” link that the
  download endpoint uses `Type=IR`, while `Type=IR-SPEC` identifies the HTML
  spectrum page.
- Changed ECAS IR JCAMP requests to `Type=IR`.
- Retained `Units=SI`, the ECAS User-Agent, and exact-name resolution.
- Updated the regression assertion for the corrected endpoint selector.

## Verification

- `python -m py_compile app/core/library.py`: passed.
- `python -m pytest -p no:cacheprovider --basetemp C:\Code\ECAS\ecas\.pytest-tmp tests/test_identify.py`: **17 passed**.

## Next work

1. Retry `water` and `methane`, IR, index `0` in the UI.
2. Add a visible list of available NIST spectrum records so users do not need
   to guess the index.
3. Continue validating high-range UV/Vis references for the user's
   10,000–19,500 cm⁻¹ data and preserve explicit no-overlap reporting.
