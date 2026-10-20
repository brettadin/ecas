# ECAS NIST HTTP 403 Handoff v24

Updated 2026-10-12.

## User-reported issue

After the name-resolution fix, entering `water` reached NIST but failed with:

`nist import failed: nist webbook request failed: http error 403: forbidden`

## Work completed

- Updated `app/core/library.py::_fetch_nist_text` to send an ECAS User-Agent
  instead of urllib's default Python User-Agent.
- Added an `Accept` header covering both the HTML returned by NIST name search
  and the plain-text JCAMP response returned by the spectrum endpoint.
- Kept the injectable fetch path unchanged for deterministic offline tests.
- Added a regression test proving the network request carries the ECAS
  User-Agent and timeout.

## Verification

- `python -m py_compile app/core/library.py`: passed.
- `python -m pytest -p no:cacheprovider --basetemp C:\Code\ECAS\ecas\.pytest-tmp tests/test_identify.py`: **17 passed**.

## Next work

1. Retry NIST Browse with `water`, IR, index `0`; if NIST still blocks the
   environment, surface a specific connectivity/access message and offer a
   browser URL fallback.
2. Add a visible list of available NIST spectrum records so users do not need
   to guess the index.
3. Continue the broader roadmap: validate high-range UV/Vis references for the
   user's 10,000–19,500 cm⁻¹ data and preserve explicit no-overlap reporting.
