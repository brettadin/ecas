# ECAS NIST Water Download Handoff v27

Updated 2026-10-12.

## Current status

Methane and ozone now import successfully. Water still failed in the user's
environment after the IR selector correction.

## Work completed

- Confirmed NIST's actual water “Download spectrum in JCAMP-DX” link uses:
  `JCAMP=C7732185&Index=0&Type=IR`
- Aligned ECAS with that exact query shape by removing the extra `Units=SI`
  parameter.
- Kept `Type=IR` for JCAMP downloads; `IR-SPEC` is the HTML page selector.
- Updated the request regression test.

## Verification

- `python -m py_compile app/core/library.py`: pending after this final URL edit.
- Focused Identify tests should be rerun before the next session.

## Next work

1. Retry water with IR/index `0`.
2. If water still fails, capture the exact new error and raw response status;
   distinguish a missing water record from local/network blocking.
3. Add the visible NIST record list so index selection is no longer guesswork.
