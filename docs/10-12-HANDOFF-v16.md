# ECAS Transmission Retrieval Handoff v16

Updated 2026-10-12.

## Work completed this session

Addressed a missing product integration rather than adding another validation
axis:

- `.ecas` project files now preserve the extracted transmission `DepthSpectrum`
  wavelength, depth, and uncertainty arrays;
- project capture/restore carries that spectrum through the Transit workspace;
- Markdown report export includes the current transmission-retrieval report
  state, including model/profile-scan assumptions and warnings;
- retrieval scan results are intentionally recomputed rather than serialized
  as authoritative posterior results.

## Verification

- Project serialization round-trip tests: **2 passed**, including the existing
  GUI project round-trip.
- Atmosphere regression suite remains at **22 passed**.
- Existing pytest cache collision warning remains environmental.

## Scope and guardrails

This closes the basic persistence/export gap but does not create a posterior
retrieval or make conditional profile results durable scientific conclusions.
Instrument response, correlated noise, and fixed-parameter mismatch remain
explicit diagnostics. Real-file validation remains open because no real
x1dints or downloaded reference files are present.

## Next work

1. Improve the exported retrieval section with structured assumptions and
   machine-readable depth-spectrum export where useful.
2. Continue broader UI/product work: onboarding/empty states and status/log
   visibility remain on the main roadmap.
3. Obtain real x1dints/JCAMP inputs when available.
4. Keep emission deferred until the transmission validation path is mature.
