# ECAS Reference-Coverage Finding Handoff v22

Updated 2026-10-12.

## Confirmed user finding

The user's uploaded spectrum is approximately 10,000–19,500 cm⁻¹, and the
currently available built-in/imported reference data do not overlap that range.
This is an evidence gap, not a molecular non-detection. ECAS must not report
“nothing detected” from this file until a reference covering the same range and
compatible measurement conditions is available.

The built-in simulator/ORCA-style bands remain primarily IR coverage around
400–4200 cm⁻¹. The NIST UV/Vis importer now exists, but it has not yet been
validated against the user's actual file or established as phase/condition
compatible with it.

## Product options that remain actionable now

1. Complete real-file validation: inspect the uploaded file's x/y units,
   continuum, phase, and uncertainty metadata, then import one matching UV/Vis
   reference and compare coverage/resolution.
2. Add a visible coverage filter and “no compatible references” state to
   Identify, so irrelevant references are automatically separated from usable
   ones.
3. Improve transmission identification with resolution/LSF matching, optional
   wavelength-shift grids, robust continuum models, and ranked-result plots.
4. Finish structured retrieval-result exports and onboarding/empty states.
5. Validate the existing real-pRT retrieval across additional noise,
   instrument-response, correlated-noise, and fixed-parameter mismatch cases.
6. Build a separate emission-analysis path with emission templates or a
   validated atmosphere model; do not reuse absorption references as emission
   abundances.
7. Continue archive integration and real MAST/JWST/x1dints validation.

## Documentation updated

- `docs/ROADMAP.md` now records the confirmed no-overlap finding.
- `docs/CHANGELOG.md` records it as an evidence gap.
- `docs/10-12-TODO.md` points here as the latest handoff.

