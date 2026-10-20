# ECAS Transmission Retrieval Handoff v12

Updated 2026-10-12.

## Work completed this session

The real-pRT runner now supports synthetic-only mismatch values for reference
pressure, planet radius, and stellar radius, with explicit scan-vs-synthetic
metadata and condition labels. Small checks used two seeds, injected H₂O
`1e-3`, 2.5 ppm noise, and a 40 ppm offset:

| Synthetic condition | Scan condition | Coverage | Bounded |
| --- | --- | ---: | ---: |
| 0.001 bar reference pressure | 0.01 bar | 0% | 100% |
| 0.01 bar reference pressure (control) | 0.01 bar | 100% | 100% |
| 0.1 bar reference pressure | 0.01 bar | 0% | 100% |
| 0.9 Rjup planet radius | 1 Rjup | 0% | 100% |
| 1.0 Rjup planet radius (control) | 1 Rjup | 100% | 100% |
| 1.1 Rjup planet radius | 1 Rjup | 0% | 100% |
| 0.9 Rsun stellar radius | 1 Rsun | 0% | 100% |
| 1.0 Rsun stellar radius (control) | 1 Rsun | 100% | 100% |
| 1.1 Rsun stellar radius | 1 Rsun | 0% | 100% |

These results reinforce that a bounded profile interval can fail under fixed
parameter mismatch even when no scan-edge warning is emitted.

## Verification and limits

- Runner compilation and CLI help passed.
- The experiments use one narrow wavelength range, one injected VMR, one noise
  level, one offset, and only two seeds. They are diagnostics, not universal
  failure probabilities.
- Instrument response and correlated noise remain separate axes. Emission is
  still deferred.

## Next work

1. Begin real-file validation for x1dints, MAST/TAP, and NIST JCAMP fixtures.
2. Add an independent instrument-response experiment without mixing it with
   fixed-parameter mismatch.
3. Add correlated-noise/systematics validation separately from measured-error
   likelihood tests.
4. Reassess profile-vs-posterior scope only after these validation axes.
