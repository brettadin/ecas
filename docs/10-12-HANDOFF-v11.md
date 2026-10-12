# ECAS Transmission Retrieval Handoff v11

Updated 2026-10-12.

## Work completed this session

Ran separate real-pRT fixed-parameter mismatch grids using the new runner
options. Each condition used three noise seeds, injected H₂O VMRs `1e-4` and
`1e-3`, 2.5 ppm channel noise, and a 40 ppm depth offset.

| Synthetic condition | Scan condition | Coverage | Bounded | Boundary-limited |
| --- | --- | ---: | ---: | ---: |
| 1000 K, 1000 cgs | 1200 K, 1000 cgs | 0% | 100% | 0% |
| 1200 K, 1000 cgs (matched control) | 1200 K, 1000 cgs | 100% | 100% | 0% |
| 1400 K, 1000 cgs | 1200 K, 1000 cgs | 0% | 100% | 0% |
| 1200 K, 500 cgs | 1200 K, 1000 cgs | 0% | 100% | 0% |
| 1200 K, 2000 cgs | 1200 K, 1000 cgs | 0% | 50% | 50% |

The matched control is useful as a sanity check for this small run. The
mismatch results show that bounded conditional intervals can still fail to
cover the injected VMR, and that gravity mismatch can additionally drive the
scan to an edge.

## Guardrails

These are small, one-range, one-noise-level experiments. They do not establish
failure probabilities for real observations. Instrument response, correlated
noise, reference pressure/radius, stellar radius, and emission remain separate
work items. The profile scan remains a conditional diagnostic, not a calibrated
posterior abundance estimate.

## Next work

1. Add reference pressure/radius and stellar-radius mismatch conditions, with
   the same matched controls and explicit condition labels.
2. Begin real-file validation for x1dints and safe archive/reference fixtures.
3. Add instrument-response and correlated-noise experiments independently of
   fixed-parameter mismatch.
4. Reassess profile-vs-posterior retrieval scope only after these diagnostics.
