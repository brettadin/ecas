# ECAS Transmission Retrieval Handoff v15

Updated 2026-10-12.

## Work completed this session

Added a separate correlated-channel-noise validation axis. Synthetic depth
observations can now use an exponential covariance matrix with a correlation
length in channel units, while the retrieval scan continues to use the agreed
independent measured-error likelihood.

Smoke result with injected H₂O `1e-3`, two seeds, 2.5 ppm nominal channel
uncertainties, and a 40 ppm offset:

| Synthetic noise | Coverage | Bounded | Absolute log-VMR bias |
| --- | ---: | ---: | ---: |
| Independent control | 100% | 100% | 0.031 dex |
| Exponential correlation length 4 channels | 50% | 100% | 0.047 dex |

The mismatch condition remains explicitly labeled and does not alter the
reported channel uncertainties; it tests the consequence of an unmodeled
covariance structure.

## Verification

- `tests/test_atmosphere.py`: **22 passed**.
- Runner compilation and the real-pRT correlated-noise smoke run passed.
- Invalid non-positive correlation lengths are rejected.
- The existing pytest cache collision warning remains environmental.

## Guardrails

This is a two-seed, one-VMR, one-noise-level diagnostic on one narrow channel
range. It is not a calibrated failure probability. Instrument response and
fixed-parameter mismatch remain separate axes; real-file validation remains
open because no real x1dints or downloaded references are present.

## Next work

1. Expand correlated-noise tests across correlation lengths, VMRs, and seeds.
2. Expand line-spread tests independently.
3. Obtain and validate real JWST x1dints and JCAMP files when available.
4. Reassess profile-vs-posterior scope only after separate validation axes are
   characterized.
