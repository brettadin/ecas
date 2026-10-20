# ECAS Transmission Retrieval Handoff v14

Updated 2026-10-12.

## Work completed this session

Added an independent instrument-response path to the synthetic validation
harness and real-pRT runner:

- optional positive wavelength-dependent throughput weights;
- optional Gaussian line-spread blur specified in channel-width units;
- explicit response and blur condition metadata;
- fixed scan model remains unaware of the synthetic response;
- invalid line-spread widths are rejected.

Smoke result with injected H₂O `1e-3`, one seed, 2.5 ppm noise, and 40 ppm
offset:

| Synthetic condition | Coverage | Bounded | Absolute log-VMR bias |
| --- | ---: | ---: | ---: |
| Matched response, no blur | 100% | 100% | 0.031 dex |
| Gaussian blur, 2 channel widths | 0% | 100% | 0.219 dex |

The smooth throughput-slope smoke condition had no measurable retrieval change,
which is expected for normalized transit depths when throughput is smooth. The
line-spread blur does redistribute spectral structure and produced a clear
diagnostic mismatch.

## Verification

- `tests/test_atmosphere.py`: **22 passed**.
- Runner compilation and the real-pRT response smoke run passed.
- The existing pytest cache collision warning remains environmental.

## Guardrails

This is one narrow wavelength range, one VMR, one noise level, one offset, and
one blur width. It is not instrument calibration. Real-file validation remains
open because no real x1dints or downloaded reference files are in the workspace.
Correlated/systematic noise remains a separate axis, and emission remains
deferred.

## Next work

1. Expand the line-spread experiment across blur widths and VMRs.
2. Add correlated-noise validation separately from instrument response.
3. Obtain and validate one real JWST x1dints file and one real JCAMP reference.
4. Reassess profile-vs-posterior scope only after the separate axes are tested.
