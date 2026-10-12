# ECAS Transmission Retrieval Handoff v8

Updated 2026-10-12.

## Work completed this session

The real-pRT validation runner now accepts multiple conditions:

```powershell
& '.venv-1\Scripts\python.exe' tools\run_real_profile_validation.py `
  --seeds 10 --vmrs 1e-4 1e-3 `
  --uncertainties-ppm 2.5 5 --offsets-ppm 40 --summary-only
```

It caches the expensive forward models across conditions and emits aggregate
plus per-injected-VMR summaries.

The 40-case result was:

| Noise σ | Injected VMR | Coverage | Bounded | Boundary-limited |
| ---: | ---: | ---: | ---: | ---: |
| 2.5 ppm | `1e-4` | 90% | 100% | 0% |
| 2.5 ppm | `1e-3` | 90% | 100% | 0% |
| 5 ppm | `1e-4` | 80% | 100% | 0% |
| 5 ppm | `1e-3` | 90% | 100% | 0% |

The aggregate coverage is 90% at 2.5 ppm and 85% at 5 ppm. These results are
conditional, use one narrow wavelength range and one fixed 40 ppm offset, and
do not establish universal calibration.

## Alignment with v1

This is still validation of the agreed one-gas VMR plus constant offset profile.
No sampler, multi-gas retrieval, emission workflow, instrument response, or
correlated-noise model was added. The next v1-required experiment is fixed
parameter misspecification, kept separate from the measured-error likelihood.

## Next work

1. Add real-pRT misspecification experiments where synthetic data use one fixed
   temperature, gravity, reference pressure/radius, or stellar radius and the
   scan intentionally holds a different value. Record bias and coverage by
   mismatch size.
2. Keep instrument response and correlated/systematic noise as separate axes.
3. Add more offsets/channel ranges only after the misspecification harness is
   recording per-condition diagnostics.
4. Decide whether the conditional profile remains a quick-look diagnostic or
   whether the agreed scope requires a posterior sampler. Do not treat the
   current 85–90% results as general calibrated uncertainty.

## Verification

- Full suite before this runner-only change: **132 passed**.
- Existing pytest cache collision warning remains unrelated to test outcomes.
