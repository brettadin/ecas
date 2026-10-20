# ECAS Transmission Retrieval Handoff v5

Updated 2026-10-12.

## Expanded real-pRT validation

The reproducible runner is:

```powershell
& '.venv-1\Scripts\python.exe' tools\run_real_profile_validation.py `
  --seeds 10 --vmrs 1e-5 1e-4 1e-3 1e-2 --summary-only
```

It uses the installed petitRADTRANS 3.4.0 H₂O HITEMP R1000 and H₂/He BoRi CIA
files, 21 channels over approximately 2.4–2.5 µm, 2.5 ppm independent noise,
and a 40 ppm constant offset. The scan grid contains 33 log-spaced values from
`1e-5` to `1e-2` plus each injected VMR.

The 40-case result, with 10 seeds per injected VMR, was:

| Injected H₂O VMR | Interval coverage | Bounded intervals | Boundary-limited |
| --- | ---: | ---: | ---: |
| `1e-5` | 0% | 10% | 90% |
| `1e-4` | 90% | 100% | 0% |
| `1e-3` | 90% | 100% | 0% |
| `1e-2` | 0% | 10% | 90% |

The aggregate coverage was 45%, but that aggregate is not scientifically useful
because it combines identifiable and boundary-limited regimes. The result
supports a narrower conclusion: this conditional profile behaved reasonably in
the two tested interior regimes, while the current fixed scan is not an
uncertainty interval for weak or strong edge cases. The existing warnings are
working as intended; bounded-looking results at edge cases must not be treated
as abundance measurements.

## Current interpretation

No posterior sampler or formal uncertainty claim should be added yet. The
validation still covers only one molecule, one narrow wavelength interval, one
noise level, one constant offset, and a matched synthetic model. It does not
test instrument throughput, line-spread response, correlated/systematic noise,
or fixed-parameter misspecification.

## Next work

1. Add explicit grid-resolution tests and a deliberate non-detection/upper-limit
   reporting path for boundary-limited scans.
2. Run additional real-pRT cells at multiple channel uncertainties and offsets,
   then test realistic instrument response separately from measured errors.
3. Add fixed-temperature/radius/gravity misspecification experiments and record
   how quickly conditional intervals fail under model mismatch.
4. Decide whether a posterior sampler is necessary for the agreed scope. Keep
   the current profile scan as a clearly labeled quick-look diagnostic until
   those checks are complete. Continue deferring emitted-flux work.

## Verification

- Focused atmosphere tests: **19 passed**.
- Previous full suite: **130 passed** with a workspace-local pytest temp root;
  one existing pytest cache collision warning remains.
