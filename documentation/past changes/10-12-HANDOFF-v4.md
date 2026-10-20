# ECAS Transmission Retrieval Handoff v4

Updated 2026-10-12.

## Current result

The first real petitRADTRANS multi-seed validation study is complete for the
current narrow scope:

- backend: petitRADTRANS 3.4.0;
- opacities: H₂O HITEMP R1000, H₂-H₂ BoRi CIA, H₂-He BoRi CIA;
- wavelength range: approximately 2.4–2.5 µm;
- 21 measured channels;
- injected H₂O VMRs: `1e-4` and `1e-3`;
- 10 independent noise seeds per injection;
- independent Gaussian depth noise: 2.5 ppm;
- injected constant depth offset: 40 ppm;
- scan grid: 33 log-spaced values from `1e-5` to `1e-2`, plus each injected VMR.

Results across all 20 cases:

- median absolute log10-VMR bias: **0 dex**;
- conditional Δχ²=1 interval coverage: **90%**;
- bounded interval fraction: **100%**;
- best-fit scan-boundary fraction: **0%**.

The fitted offsets were generally close to 40 ppm, with some noise cases
showing larger deviations. The one non-covering case at each injection is a
useful warning, not evidence of calibration. This result is scope-specific and
does not validate interpretation of real observations or establish universal
coverage.

## Implementation

`tools/run_real_profile_validation.py` now binds the generic
`app/core/retrieval_validation.py` harness to the installed pRT adapter. It
uses only the locally installed H₂O/H₂-He opacity files and sets
`HDF5_USE_FILE_LOCKING=FALSE`; it does not trigger pRT's online Keeper fallback.
The scan grid explicitly includes injected values so coarse-grid omission does
not masquerade as retrieval bias.

The synthetic generator now integrates each model over the same inferred
top-hat channels used by the retrieval. A direct adapter check also confirmed
that separate pRT calls with identical parameters produce identical channel
models.

## Verification

- `tests/test_atmosphere.py`: **19 passed**.
- Previous full-suite result: **130 passed** with a workspace-local pytest
  temp root for GUI child processes. One existing pytest cache collision warning
  remains.
- Real validation command:

```powershell
& '.venv-1\Scripts\python.exe' tools\run_real_profile_validation.py --seeds 10 --vmrs 1e-4 1e-3
```

## Next work

1. Extend the real-pRT study over more VMRs, especially weak/undetectable and
   stronger signals, and several uncertainty levels. Keep at least 10 seeds per
   cell and record per-cell coverage rather than only the aggregate.
2. Add scan-grid resolution checks and explicit fixed-parameter misspecification
   experiments. Keep instrument response and correlated/systematic noise in
   separate experiments.
3. Decide whether the conditional profile remains a quick-look diagnostic or
   whether the agreed scope requires a posterior sampler. Do not advertise the
   90% result as calibrated uncertainty.
4. Only after those checks, integrate a formal retrieval result into persistence,
   exports, and the UI. Continue deferring emitted-flux work.
