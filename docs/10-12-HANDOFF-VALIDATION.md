# ECAS Transmission Validation Handoff

Updated 2026-10-12.

## Work completed this session

The first retrieval scope remains the agreed conditional problem: one gas VMR
on a bounded positive grid plus an analytically fitted constant depth offset.
All other model inputs are fixed, measured channel errors are treated as
independent Gaussian uncertainties, and no posterior or calibrated uncertainty
claim is made.

Added `app/core/retrieval_validation.py`:

- `make_synthetic_depth_spectrum(...)` interpolates a forward model at measured
  channel centers and adds seeded independent Gaussian depth noise.
- `run_profile_validation(...)` runs every combination of injected VMR and
  noise seed using explicit forward-model and scan callbacks.
- `summarize_profile_scan(...)` reports median absolute log10-VMR bias, interval
  coverage, bounded-interval fraction, and scan-boundary fraction.
- Unbounded intervals count as not covered, conservatively, when deciding
  whether the interval method is usable.

The harness deliberately does not emulate instrument throughput, line-spread
functions, or correlated systematics. Those must be separate validation axes.
The callbacks can bind the real petitRADTRANS model and current scan, so a
real-backend study does not need a second implementation of the experiment.

## Tests

Focused atmosphere tests: **19 passed**. The full suite is **130 passed**
when GUI child pytest processes receive a workspace-local `--basetemp`; the
default Windows pytest temp root is inaccessible in this environment. One
existing pytest cache collision warning remains.

## Next steps

1. Run `run_profile_validation` against the installed petitRADTRANS 3.4.0
   backend with several injected H₂O VMRs, at least 10 independent seeds per
   injection, a fixed known depth offset, and the same channel uncertainties
   used by the existing synthetic recovery.
2. Record the actual bias, coverage, bounded fraction, and edge-hit fraction in
   `docs/10-12-TODO.md`; do not interpret the results until the run completes.
3. Add realistic instrument-response and fixed-parameter misspecification
   experiments separately from the measured-error likelihood.
4. Use the validation results to decide whether the profile scan is acceptable
   as a clearly labeled quick-look tool or whether the agreed scope needs a
   posterior sampler. Keep emitted-flux work deferred.

## Reproduction context

Use `.venv-1` and the real pRT setup documented in
`docs/10-12-HANDOFF.md`. The validation helper is Qt-free and can be exercised
from a small script or test by binding `model_for_vmr` to
`compute_petit_radtrans_transmission` and `scan_observation` to
`scan_single_gas_abundance` with fixed parameters.
