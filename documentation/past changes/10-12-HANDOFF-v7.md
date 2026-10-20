# ECAS Transmission Retrieval Handoff v7

Updated 2026-10-12.

## Work completed

The original v1/v5 sequence remains the governing plan. This session completed
the next boundary-case requirement without broadening the retrieval scope.

`AbundanceScanResult` now reports structured edge semantics:

- `boundary_status`: `interior`, `lower_scan_edge`, or `upper_scan_edge`;
- `conditional_limit_kind`: `upper`, `lower`, or `None`;
- `conditional_limit_vmr`: a one-sided Δχ²=1 crossing when it exists.

The Transit UI labels these cases as conditional upper/lower limits. If the
profile has no crossing, it explicitly says the bound is not established in the
scan. It does not present an edge-limited result as a two-sided interval.

Regression coverage now includes both a finite upper limit at the lower scan
edge and a finite lower limit at the upper scan edge, in addition to the
unbounded flat-profile case.

## Verification

- Boundary/one-sided checks: **3 passed**.
- Full suite: **132 passed**.
- One existing pytest cache collision warning remains; it does not affect test
  outcomes. GUI child tests require the workspace-local pytest temp-root setup
  documented in v5/v6.

## Alignment check

This work directly addresses v1's required boundary/failure diagnostics and
v5's finding that `1e-5` and `1e-2` regimes are usually scan-boundary limited.
No multi-gas retrieval, posterior sampler, emission workflow, instrument
response, or correlated-noise model was introduced.

## Next work

1. Extend the real-pRT study across multiple channel uncertainty levels and
   offsets, retaining at least 10 seeds per cell and per-injection summaries.
2. Add fixed-temperature, gravity, reference-radius, and pressure
   misspecification experiments. Keep these separate from the measured-error
   likelihood and report their bias/coverage impact.
3. Decide whether the conditional profile remains a quick-look diagnostic or
   whether the agreed one-gas scope requires a posterior sampler. Do not call
   the current intervals calibrated abundance uncertainties.
4. After those checks, consider persistence/export integration. Emitted flux
   remains deferred as required by v1.
