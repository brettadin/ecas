# ECAS Transmission Retrieval Handoff v6

Updated 2026-10-12.

## Alignment with the original plan

This session stayed within the original v1 sequence: validate the bounded
one-gas profile before adding a broader optimizer or sampler. The expanded
real-pRT study is recorded in v5; this session addressed the v1 requirement to
test boundary cases and provide warnings for unbounded results.

## Boundary reporting changes

`AbundanceScanResult` now includes:

- `boundary_status`: `interior`, `lower_scan_edge`, or `upper_scan_edge`;
- `conditional_limit_kind`: `upper`, `lower`, or `None`;
- `conditional_limit_vmr`: the one-sided Δχ²=1 crossing when it exists.

The result still retains the two crossing fields for interior profiles. When a
best fit is on the lower scan edge, the UI reports a conditional upper limit
only if an upper Δχ²=1 crossing exists. The upper-edge case is symmetric. If
there is no crossing, the UI says the bound is not established within the scan.
It no longer presents edge text as an ordinary two-sided interval.

## Verification

- Focused atmosphere/scan checks: **22 passed**.
- Full suite after this change: **132 passed**; rerun the full
  suite after handoff if additional changes are made.
- Existing pytest cache collision warning remains unrelated to test outcomes.

## Next work from v1/v5

1. Add a focused regression with a monotonic synthetic profile that produces a
   finite conditional one-sided upper limit and another for a lower limit.
2. Extend real-pRT validation over multiple noise levels and fixed-parameter
   misspecification. Keep instrument response and correlated noise separate.
3. Decide whether the conditional profile is acceptable as a quick-look tool or
   whether the agreed scope needs a posterior sampler. Do not promote it to a
   calibrated abundance uncertainty yet.
4. Only then integrate formal retrieval persistence/export details. Emitted flux
   remains deferred.
