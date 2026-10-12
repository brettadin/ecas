# ECAS Transmission Retrieval Handoff — Next Session

Updated 2026-10-12.

## Session result

The agreed initial retrieval scope is now recorded in `docs/10-12-TODO.md`:

- one selected gas volume mixing ratio (VMR) is profiled on an explicit positive,
  bounded grid;
- a constant transit-depth offset is fitted analytically at every VMR value;
- temperature, gravity, reference radius/pressure, stellar radius, other gases,
  and the clear isothermal H₂/He atmosphere are fixed;
- the quick-look likelihood uses independent Gaussian measured-channel errors;
- instrument response, correlated noise, clouds/hazes, stellar contamination,
  temperature structure, and multi-gas inference are excluded;
- the Δχ² profile is conditional and is not a posterior or calibrated abundance
  uncertainty.

## Code changes

`app/core/atmosphere.py` now documents the bounded profile semantics on
`AbundanceScanResult` and returns `warnings` when:

- the best-fit VMR is at the minimum or maximum scan value;
- the lower Δχ²=1 crossing is not present in the scan grid;
- the upper Δχ²=1 crossing is not present in the scan grid.

`app/ui/transit_panel.py` displays these warnings with the scan report. This
prevents an edge-limited scan from looking like a bounded abundance result.

`tests/test_atmosphere.py` adds a flat-spectrum boundary case that verifies the
best-fit edge and both unbounded interval warnings.

## Verification

Focused command, run with a workspace-local pytest temp directory because the
default Windows temp root had an access-denied directory:

```powershell
& '.venv-1\Scripts\python.exe' -m pytest -q --basetemp .pytest-tmp tests\test_atmosphere.py tests\test_timeseries.py
```

Result: **26 passed**. The pytest cache emitted one existing Windows file
collision warning; it did not affect test outcomes.

The full suite was also verified. Because GUI tests launch isolated child
pytest processes, `PYTEST_ADDOPTS` propagated a workspace-local child
`--basetemp`:

```powershell
$env:PYTEST_ADDOPTS='--basetemp=C:\Code\ECAS\ecas\.pytest-child-tmp'
& '.venv-1\Scripts\python.exe' -m pytest -q --basetemp .pytest-tmp
```

Result: **128 passed**. One existing pytest cache collision warning remained.

## Next work

1. Add a repeatable validation experiment over multiple independent noise seeds
   and injected VMRs using the real pRT backend where available. Measure bias and
   interval coverage rather than only best-grid recovery.
2. Add explicit tests for undetectable gas signals, scan-edge injections, and
   fixed-parameter misspecification. Keep instrument-response and correlated-noise
   experiments separate from measured uncertainty.
3. Use those results to decide whether the conditional profile can remain a
   quick-look tool or whether a posterior sampler is required for the agreed
   scope. Do not call the current Δχ² interval a validated uncertainty yet.
4. Only after validation, consider persistence/export/UI integration of a formal
   retrieval result. Continue deferring emitted-flux work.

## Reproducibility context

The real-backend environment remains the one documented in
`docs/10-12-HANDOFF.md`: ECAS `.venv-1`, petitRADTRANS 3.4.0, MSYS2 UCRT64
gfortran runtime, Microsoft MPI, and the recorded H₂O/H₂-H₂/H₂-He opacity data.
