# Transmission Retrieval Project Handoff

Updated 2026-10-12.

This note is a detailed handoff for continuing ECAS's transmission-spectrum
atmospheric-analysis work. It records the current implementation and evidence,
why the work is being done, and what should happen next. The maintained roadmap
is [10-12-TODO.md](./10-12-TODO.md); update it as milestones change.

## User goal and project scope

The user wants to turn retrieved or locally loaded exoplanet transit time-series
data into a wavelength-dependent transmission spectrum and use ECAS to compare
that spectrum with atmospheric models and known molecular absorbers. The
long-term goal is in-program atmospheric-composition analysis that is
scientifically responsible, reproducible, and supported by references and
clear uncertainty reporting.

The active work is **transmission retrieval**. The recent emitted-flux thread is
separate and explicitly deferred; do not treat emitted flux as transmission or
resume the emission workflow until transmission validation milestones are
sufficiently complete.

Important existing workflow context:

- Archive time-series data should be reused locally when already downloaded,
  rather than downloaded repeatedly.
- Users can also load local datasets.
- A downloaded JWST spectrum can be extracted and displayed, while some
  time-series-derived plots have differed depending on dataset/source.
- The atmospheric workflow currently extracts transit depths and offers a
  physical-model overlay and an exploratory abundance scan. It is not yet a
  validated multi-parameter atmospheric retrieval.

## Current state at handoff

### Work completed in this session

1. **Real petitRADTRANS runtime setup and forward-model verification**
   - With the user's prior approval, installed native Windows prerequisites:
     MSYS2 UCRT64 Fortran tooling, Microsoft MPI, and petitRADTRANS 3.4.0 in the
     selected ECAS Python environment.
   - Set up opacity data required by a real H₂O forward model: H₂O HITEMP R1000,
     H₂-H₂ BoRi CIA, and H₂-He BoRi CIA.
   - Successfully generated a real transmission model; previously recorded
     validation produced 41 wavelengths over approximately 2.4018–2.4998 µm
     for a 2.4–2.5 µm requested interval.
   - Opacity setup used TLS-verified downloads. Do not disable TLS verification
     to work around the pRT automatic downloader's certificate issue.

2. **Fixed pRT wavelength edge coverage**
   - A real-model synthetic recovery attempt found that pRT's returned
     wavelengths are bin centers. Requesting boundaries equal to the observed
     channel outer edges could therefore leave the returned centers slightly
     inside those edges, causing the channel comparison to reject the model.
   - `app/core/atmosphere.py` now pads pRT's requested wavelength bounds by
     0.5% on each side.
   - `tests/test_atmosphere.py` now checks this padding and includes a regression
     case with model bin centers inset from the requested bounds, exercising
     comparison over inferred measured-channel edges.

3. **Made optional pRT import work with native Windows Fortran DLLs**
   - A direct pRT import failed on Windows because the Fortran runtime DLL
     directory was not in the process DLL search path.
   - `app/core/atmosphere.py` now discovers candidate MSYS2/Fortran runtime
     directories (`gfortran` on PATH, `MSYS2_ROOT`, `MSYS2_HOME`, standard
     install roots, and PATH entries) and registers directories containing
     `libgfortran*.dll` before importing pRT. Handles are retained for the
     process lifetime. Already-registered directories are not registered
     repeatedly.
   - A focused regression test verifies Windows-path discovery and one-time
     registration.
   - `README.md` documents runtime discovery and the `MSYS2_ROOT` option.
   - A real pRT model call through the adapter succeeded without a manual
     `os.add_dll_directory` call. A complete interactive UI launch has not yet
     been independently verified.

4. **Ran a real-backend synthetic recovery experiment**
   - Used actual pRT to generate a synthetic 21-channel spectrum with known
     H₂O VMR `1e-3`, then added a constant 40 ppm depth offset and seeded
     2.5 ppm Gaussian noise (RNG seed `20261012`).
   - The existing one-gas profile scan recovered:
     - best H₂O VMR: `1e-3`
     - fitted constant depth offset: `40.1853 ppm`
     - χ²: `22.657` for 21 channels
     - nominal conditional Δχ²=1 VMR interval:
       `[9.747e-4, 1.065e-3]`
   - This demonstrates recovery for a controlled synthetic case. It does **not**
     calibrate uncertainty coverage or establish that the scan is suitable for
     interpreting real observations.

5. **Validation and records**
   - Focused tests: `tests/test_atmosphere.py` and `tests/test_timeseries.py`,
     **25 passed**.
   - Full suite: **127 passed**.
   - Pylance reported no issues in the modified Python files.
   - Updated `docs/10-12-TODO.md` with verified runtime/recovery work,
     limitations, and remaining milestones.
   - Updated session todo state for the real pRT runtime, synthetic recovery,
     and roadmap verification milestones.

### Changed files

- `app/core/atmosphere.py`
  - Adds 0.5% pRT wavelength-boundary padding.
  - Adds Windows Fortran runtime DLL discovery before optional pRT imports.
  - Improves import-failure guidance for missing/unavailable pRT runtime.
- `tests/test_atmosphere.py`
  - Updates expected padded wavelength boundaries.
  - Adds channel-edge coverage and Windows DLL discovery regression tests.
- `README.md`
  - Documents optional Windows pRT runtime DLL discovery.
- `docs/10-12-TODO.md`
  - Marks real forward-model validation complete and records synthetic recovery
    evidence and limitations.
- `docs/10-12-HANDOFF.md`
  - This handoff document.

## Relevant implementation details

The atmospheric implementation is in `app/core/atmosphere.py`.

- `PetitRadtransParameters` describes the current deliberately simple
  isothermal, constant-composition model.
- `compute_petit_radtrans_transmission(...)` computes an optional
  petitRADTRANS 3 forward spectrum.
- `depth_spectrum_channel_edges(...)` infers channel edges from measured
  wavelength centers.
- `compare_model_to_depth_spectrum(...)` integrates the model into inferred
  top-hat channels and reports residuals/χ² diagnostics.
- `scan_single_gas_abundance(...)` profile-scans one configured gas VMR while
  holding other parameters fixed and fitting a constant depth offset.

The current scan is an exploratory conditional one-dimensional scan, **not**
a posterior sampler or calibrated abundance measurement. The exact model,
likelihood, fixed parameters, and caveats are described in code/docstrings and
`docs/10-12-TODO.md`.

The UI currently provides model overlay and one-gas scan actions in
`app/ui/transit_panel.py`. Do not present the current χ² or profile interval as
a validated goodness-of-fit probability or general posterior uncertainty.

## Environment and reproducibility

Workspace: `C:\Code\ECAS\ecas`

Selected Python executable used for tests:

`C:\Code\ECAS\ecas\.venv-1\Scripts\python.exe`

Native Windows setup performed with user approval:

- MSYS2 under `C:\msys64`
- UCRT64 `gfortran` 16.2.0 and `make`
- Microsoft MPI 10.1.12498.52
- petitRADTRANS 3.4.0, plus `meson-python` and `ninja` in the ECAS environment
- pRT inputs in `C:\Users\brett\petitRADTRANS\input_data`
- H₂O HITEMP R1000 line opacity and H₂-H₂/H₂-He BoRi CIA opacity

The pRT build originally needed venv Scripts and MSYS2 UCRT64 on PATH. Direct
imports originally needed `os.add_dll_directory(r"C:\msys64\ucrt64\bin")`; the
application helper now handles discoverable installations automatically.
HDF5 emitted a notice and selected `HDF5_USE_FILE_LOCKING=FALSE` when unset.
The manual successful modeling/recovery commands explicitly set this variable.
If a new environment fails to import or load pRT, first check that compiler,
MPI, DLL runtime, and opacity prerequisites are installed and discoverable;
report the concrete underlying import error rather than masking it.

## Scientific limitations that must remain visible

- The initial atmospheric model is clear, isothermal, and uses constant VMRs
  with an H₂/He background.
- The current observed-channel operation infers top-hat channels from centers;
  it does not use measured throughput or instrument line-spread functions.
- The comparison likelihood assumes independent Gaussian channel errors.
- Existing extracted depth uncertainties do not fully propagate baseline
  uncertainty or correlated noise.
- The one-gas scan fixes other model inputs and does not explore degeneracies
  with temperature, gravity, reference pressure/radius, clouds/hazes, stellar
  properties, or instrument response.
- The synthetic recovery test is one seeded noise realization and does not
  test coverage, robustness, realistic instrumental effects, or degeneracies.
- No multi-parameter posterior retrieval, sampler diagnostics, or published
  spectrum validation is implemented.

Do not make retrieval claims that exceed these limits. Preserve data
provenance, model assumptions, priors, nuisance treatment, and diagnostics in
future UI, project persistence, and exports. Add citations for external methods,
data, and software as that work is incorporated.

## Ordered next steps

### 1. Confirm and document the first retrieval scope

The proposed first step is intentionally narrow: one selected gas VMR plus a
constant depth offset, with temperature, gravity, reference radius/pressure,
stellar radius, and other species fixed. Use measured channel uncertainties as
the starting independent-error likelihood and label all results conditional.
This was proposed to the user but no selection was captured before handoff.
Confirm that scope with the user before implementing a materially broader
retrieval. Do not silently leap to multi-gas/cloud inference.

Write down:

- Free and fixed parameters, including units and allowed domains.
- Explicit priors (not merely parameter bounds) and why they are defensible.
- Treatment of the constant depth offset and measured uncertainties.
- Exclusions: instrument response, correlated noise, clouds/hazes, stellar
  contamination, temperature structure, and other known degeneracies unless
  specifically included.
- What diagnostics and failure warnings the user will receive.

Update `docs/10-12-TODO.md` when the scope is agreed.

### 2. Improve synthetic validation before treating intervals as uncertainty

Build repeatable tests/experiments around the real forward model and fixed
assumptions:

- Repeat over multiple independent noise seeds and injected VMRs.
- Measure bias and interval coverage, not just whether one best grid point is
  recovered.
- Test boundary cases where the gas is undetectable or the best value lies at
  the scan edge.
- Add known degeneracies and misspecification checks.
- Introduce realistic measured/instrument channel response once the source
  product exposes it; do not call inferred top-hat averaging an instrument
  response.
- Keep modeled correlated/systematic noise separate from reported measured
  uncertainty.

These results should determine whether the first uncertainty method can
reasonably remain a profile-likelihood interval or whether a posterior sampler
is needed for the chosen scope.

### 3. Implement uncertainty-aware inference only after scope and tests

Choose an optimizer/sampler based on the agreed free parameters and validation
requirements. Ensure outputs include parameter intervals/distributions,
likelihood/fit diagnostics, reproducibility controls, and clear warnings for
non-convergence, unbounded parameters, and model limitations. Add focused
regression tests for both recovery and failure cases.

Do not represent the present Δχ² scan as a posterior. If it is retained as a
quick-look tool, label it explicitly and keep it distinct from the validated
retrieval.

### 4. Validate externally and integrate responsibly

- Select appropriate published transmission spectra and independent literature
  results; document source URLs/citations, data versions, and any conversions.
- Compare assumptions and expected results without treating agreement alone as
  proof of correctness.
- Integrate validated controls/results into the Transit UI, `.ecas` project
  persistence, and exports.
- Show free/fixed parameters, priors, provenance, assumptions, uncertainties,
  fit diagnostics, and limitations with results.
- Test archive and local-data flows so extracted spectra can be analyzed
  consistently.

### 5. Close the roadmap milestone and defer emission until then

After targeted regressions, run the full suite, inspect diagnostics, update
`docs/10-12-TODO.md`, and mark milestones complete only with evidence. Continue
to defer emitted-flux work until the transmission workflow has met its
validation criteria.

## Immediate verification commands

From the workspace root in PowerShell:

```powershell
& 'C:\Code\ECAS\ecas\.venv-1\Scripts\python.exe' -m pytest -q tests\test_atmosphere.py tests\test_timeseries.py
& 'C:\Code\ECAS\ecas\.venv-1\Scripts\python.exe' -m pytest -q
```

The results recorded at handoff are 25 focused tests and 127 total tests passing.
Re-run after subsequent code changes.

