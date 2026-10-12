# ECAS — Changelog (patch notes)

Newest first.  Versions follow `0.PHASE.PATCH`.  Every entry says what changed, why, and how it was verified.

## Unreleased — Conditional retrieval contract and edge diagnostics (Oct 2026)
**Added**
- Formalized the first retrieval scope: one gas VMR plus a fitted constant depth
  offset, with all other atmospheric inputs fixed and independent measured
  channel errors as the quick-look likelihood.
- Abundance scan results now include machine-readable warnings when the best fit
  is on the scan boundary or a conditional Δχ²=1 bound is not established.
- Transit UI reports those warnings alongside the existing scientific caveats.
- Added `core/retrieval_validation.py` with seeded synthetic-observation,
  multi-injection/multi-seed execution, recovery-bias, and interval-coverage
  diagnostics. The forward-model and scan callbacks are explicit so the same
  harness can be used with real petitRADTRANS.
- Added `tools/run_real_profile_validation.py`, an offline real-pRT experiment
  runner using the installed H₂O HITEMP and H₂/He BoRi opacity set. Its initial
  20-case study (two H₂O VMRs × ten seeds) gave 90% conditional Δχ²=1 interval
  coverage, with all intervals bounded and no scan-edge best fits. This is
  preliminary scope-specific evidence, not general uncertainty calibration.
- Expanded the real-pRT study to 40 cases across four H₂O VMR regimes. The
  weak (`1e-5`) and strong (`1e-2`) regimes were correctly exposed as mostly
  scan-boundary/unbounded cases, while `1e-4` and `1e-3` each showed 90%
  conditional interval coverage. The runner now supports concise per-injection
  summaries with `--summary-only`.
- Added structured boundary semantics to abundance results. The scan now
  distinguishes interior profiles from lower/upper scan-edge fits and exposes
  a conditional one-sided limit only when the opposite Δχ²=1 crossing exists.
  The Transit report labels these as upper/lower limits or explicitly
  unestablished bounds.
- Extended the real-pRT runner to evaluate condition grids. A 40-case study
  across H₂O VMRs `1e-4`/`1e-3` and 2.5/5 ppm noise measured 90% and 85%
  aggregate conditional interval coverage respectively, with all intervals
  bounded and no scan-edge fits. This remains scope-specific validation.
- Completed and documented the archive UI path: planet lookup, TESS light-curve
  retrieval, and JWST x1dints retrieval are wired into the Transit workspace.
  Downloaded TESS light curves now retain the MAST target/observation and
  selected planet-record provenance in their session metadata, matching the
  existing JWST provenance behavior. The roadmap's former “UI hookup pending”
  note is now cleared.
**Why**
- A bounded profile scan must not present an edge-limited result as a measured
  abundance or a missing crossing as a finite uncertainty interval.
**Verified**
- Reference-library and identification focused suite: **13 passed**; Python compilation passed
  for the modified core and Identify UI modules. The workspace-local full
  suite reached **130 passed, 1 skipped**; five isolated GUI subprocess tests
  were blocked by the environment's inaccessible global pytest temp directory
  (`C:\\Users\\brett\\AppData\\Local\\Temp\\pytest-of-brett`), not by
  reference-library assertions.
- `tests/test_atmosphere.py`: 19 passed, including validation-helper tests.
- Full suite: 132 passed with child GUI pytest processes given a workspace-local
  `--basetemp`; one existing pytest cache collision warning remained.
- Archive/core checks plus the existing archive GUI workflow: 10 passed; the
  same pytest cache collision warning remained.
- Extended the real-pRT validation runner with separately labeled synthetic
  temperature and gravity mismatch axes. The scan keeps the retrieval values
  fixed while synthetic data can use alternate values; model caching now keys
  on VMR and both synthetic fixed parameters. A smoke mismatch run at
  1000 K synthetic versus 1200 K scan (one seed, injected H₂O `1e-3`) produced
  a 0.875 dex absolute log-VMR bias and 0% interval coverage, while remaining
  bounded. This is a diagnostic case, not a population-level calibration.
- Completed small multi-seed temperature and gravity mismatch grids. With three
  seeds per interior VMR, 1000 K and 1400 K synthetic data versus the 1200 K
  scan both gave 0% interval coverage; the matched 1200 K control gave 100%
  coverage. Separately, 500 and 2000 cgs synthetic gravity versus the fixed
  1000 cgs scan both gave 0% coverage; the 2000 cgs condition was boundary-
  limited in half its cases. These are controlled diagnostics, not universal
  failure rates.
- Extended the mismatch runner to reference pressure, planet radius, and
  stellar radius. Small two-seed H₂O `1e-3` checks found 0% coverage at 0.001
  and 0.1 bar versus the fixed 0.01 bar scan, at 0.9 and 1.1 Rjup versus the
  fixed 1 Rjup scan, and at 0.9 and 1.1 Rsun versus the fixed 1 Rsun scan;
  matched controls had 100% coverage. These are diagnostic checks, not
  calibrated failure probabilities.
- Completed the available offline loader validation pass: x1dints tests cover
  both per-extension and vector-column JWST layouts, BJD/MJD timing, missing
  timing tables, wavelength-grid consistency, and malformed files; JCAMP and
  emission-table tests also pass. No real observation or downloaded reference
  files are present in the repository, so live/real-file validation remains
  explicitly open.
- Added a separate instrument-response validation path. Synthetic observations
  can now apply a positive throughput function and Gaussian line-spread blur
  before the fixed profile scan; condition metadata records the response slope
  and blur width. A one-seed two-channel-width blur smoke case produced 0%
  coverage and 0.219 dex bias versus 100% coverage for the matched control.
  Added regression coverage for blur behavior and invalid sigma handling.
- Added a separate correlated-channel-noise path using an exponential
  covariance length in channel units. The scan still assumes independent
  errors. A two-seed length-4 smoke case reduced coverage to 50% from 100% for
  the independent matched control, with all intervals bounded. Added invalid
  correlation-length regression coverage.
- Integrated transmission-retrieval persistence and reporting. `.ecas` project
  files now preserve the extracted `DepthSpectrum` arrays, and the Transit
  report lines included in Markdown exports include the current transmission
  retrieval state/profile-scan diagnostics. Added project round-trip coverage
  for the saved depth spectrum.
- Added a persistent View → Activity log dock to the main window. It records
  project saves/opens, exports, and warnings with timestamps, keeps the latest
  500 events, and complements the transient status bar for longer workflows.
- Completed the reference-library integration slice: NIST WebBook CAS/index
  browse-and-import with URL provenance, a credential-safe HAPI cross-section
  adapter/import path for sampled `cm-1`/`cm2 molecule-1` output, and an ExoMol
  `.states`/`.trans` (including gzip/bzip2) line-strength importer. ExoMol
  references are explicitly shape-only because raw transitions are not yet a
  pressure-broadened, partition-normalized cross-section.
- Added Identify-panel actions for Browse NIST, Import ExoMol, and Import HAPI
  CSV, with persisted library entries and citation-log records. The new
  reference assumptions remain explicit because credentials and raw line-list
  data must not be presented as calibrated abundance cross-sections.
- Improved molecule identification with a continuum-corrected,
  uncertainty-weighted `rank_references` screening pass. The Identify panel
  can now auto-rank the selected library, retain the strongest candidates, and
  send that smaller set to the existing non-negative joint mixture fit. The
  result reports template score and correlation as screening diagnostics rather
  than overstating them as posterior detection probabilities.
- Research basis recorded for the method: chi-square template matching and
  cross-correlation are established spectral-analysis operations; NIST WebBook
  provides IR and UV/Vis reference collections. See the cited links in the
  session handoff and final report.
- Expanded the default MOCK scan library beyond the original five gases with
  SO₂, NH₃, HCN, and CO. These candidates are enabled and visible by default,
  but start at zero multiplier so they do not silently alter a fresh model.
- Emission-flux observations are now rendered in the Identify workspace as
  relative flux, with an appropriate axis/range and an explicit message that
  the absorption reference fitter is not being applied. The Spectroscopy view
  continues to support the exploratory blackbody continuum fit.
- Identify now clips its default fit window to the actual overlap between the
  uploaded spectrum and selected references, with a clearer no-overlap error.
  The opaque-spectrum error now explains that saturated points are lower limits
  and suggests checking y-axis interpretation or continuum normalization.
- NIST browsing now supports UV/Vis JCAMP retrieval in addition to IR. UV/Vis
  `LOG EPSILON` spectra are imported as non-quantitative shape references after
  wavelength-to-wavenumber conversion, preserving the solvent/path-length
  limitation instead of presenting them as abundance-ready cross-sections.
- Recorded the user's real-file finding: the currently available reference data
  do not overlap the uploaded 10,000–19,500 cm⁻¹ spectrum. ECAS must report
  this as an evidence gap, not as a molecular non-detection; real UV/Vis
  reference validation remains open.

## 0.8.0 — Identification, sources & citations, professional UI (Oct 2026)
**Added**
- **Multi-molecule identification** (`core/identify.py`, *Identify* tab).  The observed spectrum is explained as a
  non-negative combination of reference spectra by exact bounded linear least squares in optical-depth space.
  Reports per molecule: amplitude ± 1σ, **drop-one significance** (detected only above a chosen σ), **degeneracy
  diagnostics** (shape similarity + amplitude correlation; strongly overlapping molecules are flagged because their
  split cannot be trusted), and **per-feature attribution** (share of each peak's integrated optical depth explained by
  each molecule, overlap flag, unexplained remainder).  Nearly-opaque points are excluded as lower limits.
  Noise is estimated in transmittance (where it is additive) and propagated to optical depth.
- **Reference-spectrum library** (`core/library.py`): simulator gases + **JCAMP-DX import** (NIST WebBook downloads etc.;
  header title/CAS/formula/origin kept as provenance; transmittance/absorbance/absorption-coefficient handled;
  quantitative vs shape-only is tracked and shown — shape-only amplitudes are never presented as abundances).
  Persisted in `~/.ecas/library`.
- **Source & citation registry** (`core/sources.py`, *Help → Sources & citations…*): every database/paper/tool has a
  formatted reference, licence note and a `verified` flag; a per-session log records what was actually used with access
  date and exact query; copy / save Markdown / BibTeX; references are appended to exported reports.
- **Archive clients** (`core/archives.py`): NASA Exoplanet Archive TAP (planet + star parameters from the default
  parameter set, each with its literature reference), name search, on-disk cache; MAST/astroquery wrappers for JWST/HST
  spectral products (x1dints / x1d / whtlt).  Network is injectable; covered by offline tests.
- **UI redesign** (`ui/theme.py`): consistent dark instrument theme, collapsible sidebar sections, workspace tabs
  (Spectroscopy · Identify · Transit & 3D), primary-action buttons, themed tables/lists/sliders/scrollbars.
- **Planet lookup** (Transit tab → *Look up planet…*): NASA Exoplanet Archive default parameters with their literature
  references; fills R★/M★/Teff, records the citation automatically, and **compares your own fit with the published
  values** (flags only disagreements larger than both 5 % and 3σ of the fit's own error).
- **Spectral time series** (`core/timeseries.py`): JWST `*_x1dints.fits` loader (one spectrum per integration) and a
  time × wavelength CSV matrix loader; **spectrum at time t** (F(t)/F_out per wavelength) follows the transit scrubber;
  **Extract spectrum → Identify** turns in-transit vs out-of-transit into a transmission spectrum and sends it to the
  Spectroscopy/Identify tabs.  The x1dints layout was implemented from documentation recall and tested on synthetic files
  built to that layout — **verify against one real file**.
- **Single-transit analysis**: with the archive period the search is skipped and only (t0, Rp/R★, a/R★, b) are fitted
  (JWST visits usually contain one transit); without it, a single transit is refused with instructions instead of
  returning a bogus short-period fit.  "Transits seen" now counts epochs where the *data* really dip.
**Fixed**
- JCAMP-DX reader scaled `DELTAX` by `XFACTOR` (wrong per the format); now uses real X units, derives spacing from
  FIRSTX/LASTX/NPOINTS when absent.
- Chi² from self-estimated noise is not comparable between models → RMS residual in T is now reported alongside it.
**Research notes (verified this session)**: NASA Exoplanet Archive *Atmospheric Spectroscopy Table* (DOI 10.26133/NEA36)
replaced the retired transmission/emission-spectra tables; TAP `ps` table columns incl. `pl_refname`/`st_refname`;
astroquery.mast for JWST; HITRAN needs an account/API key (HAPI); NIST WebBook = SRD 69 (DOI 10.18434/T4D303), IR spectra
as JCAMP-DX, mostly qualitative.  **Not verified live**: MAST wrappers and the TAP endpoint were exercised only against
fakes/fixtures because the build sandbox has no access to those hosts — please test them once on a networked machine.
**Verified**: 97 tests (each GUI test in its own process).

## 0.7.0 — Bug-fix release from the first hands-on review (Oct 2026)
**Fixed**
- **Imported spectra shown as a single point / flat line.** Root causes reproduced: nm values read as µm
  (x up to 2600 µm, far off-axis), percent never converted, tiny absolute flux (<0.1) drawn on a 0–1 axis,
  decimal-comma files rejected.  New table reader (`core/tables.py`) + explicit unit/scale handling +
  import dialog that shows every assumption and lets you override it.  Data now draws immediately
  (no fit needed) and the view widens to include it.
- **Messy files.** Free-text metadata lines, `#`/`%`/`//` comments, ragged rows, any delimiter,
  UTF-8/Latin-1, FITS tables (JWST x1d, TESS), JCAMP-DX.  Column picker + auto-suggestion.
- **Gaussian line shape did nothing.** It used only the ~0.002 cm⁻¹ Doppler width (min T 0.997).  It now uses
  the quadrature sum of Doppler and total Lorentzian width, so pressure/envelope act on it.
- **Transit "Analyze" froze the app on non-light-curve files.** A spectrum treated as a time series cost 52 s on
  the UI thread.  Now: validated input, bounded period-search cost model, spectrum detection from headers,
  plausibility checks (in-transit sample count), and analysis runs on a worker thread.
- **3-D viewport message / dead controls.** Now explains the missing module with the install command, and
  falls back to a 2-D projected orbit view so *planet ×*, *Observer view* and *Oblique view* always work.
- **R★/M★/Teff did nothing.** They now update derived quantities live (Rp, a, v, Teq), on-chart captions and the
  star colour (blackbody colour from Teff).  Speed slider added (frame-rate independent animation).
**Added**
- Dataset list (multiple files, show/hide, choose which is fitted), Zoom-to-data / Reset-view, Show-model toggle.
- Model-free **Identify peaks** → matches features to engine gases *and* the band table.
- Click an anomaly in the list *or on the chart* to highlight + zoom to it.
- Optional fitted continuum (level + tilt) and a fit window limited to the model's range.
**Verified**: 73 tests (core + GUI), GUI tests each in an isolated process (pyqtgraph segfaults when ~10
windows are built in one process — never in a real single-window session); 10 consecutive clean runs.

## 0.6.0 — Projects & exports
`.ecas` project files (zip: JSON + npz), CSV/Markdown/PNG exports, `pyproject.toml`, README.

## 0.5.0 — Transit photometry & 3-D
Transit model, BLS period search, refit, derived planet properties; spectral-light-curve depth extraction;
synchronized light-curve tracker / sky view / PyVista 3-D.

## 0.4.0 — Inverse engine
Bounded least-squares multiplier fit, residual anomalies, candidate identification.

## 0.1–0.3 — Core
ORCA IR parser, vectorized Lorentz/Gauss/Voigt engine with two-tier cache, PyQt6 + pyqtgraph dashboard.
