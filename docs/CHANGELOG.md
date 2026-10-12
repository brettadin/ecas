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
**Why**
- A bounded profile scan must not present an edge-limited result as a measured
  abundance or a missing crossing as a finite uncertainty interval.
**Verified**
- `tests/test_atmosphere.py`: 19 passed, including validation-helper tests.
- Full suite: 130 passed with child GUI pytest processes given a workspace-local
  `--basetemp`; one existing pytest cache collision warning remained.

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
