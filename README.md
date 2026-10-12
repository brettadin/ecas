# ECAS — Exoplanetary Characterization & Astrochemistry Suite

    pip install -r requirements.txt
    python main.py            # dashboard (mock gases by default)
    python main.py --check    # headless self-check
    python main.py --no-3d    # skip the PyVista viewport
    python -m pytest -q       # tests (GUI tests run offscreen)

Workflow: **Spectroscopy** tab (gas sliders, P/T broadening, unit tabs) → *Mock observation* or
*Load CSV* → *Fit gas multipliers* → anomaly list + candidate molecules.
**Transit & 3D** tab: look up a planet → *Fetch TESS light curve…* for broadband photometry,
or *Fetch JWST spectra…* for wavelength-resolved IR time series → *Analyze transit* → scrub/play
the synchronized sky view and 3-D orbit → *Extract spectrum → Identify* sends the measured
wavelength-dependent transit depths to atmospheric analysis.
In the Transit panel, **Extract measured spectrum** first plots the measured transit depths with
uncertainties (ppm versus µm); optional adjacent-channel binning uses inverse-variance weights.
Only **Send spectrum to Identify** changes to the molecular-comparison workspace. That comparison
still uses a relative pseudo-transmittance and does not return atmospheric abundances. Spectral
time-series inputs need at least 50 usable integrations and 8 wavelength channels to build the
light curve. Saving an `.ecas` project preserves the downloaded wavelength-by-time series and its
archive provenance so it can be reopened and re-analyzed.
**File** menu: save/open `.ecas` projects; export CSV, Markdown report, PNG.

### Optional physical transmission model

The Transit panel can overlay a **petitRADTRANS 3** clear-atmosphere forward spectrum on the measured
transit depths. Install the optional backend separately; ECAS's normal installation does not require it.
On native Windows, petitRADTRANS needs a Fortran compiler (MSYS2 or Visual Studio); follow its
[installation guide](https://petitradtrans.readthedocs.io/en/latest/content/installation.html), then run:

    pip install numpy meson-python ninja
    pip install -r requirements-atmosphere.txt --no-build-isolation

For an MSYS2 UCRT64 build, ECAS discovers the Fortran runtime DLL directory from `gfortran` on
`PATH`, `MSYS2_ROOT`/`MSYS2_HOME`, or the standard `C:\msys64` installation. If MSYS2 is installed
elsewhere, set `MSYS2_ROOT` to its root directory before launching ECAS. Restart ECAS after
installing or changing the compiler runtime.

Install/download the requested pRT opacity data before generating a model. The initial in-app model is
explicitly a clear, isothermal atmosphere with constant gas volume mixing ratios and an H₂/He background.
The model curve is integrated over inferred top-hat wavelength-channel edges before comparison with the
measured spectrum. The residual tab and reported χ²/N for this overlay are diagnostics only. The separate
**Scan one gas abundance** action runs a conditional, one-dimensional pRT profile scan for a selected
configured gas; all other gas abundances and atmospheric inputs are fixed, while a constant transit-depth
offset is fitted at each grid point. It reports the best grid VMR and a Δχ²=1 profile interval when the
scan crosses that threshold (otherwise it reports the unbounded scan side). This is an exploratory
conditional estimate, not a posterior or calibrated abundance measurement. Neither mode includes
instrument line-spread convolution or correlated-noise likelihoods, and the abundance scan does not
marginalize over other atmospheric parameters. The gas names supported by ECAS's initial settings are
H2O, CO2, CH4, O3, N2O, CO-NatAbund, NH3, HCN, Na, and K, subject to availability in the installed
pRT opacity database.

### Time-correlated noise diagnostic

For a loaded spectral time series, **Characterize time-correlated noise** removes the fixed-geometry
transit template and compares robust MAD residual scatter across consecutive-integration bin sizes with
the white-noise 1/√N expectation. The plot shows the median beta-style ratio across wavelength channels
and the 16th–84th percentile channel spread; gaps larger than 1.5 median cadences are not bridged.
This is an empirical residual diagnostic, not a validated uncertainty correction: it does not modify
transit-depth errors, model baseline-estimation uncertainty, or replace instrument/systematics modeling.

Real ORCA data: edit `app/data/gases.json` and set a gas's `orca_out` to the `.out` path.
Input spectra: transmittance 0–1, columns `x, y[, sigma]`; light curves: `time(days), flux[, err]`.
TESS photometry and JWST `x1dints` spectral time series can be searched and downloaded from MAST after
selecting a planet in the lookup; previously downloaded products are reused from `~/.ecas/cache`.
TESS is broadband and cannot produce a spectrum; UV-vis HST spectral
extraction is not yet supported. The atmospheric fitter currently consumes a continuum-relative
pseudo-transmittance derived from the measured transit depths, so its fitted abundances are relative,
not absolute.

## Related projects and methods

These projects were reviewed as scientific or implementation references. ECAS does not copy or bundle
their source code, data, or pretrained models. License notes refer to the repositories' published
licenses; review does not establish that their scientific results or artifacts have been independently
validated.

- [petitRADTRANS](https://github.com/exoclime/petitRADTRANS) — optional atmospheric radiative-transfer
  backend used by ECAS for exploratory forward spectra. See its
  [documentation](https://petitradtrans.readthedocs.io/) and the license in its repository.
- [ExoIris](https://github.com/hpparvi/ExoIris) — joint spectrophotometric time-series modeling reference.
  Its repository is GPL-3.0; ECAS does not reuse its code.
- [StarRotator](https://github.com/Hoeijmakers/StarRotator) — MIT-licensed stellar-activity and
  spectral-modeling reference for possible future stellar-contamination work.
- [Chromatic noise-characterization example](https://zkbt.github.io/chromatic/example-timeseries-spectra/#characterize-the-noise)
  — MIT-licensed project/example; a reference for robust time-series noise diagnostics, not yet an ECAS
  uncertainty correction.
- [exosports/transit](https://github.com/exosports/transit) — reviewed but not used. Its repository's
  test-version license restricts modification and redistribution; do not use its code or output unless
  the license and intended use are confirmed to permit it.
- [ExoSpectraNet](https://github.com/Tharun007-TK/exoplanet-spectrum-recovery-ml) — MIT-licensed
  research lead for machine-learning spectrum recovery. Its reported training data, checkpoints, and
  scientific performance have not been independently validated; ECAS does not use its model or artifacts.

## Loading your own data
**Load…** (Spectroscopy tab) accepts CSV/TSV/text (any delimiter, decimal commas, comment/metadata lines),
FITS tables (e.g. JWST `x1d`) and JCAMP-DX. An import dialog shows the detected columns, x unit
(cm⁻¹ / µm / nm / Å) and y scaling (transmittance, percent, absorbance, optical depth, or
"normalise to continuum" for raw flux) so you can correct any guess. Files appear immediately; tick/untick
them in the list. **Identify peaks (no model)** finds absorption features directly in the data and matches them
to known bands. **Fit gas multipliers** can also fit a continuum level/tilt, and by default only fits inside the
model's spectral range (the model covers fundamentals only, not overtones or electronic bands).
Comma-delimited archive tables with blank cells and text-valued citation columns are supported. For archive
flux tables (for example columns `CENTRALWAVELNG` and `FLAM`), the importer selects the wavelength and flux
columns and normalises flux to a continuum for display/feature analysis; this does not turn emitted flux
into a planetary transmission spectrum. If archive error columns are empty, the imported spectrum has no
uncertainties and is displayed without error bars.
Archive `FLAM`/`FNU` spectra are recognized as emission-flux inputs: they are displayed in both Spectroscopy
and Identify as relative emission data; transmission gas fitting and transmission peak identification are
disabled for them, and **Fit blackbody emission continuum** provides a simple
color-temperature baseline with arbitrary scale. Because import normalization discards absolute calibration,
this does not determine luminosity, radius, or atmospheric composition; molecular emission modeling is not
yet implemented.

The Identify tab's **Browse NIST…** action supports both IR and UV/Vis JCAMP references. UV/Vis data are
normally reported by NIST on a wavelength axis and are converted to wavenumber for matching; they are
treated as relative spectral shapes because solvent, path length, and concentration conditions matter.
The simulator's built-in IR bands cover roughly 400–4200 cm⁻¹, so observations around 10,000–19,500 cm⁻¹
need imported UV/Vis references with overlapping coverage.

Optional 3-D viewport: `pip install pyvista pyvistaqt` (otherwise a 2-D orbit view is used).
