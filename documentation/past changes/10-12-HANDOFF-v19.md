# ECAS Identification Handoff v19

Updated 2026-10-12.

## Work completed this session

Improved the molecule-identification workflow so a broad reference library can
be screened before the joint fit:

- Added Qt-free `rank_references(...)` in `app/core/identify.py`.
- Each candidate is resampled onto the observation grid, restricted to shared
  coverage, converted to optical depth, and compared after fitting/removing a
  level-plus-tilt continuum.
- Supplied transmittance uncertainties are propagated to optical-depth errors;
  otherwise the screening uses unit weights. The output includes
  `sqrt(delta-chi2)`, continuum-corrected correlation, delta-chi-square, and
  usable-point count.
- Added an Identify-panel “Auto-rank candidates” action. It retains the
  strongest candidates and then uses the existing non-negative joint mixture
  fit, preserving its overlap/degeneracy and relative-amplitude warnings.
- Added regression coverage showing a known multi-band target outranks a
  decoy. The ranking score is explicitly documented as a screening diagnostic,
  not a posterior probability or calibrated detection significance.
- Updated the roadmap, changelog, and TODO pointer.

## Research basis and citations

The design follows the established spectral-analysis pattern of resampling
templates and selecting the smallest weighted chi-square, as documented by
[specutils template matching](https://specutils.readthedocs.io/en/stable/api/specutils.analysis.template_match.html).
Cross-correlation is also used for molecular-template searches in exoplanet
spectra; see [Boucher et al. 2018](https://arxiv.org/abs/1807.10769). The
library strategy uses NIST’s documented IR and UV/Vis collections; see the
[NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) and its
[spectral-data guide](https://webbook.nist.gov/chemistry/guide/).

These sources support the method and data-source choices, not a claim that ECAS
is already calibrated for arbitrary instruments, temperatures, resolutions, or
mixtures.

## Verification

- `tests/test_identify.py`: **13 passed**.
- Python compilation passed for `app/core/identify.py` and
  `app/ui/identify_panel.py`.
- The new ranking has not yet been validated against a real downloaded
  observation or a real NIST/HAPI/ExoMol file.

## Next work

1. Add resolution/LSF matching and optional wavelength/velocity shift grids;
   these are important when a reference and observation are sampled at
   different resolutions or have calibration offsets.
2. Add an explicit ranked-candidate table/plot to the UI so users can inspect
   all scores before accepting the auto-selection.
3. Validate against real reference files and injected mixtures spanning S/N,
   resolution, continuum curvature, and overlapping molecules.
4. Keep improving the physical joint model: pressure/temperature-dependent
   templates, instrument response, correlated noise, and eventually a proper
   retrieval rather than relative shape amplitudes.

