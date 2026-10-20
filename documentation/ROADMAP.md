# ECAS — Roadmap & hand-off notes

> If a session ends abruptly, read this file and `CHANGELOG.md` first.  Update the checkboxes as work lands.

## Product goal (from the author)
Import exoplanet observations (a **spectrum** or a **transit data set**, with messy real-world headers),
**assign peaks/features to individual molecules or combinations of molecules** (including overlapping
signals), and eventually estimate **concentrations**.  Use a wide range of molecules from **reliable,
cited sources**; get observation data from **reliable databases**; **everything cited and easily
referenced**.  Professional, high-end UI.  Accuracy above all.

## Principles
1. Never invent data.  Every reference spectrum, line list and planet parameter carries a citation
   (source, version/date retrieved, licence).  Mock data is always labelled MOCK.
2. Every assumption made on import is shown to the user and overridable.
3. Core physics is Qt-free and tested; the GUI is a thin layer.
4. Honest limits: harmonic ORCA fundamentals ≠ real spectra (no overtones, rotational structure, hot bands).

## Status
- [x] Phases 1–6 (see CHANGELOG)
- [x] 0.7 bug-fix pass
- [x] **0.8 Sources & citations registry** (every dataset → citation, licence, retrieval date; exported in reports)
- [x] **0.8 Reference-spectrum library** — JCAMP-DX import done (NIST WebBook, PNNL files)
- [x] Reference library integrations: credential-gated HAPI sampled cross-section import, ExoMol
  `.states`/`.trans` line-strength import, and in-app NIST WebBook IR browse/import. Full
  authenticated HAPI retrieval and pressure-broadened ExoMol cross-section synthesis remain
  deliberate follow-ons; imported ExoMol data is labelled shape-only.
- [x] **0.8 Online archives**: NASA Exoplanet Archive TAP client and MAST wrappers with planet lookup, TESS light-curve, and JWST x1dints dialogs; downloaded inputs retain source/provenance metadata
- [ ] Archive: published-spectra download from the Atmospheric Spectroscopy Table (verify its TAP columns first)
- [x] **0.8 Multi-molecule decomposition** (Identify tab), including continuum-corrected
  candidate ranking before the joint fit; still requires real-file and library-coverage validation
- [~] Emission-spectrum display is now available in Spectroscopy and Identify; molecular emission
  templates/retrieval remain open because the current library and fitter describe absorption.
- [x] Identify now reports actual observation/reference overlap instead of assuming the simulator's
  400–4200 cm⁻¹ IR range, and NIST browsing supports UV/Vis references for higher-wavenumber data.
- [~] **UI redesign**: theme + collapsible sections + workspace tabs + status/log panel done;
  still to do: onboarding/empty states, icons, Spectroscopy tab layout polish
- [ ] 1.0 Concentration estimates with uncertainties (needs path length / column density model, T–P profile)
- [x] Spectrum-at-time from spectral light curves (x1dints / matrix) + extraction → Identify (verify x1dints on a real file)

## Next up (suggested order)
1. Real-file validation pass: x1dints, MAST/TAP live calls, NIST JCAMP downloads, and user-supplied
   HAPI/ExoMol files
2. Per-integration spectra: model-based removal of systematics (limb darkening per channel, red noise) before extraction
3. Retrieval validation: fixed-parameter misspecification, instrument response, and correlated-noise axes
4. Structured retrieval-result export and final onboarding/empty-state polish
5. Concentration estimates (needs quantitative references + path-length/column model)

## Known issues / user notes
- [ ] **Confirmed user limitation (2026-10-12):** the currently available
  built-in and imported reference set does not overlap the user's uploaded
  10,000–19,500 cm⁻¹ spectrum. Identify must not claim a molecular result for
  that file until references covering that range are obtained and validated.
  The simulator's built-in mock/ORCA bands remain primarily 400–4200 cm⁻¹;
  NIST UV/Vis access is available but has not yet been validated against the
  user's file or shown to provide the same phase/measurement conditions.
