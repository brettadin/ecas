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
- [ ] Reference library: HITRAN cross-sections/lines via HAPI (needs user API key), ExoMol line lists, in-app NIST browsing
- [x] **0.8 Online archives (core)**: NASA Exoplanet Archive TAP client, MAST wrappers — **UI hookup pending** (planet lookup dialog, MAST browser)
- [ ] Archive: published-spectra download from the Atmospheric Spectroscopy Table (verify its TAP columns first)
- [x] **0.8 Multi-molecule decomposition** (Identify tab)
- [~] **UI redesign**: theme + collapsible sections + workspace tabs done; still to do: status/log panel, onboarding/empty states, icons, Spectroscopy tab layout polish
- [ ] 1.0 Concentration estimates with uncertainties (needs path length / column density model, T–P profile)
- [x] Spectrum-at-time from spectral light curves (x1dints / matrix) + extraction → Identify (verify x1dints on a real file)

## Next up (suggested order)
1. MAST browser dialog (search target, list x1dints/x1d, download) — core wrappers exist (`core/archives.py`), no UI yet
2. Per-integration spectra: model-based removal of systematics (limb darkening per channel, red noise) before extraction
3. Real-file validation pass: x1dints, MAST/TAP live calls, NIST JCAMP downloads
4. HITRAN (HAPI) reference import with API key; ExoMol
5. Concentration estimates (needs quantitative references + path-length/column model)

## Known issues / user notes
(Add the user's findings here.)
