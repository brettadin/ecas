# ECAS Default Library and Emission Display Handoff v20

Updated 2026-10-12.

## Work completed this session

- Expanded the default configured MOCK candidate set from five gases to nine:
  H₂O, CO₂, CH₄, O₃, N₂O, SO₂, NH₃, HCN, and CO.
- Added representative MOCK bands for NH₃, HCN, and CO. They are explicitly
  placeholders for UI/testing and do not replace HITRAN, ExoMol, or measured
  spectra.
- Enabled SO₂, NH₃, HCN, and CO in the default library while setting their
  initial multipliers to zero. They are available to Identify and scanning but
  are not silently injected into a fresh forward model.
- Updated regression expectations and mock-observation tests so the expanded
  model remains honest about optional species and intentionally excluded
  anomaly tests.
- Emission-flux observations now appear in the Identify plot as relative
  emission data, with a flux axis, automatic data framing, and a clear note
  that the absorption/transmission molecular fitter is not applied.
- Updated README, roadmap, changelog, and TODO pointer.

## Verification

- Targeted emission/default-library/identification checks: **15 passed**.
- Non-GUI suite: **138 passed, 1 skipped, 10 deselected**.
- Python compilation passed for the modified mock-data and Identify UI files.

## Current boundary

Emission is visible now, but molecular emission identification is not yet
implemented. The current references are absorption/optical-depth shapes and
cannot be reused as emission templates without a source function, temperature,
opacity/emissivity treatment, and instrument model.

## Next work

1. Add a dedicated emission-analysis core path: continuum removal, emission
   feature detection, and a separate emission-template interface.
2. Decide whether the first emission model should use blackbody-plus-opacity
   templates or an atmosphere forward model; do not label the result an
   abundance until that physical model is validated.
3. Replace MOCK candidates with user-imported or database-backed quantitative
   references where available, preserving source and condition metadata.
4. Add ranked-candidate display and resolution/LSF matching for transmission
   identification.

