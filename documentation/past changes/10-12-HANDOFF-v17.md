# ECAS Transmission Retrieval Handoff v17

Updated 2026-10-12.

## Work completed this session

Added a persistent user-facing Activity log, addressing a roadmap UI gap:

- View → Activity log opens a dockable bottom panel;
- project saves and opens are recorded;
- model, fit, anomaly, report, and plot exports are recorded;
- warnings are recorded with timestamps;
- the log retains the latest 500 events and does not replace the transient
  status-bar message.

The session also confirmed that the new transmission-spectrum project
persistence and report integration remain compatible with the existing GUI
project round-trip.

## Verification

- GUI project round-trip: **1 passed**.
- Project serialization and export checks: **2 passed**.
- Python compilation passed for the modified UI modules.
- Existing pytest cache collision warning remains environmental.

## Remaining product work

1. Add structured retrieval-result export where useful; do not serialize a
   conditional profile scan as a posterior.
2. Improve onboarding and empty states across the workspace tabs.
3. Obtain real JWST x1dints and JCAMP files for validation.
4. Keep emission deferred until the transmission workflow is mature.
