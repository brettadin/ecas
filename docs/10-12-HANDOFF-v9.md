# ECAS Transmission Retrieval Handoff v9

Updated 2026-10-12.

## Work completed this session

The work broadened to the archive/UI portion of the application. Inspection
confirmed that the following path is already implemented and wired into the
Transit workspace:

- NASA Exoplanet Archive planet lookup and citation capture;
- TESS MAST search, product selection, download, FITS loading, and citation;
- JWST MAST x1dints search, download, validation through the x1dints loader,
  metadata capture, and citation.

The stale `docs/ROADMAP.md` entry saying MAST UI hookup was pending was
corrected. TESS downloads now also store `mast_target`, `mast_observation`,
`mast_obsid`, `mast_instrument`, and the selected archive planet record in the
loaded `LightCurve.meta`, so downstream reports/session code can identify the
source just as it can for JWST spectral series.

## Verification

- Archive/core tests and the existing archive GUI workflow: **10 passed**.
- The existing pytest cache collision warning remains environmental and does
  not affect test results.
- A new standalone `MainWindow` GUI regression test was not retained because
  the repository's isolated GUI subprocess can terminate silently during a
  second full-window construction; no product failure was established.

## Alignment with the original v1 handoff

The active retrieval scope remains one-gas VMR plus a fitted constant depth
offset. The 85–90% synthetic coverage results remain conditional and are not
general calibration. Emission remains deferred. The archive work is an
independent product-area increment and does not replace the required retrieval
misspecification validation.

## Next work

1. Add real-pRT fixed-parameter misspecification experiments, starting with
   temperature and gravity, and record bias, coverage, boundary status, and
   condition labels.
2. Keep instrument-response and correlated-noise experiments as separate
   validation axes.
3. Begin the real-file validation pass for x1dints, live MAST/TAP calls, and
   NIST JCAMP inputs where fixtures or safe network checks are available.
4. Revisit whether the conditional profile scan remains a diagnostic or needs a
   posterior sampler; do not present it as calibrated abundance uncertainty.
