# ECAS Transmission Retrieval Handoff v10

Updated 2026-10-12.

## Work completed this session

The real-pRT validation runner now supports fixed-parameter
misspecification experiments for temperature and gravity:

```powershell
& '.venv-1\\Scripts\\python.exe' tools/run_real_profile_validation.py `
  --seeds 10 --vmrs 1e-4 1e-3 `
  --synthetic-temperatures-k 1000 1200 1400 `
  --synthetic-gravities-cgs 1000 `
  --summary-only
```

The scan remains fixed at 1200 K and 1000 cgs. Synthetic models may vary the
temperature and gravity independently, and every output condition records the
synthetic values, scan values, and a misspecification flag. Forward models are
cached by injected VMR plus synthetic temperature and gravity.

## Verification

- Python compilation and CLI help completed successfully.
- Real-pRT smoke mismatch completed successfully: synthetic 1000 K, scan
  1200 K, gravity 1000 cgs, injected H₂O `1e-3`, one seed, 2.5 ppm noise,
  40 ppm offset.
- That diagnostic case had 0.875 dex absolute log-VMR bias, 0% interval
  coverage, 100% bounded intervals, and 0% boundary-limited cases. It is not
  enough to estimate a general failure rate or calibrate uncertainties.

## Scope and guardrails

Instrument response and correlated/systematic noise remain separate axes and
were not mixed into this experiment. The active retrieval remains conditional
one-gas VMR plus constant offset; emission remains deferred. Existing 85–90%
matched-model coverage must not be generalized to misspecified real data.

## Next work

1. Run a small temperature mismatch grid across multiple seeds and interior
   VMRs, then add gravity mismatches separately and summarize bias/coverage.
2. Add reference pressure/radius and stellar-radius mismatch only after the
   temperature/gravity results are recorded cleanly.
3. Continue the real-file validation track for x1dints, MAST/TAP, and NIST
   JCAMP inputs.
4. Keep instrument response and correlated-noise validation independent.
