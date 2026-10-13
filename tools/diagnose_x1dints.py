"""Report how the wavelength grids of a JWST x1dints file differ between integrations.

Usage:  python tools/diagnose_x1dints.py path/to/file_x1dints.fits

Prints one line per EXTRACT1D table, then the largest offset of any integration from the first, in
detector pixels. Send the output to the developer if ECAS still refuses the file.
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def main(path: str) -> None:
    with fits.open(path, memmap=False) as hdul:
        print(f"File: {Path(path).name}")
        ref = None
        spacing = None
        tables = 0
        integrations_in_file = 0
        largest = 0.0
        total = 0
        for hdu_index, hdu in enumerate(hdul):
            if str(hdu.name).upper() != "EXTRACT1D" or hdu.data is None:
                continue
            names = {n.upper(): n for n in hdu.columns.names}
            w = np.asarray(hdu.data[names["WAVELENGTH"]], dtype=float)
            if w.ndim == 1:
                w = w[None, :]
            tables += 1
            integrations_in_file += w.shape[0]
            if tables == 1:
                print(f"  first EXTRACT1D table (HDU {hdu_index}): {w.shape[1]} channels per integration")
            for row in w:
                finite = row[np.isfinite(row) & (row > 0)]
                if finite.size < 2:
                    continue
                if ref is None:
                    ref = finite
                    spacing = float(np.median(np.diff(finite)))
                    print(f"    first grid: {finite[0]:.6f} - {finite[-1]:.6f} µm, "
                          f"{finite.size} finite channels, pixel ≈ {spacing:.3e} µm")
                    continue
                total += 1
                if finite.size != ref.size:
                    print(f"    integration {total}: {finite.size} finite channels (first has {ref.size}) "
                          "-> different number of channels")
                    largest = float("inf")
                    continue
                shift = float(np.max(np.abs(finite - ref)) / spacing)
                largest = max(largest, shift)
        print(f"EXTRACT1D tables: {tables}, integrations in file: {integrations_in_file}")
        print(f"Integrations compared with the first: {total}")
        print(f"Largest offset: {largest:.4f} pixels "
              f"({'within' if largest <= 0.5 else 'OUTSIDE'} ECAS's 0.5-pixel tolerance)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
