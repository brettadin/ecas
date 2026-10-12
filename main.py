"""
ECAS entry point.

    python main.py                      # launch the dashboard with the bundled mock gases
    python main.py --config my.json     # use your own gas registry / ORCA tracks
    python main.py --check              # headless: parse + build engine, print summary, exit
    python main.py --points 16384       # finer spectral grid (default from the JSON file)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from app.core.config import ConfigError, ConfigManager
from app.core.engine import Unit, build_engine


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Exoplanetary Characterization & Astrochemistry Suite")
    p.add_argument("--config", help="path to a gases JSON file (default: app/data/gases.json)")
    p.add_argument("--points", type=int, help="override grid.n_points")
    p.add_argument("--opengl", action="store_true", help="enable pyqtgraph OpenGL (needs PyOpenGL)")
    p.add_argument("--no-3d", action="store_true", help="disable the PyVista 3-D viewport")
    p.add_argument("--check", action="store_true", help="headless self-check; no GUI is created")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cfg = ConfigManager.load(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.points:
        cfg = replace(cfg, grid=replace(cfg.grid, n_points=args.points))

    engine, report = build_engine(cfg)
    notes = [f"{gid} skipped ({msg})" for gid, msg in report.errors.items()]

    if args.check:
        m = engine.default_multipliers()
        print(f"gases loaded : {', '.join(report.loaded)}")
        print(f"flattened    : {engine.n_lines} lines, grid N = {engine.grid.n_points}")
        for unit in Unit:
            t = engine.transmittance(unit, m)
            print(f"{unit.value:<11}: T in [{t.min():.4f}, {t.max():.4f}]")
        for gid, msg in report.errors.items():
            print(f"skipped {gid}: {msg}")
        return 0

    import pyqtgraph as pg                       # GUI imports only when actually needed
    from PyQt6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow

    from app.ui.theme import apply_theme, configure_plots

    configure_plots(use_opengl=bool(args.opengl))
    app = QApplication(sys.argv[:1])
    apply_theme(app)
    window = MainWindow(engine, cfg.physics, notes, enable_3d=not args.no_3d)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
