"""
Test isolation for the GUI tests.

Building many pyqtgraph windows in ONE process occasionally segfaults inside pyqtgraph's own
plot-menu construction (seen on the ~10th window; every first window in a process is fine, and
the application only ever creates one).  Rather than let that flake the suite, every
``test_gui_*`` test runs in its own fresh interpreter.  Set ``ECAS_GUI_INPROCESS=1`` to run them
in-process anyway (useful for debugging with a single test).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if not pyfuncitem.name.startswith("test_gui_") or os.environ.get("ECAS_GUI_CHILD") or \
            os.environ.get("ECAS_GUI_INPROCESS"):
        return None                                          # run normally (child process or opt-out)
    env = dict(os.environ, ECAS_GUI_CHILD="1", QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                           pyfuncitem.nodeid], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"GUI test failed in its isolated process:\n{proc.stdout[-3000:]}\n{proc.stderr[-1500:]}",
                    pytrace=False)
    return True                                              # handled: skip the in-process call
