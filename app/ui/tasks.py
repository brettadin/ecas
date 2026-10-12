"""Run a function on a worker thread and get the result back on the UI thread."""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class _Worker(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn())
        except Exception as exc:                          # never let a worker die silently
            self.failed.emit(str(exc) if str(exc) else type(exc).__name__)


def run_async(owner: QObject, fn: Callable[[], Any], on_done: Callable[[Any], None],
              on_error: Callable[[str], None]) -> QThread:
    """Start ``fn`` off the UI thread; callbacks run on the UI thread.  ``owner`` keeps it alive."""
    thread = QThread(owner)
    worker = _Worker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.done.connect(on_done)
    worker.failed.connect(on_error)
    worker.done.connect(thread.quit)
    worker.failed.connect(thread.quit)
    tasks = owner.__dict__.setdefault("_ecas_tasks", [])
    tasks.append((thread, worker))

    def cleanup() -> None:
        if (thread, worker) in tasks:
            tasks.remove((thread, worker))
        worker.deleteLater()
        thread.deleteLater()

    thread.finished.connect(cleanup)
    thread.start()
    return thread
