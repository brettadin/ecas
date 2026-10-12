"""Sources & citations dialog: exactly what this session used, ready to paste into a paper."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
)

from ..core.sources import SOURCES, CitationLog
from .theme import muted, primary


class SourcesDialog(QDialog):
    def __init__(self, log: CitationLog, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sources & citations")
        self.resize(780, 560)
        self._log = log
        lay = QVBoxLayout(self)
        head = QLabel("References for this session")
        head.setProperty("heading", True)
        note = QLabel("Every dataset or database ECAS used is listed with its access date and the exact query. "
                      "Entries marked 'not re-verified' have identifiers that should be checked before publication. "
                      "Mock data is labelled and must never be cited as data.")
        note.setWordWrap(True)
        muted(note)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlainText(log.references_markdown())
        known = QPlainTextEdit()
        known.setReadOnly(True)
        known.setMaximumHeight(140)
        known.setPlainText("All sources ECAS knows about:\n" + "\n".join(
            f"  • {s.title}  ({s.doi or s.url})" + ("" if s.verified else "  [not re-verified]") for s in SOURCES.values()))
        row = QHBoxLayout()
        b_copy, b_md, b_bib, b_close = (QPushButton(t) for t in ("Copy", "Save .md…", "Save BibTeX…", "Close"))
        primary(b_close)
        b_copy.clicked.connect(lambda: QApplication.clipboard().setText(self._text.toPlainText()))
        b_md.clicked.connect(lambda: self._save("references.md", "Markdown (*.md)", self._text.toPlainText()))
        b_bib.clicked.connect(lambda: self._save("references.bib", "BibTeX (*.bib)", log.bibtex()))
        b_close.clicked.connect(self.accept)
        for b in (b_copy, b_md, b_bib):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(b_close)
        for w in (head, note, self._text, known):
            lay.addWidget(w)
        lay.addLayout(row)

    def _save(self, name: str, filt: str, text: str) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save", name, filt)
        if path:
            Path(path).write_text(text, encoding="utf-8")
