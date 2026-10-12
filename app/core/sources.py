"""
ECAS source & citation registry.

Every external dataset, database, paper or tool the program uses is a :class:`Source` with a
stable key, a formatted reference, a licence note and a ``verified`` flag.  ``verified=True`` means
the identifier (DOI/URL/citation wording) was checked against the publisher's own page when the
entry was written; entries with ``verified=False`` are real references from the author's
knowledge/blueprint that should be double-checked before publication.  Nothing here is invented:
where a DOI is not known, it is left empty and the URL is given instead.

A :class:`CitationLog` records which sources were actually *used* in a session (with retrieval
date and the specific record/query), so reports can print an exact reference list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Source:
    key: str
    kind: str                       # database | paper | software | archive | mock
    title: str
    authors: str = ""
    year: str = ""
    venue: str = ""                 # journal / publisher
    doi: str = ""
    url: str = ""
    license: str = ""
    note: str = ""                  # acknowledgement wording or caveats the provider asks for
    verified: bool = False

    def apa(self) -> str:
        """Compact human-readable reference."""
        parts = [p for p in (self.authors, f"({self.year})" if self.year else "", self.title + ".",
                             self.venue + "." if self.venue else "") if p]
        link = f"https://doi.org/{self.doi}" if self.doi else self.url
        return " ".join(parts) + (f" {link}" if link else "")

    def bibtex(self) -> str:
        entry = "article" if self.kind == "paper" else "misc"
        lines = [f"@{entry}{{{self.key},", f"  title = {{{self.title}}},"]
        if self.authors:
            lines.append(f"  author = {{{self.authors}}},")
        if self.year:
            lines.append(f"  year = {{{self.year}}},")
        if self.venue:
            lines.append(f"  {'journal' if entry == 'article' else 'publisher'} = {{{self.venue}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        lines.append("}")
        return "\n".join(lines)


SOURCES: Dict[str, Source] = {s.key: s for s in (
    Source("nea_atmospheric_spectroscopy", "archive", "Atmospheric Spectroscopy Table",
           "NASA Exoplanet Archive", "", "NExScI-Caltech/IPAC", "10.26133/NEA36",
           "https://exoplanetarchive.ipac.caltech.edu/docs/atmospheres/atmospheres_home.html",
           "Public; acknowledgement requested",
           "The archive asks users to cite the DOI, state the access date/time and row count, and include: "
           "'This dataset or service is made available by the NASA Exoplanet Science Institute at IPAC, which "
           "is operated by the California Institute of Technology under contract with the National "
           "Aeronautics and Space Administration.'  The table is updated periodically.", verified=True),
    Source("nea_planetary_systems", "archive", "NASA Exoplanet Archive: Planetary Systems table (TAP service 'ps')",
           "NASA Exoplanet Archive", "", "NExScI-Caltech/IPAC", "",
           "https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html",
           "Public; acknowledgement requested",
           "Each row carries its own literature references (pl_refname, st_refname); ECAS stores them with "
           "every value.  Check the archive's 'Acknowledging the Archive' page for the DOI that applies to "
           "this table and the required acknowledgement text.", verified=True),
    Source("mast", "archive", "Mikulski Archive for Space Telescopes (MAST): TESS/JWST/HST observations",
           "Space Telescope Science Institute", "", "STScI", "", "https://mast.stsci.edu",
           "Public after the proprietary period",
           "Cite the specific program (proposal ID / PI) and the MAST DOI(s) given for the data set; follow "
           "the MAST acknowledgement guidance.", verified=True),
    Source("hitran2020", "paper", "The HITRAN2020 molecular spectroscopic database",
           "Gordon, I. E., Rothman, L. S., Hargreaves, R. J., et al.", "2022",
           "J. Quant. Spectrosc. Radiat. Transf. 277, 107949", "10.1016/j.jqsrt.2021.107949",
           "https://hitran.org", "HITRAN terms of use apply; HITRANonline needs an account/API key",
           "Downloads through HAPI require an API key and are rate-limited.", verified=True),
    Source("hapi", "software", "HITRAN Application Programming Interface (HAPI)",
           "Kochanov, R. V., Gordon, I. E., Rothman, L. S., et al.", "2016",
           "J. Quant. Spectrosc. Radiat. Transf. 177, 15-30", "10.1016/j.jqsrt.2016.03.005",
           "https://hitran.org/hapi/", "Free to use; please cite", "DOI not re-checked this session.",
           verified=False),
    Source("exomol", "database", "ExoMol molecular line lists",
           "Tennyson, J., Yurchenko, S. N., et al.", "", "ExoMol consortium", "",
           "https://exomol.com", "Public line lists; cite the specific molecule/list paper",
           "Raw .states/.trans imports are shape-only in ECAS unless a validated cross-section is supplied.",
           verified=True),
    Source("nist_webbook", "database", "NIST Chemistry WebBook, NIST Standard Reference Database Number 69",
           "Linstrom, P. J., and Mallard, W. G. (Eds.)", "", "National Institute of Standards and Technology",
           "10.18434/T4D303", "https://webbook.nist.gov/chemistry", "NIST open license",
           "Cite the data collection (e.g. 'Evaluated Infrared Reference Spectra', Coblentz Society) as well as "
           "the WebBook, and give the retrieval date.  Most WebBook gas-phase IR spectra are qualitative, not "
           "absolute absorption cross-sections.", verified=True),
    Source("nist_quantitative_ir", "database", "The NIST Quantitative Infrared Database",
           "Chu, P. M., Guenther, F. R., Rhoderick, G. C., and Lafferty, W. J.", "",
           "J. Res. Natl. Inst. Stand. Technol.", "", "https://www.nist.gov/node/797996", "NIST open license",
           "21 absorption-coefficient spectra of volatile organic compounds, JCAMP-DX format.", verified=True),
    Source("orca", "software", "The ORCA program system",
           "Neese, F.", "2012", "WIREs Comput. Mol. Sci. 2, 73-78", "10.1002/wcms.81",
           "https://www.faccts.de/orca/", "ORCA licence applies", "Listed in the project blueprint; DOI not "
           "re-checked this session.", verified=False),
    Source("mock", "mock", "MOCK / synthetic data generated by ECAS (not a measurement)", "ECAS", "", "", "", "",
           "n/a", "Representative numbers for testing the pipeline only.  Never cite as data.", verified=True),
)}


@dataclass
class CitationRecord:
    key: str
    detail: str = ""                 # e.g. planet name, file name, program id
    accessed: str = ""               # ISO timestamp (UTC)
    query: str = ""                  # the exact query/URL when the data came from a service
    rows: Optional[int] = None


class CitationLog:
    """Which sources this session actually used, with provenance."""

    def __init__(self) -> None:
        self._records: List[CitationRecord] = []

    def record(self, key: str, detail: str = "", query: str = "", rows: Optional[int] = None) -> CitationRecord:
        if key not in SOURCES:
            raise KeyError(f"unknown source key {key!r}")
        for r in self._records:
            if (r.key, r.detail) == (key, detail):
                return r
        rec = CitationRecord(key, detail, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), query, rows)
        self._records.append(rec)
        return rec

    @property
    def records(self) -> Tuple[CitationRecord, ...]:
        return tuple(self._records)

    def keys(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(r.key for r in self._records))

    def clear(self) -> None:
        self._records.clear()

    def references_markdown(self, extra: Iterable[CitationRecord] = ()) -> str:
        """A ready-to-paste reference list with access dates and provenance."""
        recs = list(self._records) + list(extra)
        if not recs:
            return "## References\n\n_No external data sources were used in this analysis._\n"
        out = ["## References", ""]
        for key in dict.fromkeys(r.key for r in recs):
            src = SOURCES[key]
            out.append(f"- {src.apa()}" + ("" if src.verified else "  *(identifier not re-verified)*"))
            if src.license:
                out.append(f"  - Licence/terms: {src.license}")
            for r in (x for x in recs if x.key == key):
                bits = [b for b in (r.detail, f"accessed {r.accessed}" if r.accessed else "",
                                    f"{r.rows} rows" if r.rows is not None else "",
                                    f"query: `{r.query}`" if r.query else "") if b]
                if bits:
                    out.append(f"  - {'; '.join(bits)}")
            if src.note:
                out.append(f"  - Note: {src.note}")
        return "\n".join(out) + "\n"

    def bibtex(self) -> str:
        return "\n\n".join(SOURCES[k].bibtex() for k in self.keys()) + "\n"
