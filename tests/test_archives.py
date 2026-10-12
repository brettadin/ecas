"""Archive client tests: network is injected, so these run offline and deterministically.
The fixtures use the archive's REAL column names (checked against its TAP schema) but invented
values for a made-up planet, 'TEST-1 b' - they test parsing, provenance and error handling only."""
import sys
import urllib.parse
from pathlib import Path

import pytest
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import archives
from app.core.archives import ArchiveError, PLANET_COLUMNS, fetch_planet, fetch_text, search_planet_names
from app.core.sources import CitationLog


def _csv(**over):
    row = {c: "" for c in PLANET_COLUMNS}
    row.update(pl_name="TEST-1 b", hostname="TEST-1", default_flag="1", pl_orbper="3.2", pl_tranmid="2459000.5",
              pl_trandur="2.8", pl_ratror="0.1", pl_ratdor="9.4", pl_imppar="0.3", pl_radj="0.95",
              pl_orbsmax="0.044", pl_eqt="1300", st_teff="5800", st_rad="1.05", st_mass="", pl_ntranspec="2",
              pl_refname='<a refstr=DOE_ET_AL__2024 href=https://example.org/paper target=ref>Doe et al. 2024</a>',
              st_refname="Roe et al. 2023", rowupdate="2026-01-01")
    row.update(over)
    return ",".join(PLANET_COLUMNS) + "\n" + ",".join(f'"{row[c]}"' if "," in str(row[c]) else str(row[c])
                                                         for c in PLANET_COLUMNS) + "\n"


def test_fetch_planet_parses_values_references_and_builds_a_documented_query():
    seen = []
    rec = fetch_planet("TEST-1 b", fetch=lambda u: seen.append(u) or _csv())
    q = urllib.parse.unquote_plus(seen[0])
    assert q.startswith(archives.TAP_URL) and "from ps" in q and "default_flag = 1" in q and "pl_name = 'TEST-1 b'" in q
    assert "format=csv" in seen[0] and "select pl_name,hostname" in q
    assert rec.get("pl_orbper") == 3.2 and rec.get("st_mass") is None and rec.name == "TEST-1 b"
    assert rec.refs["planet"] == "Doe et al. 2024 <https://example.org/paper>" and rec.refs["star"] == "Roe et al. 2023"
    sp, missing = rec.to_stellar_params()
    assert (sp.radius_rsun, sp.teff_k) == (1.05, 5800.0) and sp.mass_msun == 1.0 and missing == ["M★"]
    text = "\n".join(rec.summary_lines())
    assert "Doe et al. 2024" in text and "Published transmission spectra in the archive: 2" in text
    log = CitationLog(); rec.cite(log)
    assert "Doe et al. 2024" in log.records[0].detail and log.records[0].query == rec.url
    assert "TEST-1 b" in log.references_markdown()


def test_names_with_quotes_are_escaped_and_failures_are_explained():
    seen = []
    with pytest.raises(ArchiveError, match="no planet named"):
        fetch_planet("O'Neil b", fetch=lambda u: seen.append(u) or ",".join(PLANET_COLUMNS) + "\n")
    assert "O''Neil+b" in seen[0] or "O%27%27Neil" in seen[0]                       # ADQL quote doubling
    with pytest.raises(ArchiveError, match="rejected"):
        fetch_planet("X b", fetch=lambda u: "<?xml version='1.0'?><VOTABLE><INFO name='ERROR'>bad</INFO>")
    with pytest.raises(ArchiveError, match="enter a planet name"):
        fetch_planet("  ")

    def offline(url):
        raise ArchiveError("could not reach the archive (offline)")
    with pytest.raises(ArchiveError, match="could not reach"):
        fetch_planet("TEST-1 b", fetch=offline)


def test_name_search():
    names = search_planet_names("test-1", fetch=lambda u: "pl_name\nTEST-1 b\nTEST-1 c\n")
    assert names == ["TEST-1 b", "TEST-1 c"]
    with pytest.raises(ArchiveError):
        search_planet_names("a")


def test_disk_cache_serves_repeat_queries_without_the_network(tmp_path):
    src = tmp_path / "data.txt"; src.write_text("first")
    url = src.as_uri()
    cache = tmp_path / "cache"
    assert fetch_text(url, cache_dir=cache) == "first"
    src.write_text("second"); src.unlink()                                          # source gone: cache must answer
    assert fetch_text(url, cache_dir=cache) == "first"
    with pytest.raises(ArchiveError, match="could not reach"):
        fetch_text((tmp_path / "nope.txt").as_uri())


class _FakeTable(list):
    colnames = list(archives.MAST_COLUMNS)


class _FakeObservations:
    def __init__(self): self.calls = []
    def query_criteria(self, **kw):
        self.calls.append(kw)
        return _FakeTable([{c: f"{c}-v" for c in archives.MAST_COLUMNS} | {"obs_id": kw.get("obs_id", "jw01366-o001")}])
    def get_product_list(self, table):
        return [{"productFilename": n} for n in ("jw01366001001_04101_00001-seg001_nrs1_x1dints.fits",
                                                  "jw01366001001_04101_00001-seg001_nrs1_rateints.fits",
                                                  "jw01366001001_04101_00001_nrs1_whtlt.fits")]
    def download_products(self, products, download_dir, **_kwargs):
        return {"Local Path": [str(Path(download_dir) / p["productFilename"]) for p in products]}


def test_mast_wrappers_with_a_fake_service(tmp_path):
    fake = _FakeObservations()
    found = archives.search_mast("TEST-1", observations=fake)
    assert fake.calls[0] == {"obs_collection": "JWST", "target_name": "TEST-1"} and found[0]["obs_id"] == "jw01366-o001"
    prods = archives.list_spectral_products("jw01366-o001", observations=fake)
    assert [p["productFilename"].rsplit("_", 1)[-1] for p in prods] == ["x1dints.fits", "whtlt.fits"]   # rateints dropped
    paths = archives.download_products(prods[:1], tmp_path / "dl", observations=fake)
    assert paths[0].name.endswith("_x1dints.fits") and (tmp_path / "dl").is_dir()

    class Broken(_FakeObservations):
        def query_criteria(self, **kw): raise RuntimeError("server timeout")
    with pytest.raises(ArchiveError, match="server timeout"):
        archives.search_mast("TEST-1", observations=Broken())


def test_mast_tess_lightcurve_search_filter_and_download(tmp_path):
    class TessObservations:
        def __init__(self):
            self.calls = []
            self.download_calls = 0

        def query_object(self, target, radius):
            self.calls.append(("query_object", target, radius))
            return Table([
                {c: f"{c}-v" for c in archives.MAST_COLUMNS}
                | {"obs_id": "tess-sector-1", "obs_collection": "TESS",
                   "obsid": 42, "dataproduct_type": "timeseries"},
                {c: f"{c}-v" for c in archives.MAST_COLUMNS}
                | {"obs_id": "tess-no-lightcurve", "obs_collection": "TESS",
                   "obsid": 99, "dataproduct_type": "timeseries"},
            ])

        def query_criteria(self, **kw):
            self.calls.append(("query_criteria", kw))
            return _FakeTable([{c: f"{c}-v" for c in archives.MAST_COLUMNS} | {"obs_id": "tess-sector-1"}])

        def get_product_list(self, _table):
            return [
                {"obsID": 42, "productFilename": "tess-target_tpf.fits", "productSubGroupDescription": "TP"},
                {"obsID": 42, "productFilename": "tess-target_lc.fits", "productSubGroupDescription": "LC"},
            ]

        def download_products(self, products, download_dir, **_kwargs):
            self.download_calls += 1
            paths = []
            for product in products:
                path = Path(download_dir) / product["productFilename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fits light curve")
                paths.append(str(path))
            return {"Local Path": paths}

    obs = TessObservations()
    found = archives.search_mast_lightcurves("WASP-39", observations=obs)
    assert len(found) == 1 and found[0]["obs_id"] == "tess-sector-1"
    assert obs.calls[0] == ("query_object", "WASP-39", "0.01 deg")
    products = archives.list_lightcurve_products("tess-sector-1", observations=obs)
    assert [p["productFilename"] for p in products] == ["tess-target_lc.fits"]
    path = archives.download_mast_lightcurve("tess-sector-1", tmp_path, observations=obs)
    assert path.name == "tess-target_lc.fits" and path.read_bytes() == b"fits light curve"
    cached = archives.download_mast_lightcurve("tess-sector-1", tmp_path, observations=obs)
    assert cached == path and obs.download_calls == 1

    class NoPhotometry(TessObservations):
        def get_product_list(self, _table):
            return []

    with pytest.raises(ArchiveError, match="no TESS light-curve FITS"):
        archives.download_mast_lightcurve("empty", tmp_path, observations=NoPhotometry())


def test_mast_jwst_spectral_timeseries_search_and_download(tmp_path):
    class JwstObservations:
        def __init__(self):
            self.download_calls = 0

        def query_object(self, target, radius):
            assert target == "WASP-39" and radius == "0.01 deg"
            return Table([
                {"obsid": 10, "obs_id": "jw-observation", "obs_collection": "JWST",
                 "target_name": "WASP-39", "instrument_name": "NIRSpec", "proposal_id": "1366",
                 "t_min": 59000.0, "t_exptime": 120.0},
                {"obsid": 11, "obs_id": "other-observation", "obs_collection": "TESS",
                 "target_name": "WASP-39", "instrument_name": "Photometer", "proposal_id": "0",
                 "t_min": 59000.0, "t_exptime": 120.0},
            ])

        def get_product_list(self, candidates):
            assert len(candidates) == 1
            return Table([
                {"obsID": 10, "productFilename": "visit_x1dints.fits",
                 "dataURI": "mast:JWST/product/visit_x1dints.fits", "size": 2500000},
                {"obsID": 10, "productFilename": "visit_x1d.fits",
                 "dataURI": "mast:JWST/product/visit_x1d.fits", "size": 100000},
            ])

        def download_file(self, uri, local_path, verbose):
            self.download_calls += 1
            assert uri == "mast:JWST/product/visit_x1dints.fits" and verbose is False
            Path(local_path).write_bytes(b"jwst data")
            return "COMPLETE", None, None

    obs = JwstObservations()
    products = archives.search_mast_spectral_timeseries("WASP-39", observations=obs)
    assert len(products) == 1
    assert products[0]["instrument_name"] == "NIRSpec"
    assert products[0]["filename"] == "visit_x1dints.fits"
    path = archives.download_mast_spectral_timeseries(products[0], tmp_path, observations=obs)
    assert path.name == "visit_x1dints.fits" and path.read_bytes() == b"jwst data"
    cached = archives.download_mast_spectral_timeseries(products[0], tmp_path, observations=obs)
    assert cached == path and obs.download_calls == 1


def test_compare_with_archive_flags_real_disagreement_only():
    from app.core import mock_data
    from app.core.transit import analyze_lightcurve
    lc, truth = mock_data.make_mock_lightcurve()
    res = analyze_lightcurve(lc)
    same = fetch_planet("TEST-1 b", fetch=lambda u: _csv(pl_orbper="3.2", pl_ratror="0.1", pl_ratdor="9.0",
                                                          pl_imppar="0.3", pl_trandur=f"{truth.duration_days * 24:.3f}"))
    lines = archives.compare_with_archive(res, same)
    assert sum("consistent" in l for l in lines) >= 3 and "Doe et al. 2024" in lines[-1]
    off = fetch_planet("TEST-1 b", fetch=lambda u: _csv(pl_orbper="3.5", pl_ratror="0.15", pl_ratdor="9.0",
                                                         pl_imppar="", pl_trandur=""))
    lines = archives.compare_with_archive(res, off)
    assert "differs" in next(l for l in lines if l.startswith("Period")) and "differs" in next(l for l in lines if l.startswith("Rp/R"))
    assert "archive –" in next(l for l in lines if l.startswith("Impact"))          # missing archive value handled
