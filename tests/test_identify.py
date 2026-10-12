"""Tests for the citation registry, reference library and multi-molecule identification."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import mock_data
from app.core.config import ConfigManager
from app.core.engine import Unit, build_engine
from app.core.fitting import ObservedSpectrum, find_features
from app.core.identify import IdentifyError, decompose, rank_references
from app.core.library import (LibraryError, ReferenceLibrary, ReferenceSpectrum,
                              fetch_nist_jcamp, hapi_cross_section_reference,
                              import_exomol_transitions, import_jcamp, import_nist_jcamp,
                              nist_webbook_url, reference_from_arrays)
from app.core.physics import lorentzian
from app.core.sources import SOURCES, CitationLog

TRUTH = {"H2O": 3.0, "CO2": 2.0, "CH4": 0.8, "O3": 1.5, "N2O": 1.0}


@pytest.fixture
def engine():
    return build_engine(ConfigManager.load())[0]


def _so2_reference(engine) -> ReferenceSpectrum:
    """SO2 built with exactly the formula the mock observation injects (unit multiplier)."""
    tbl = mock_data.mock_peak_table("SO2")
    nu = np.linspace(400, 4200, 6000)
    gamma = engine.conditions.envelope_hwhm_cm1 + 0.1 * engine.conditions.pressure_atm
    k = engine.tau_scale * (lorentzian(nu[:, None], tbl.wavenumber_cm1[None, :], gamma) @ tbl.intensity_km_mol)
    return reference_from_arrays("SO2", "Sulfur Dioxide", "SO2", nu, k, "mock", False, "tau_per_multiplier")


def _bands(ref_id, centres, weights, width=6.0, name=None):
    nu = np.linspace(400, 4200, 6000)
    k = sum(w * np.exp(-0.5 * ((nu - c) / width) ** 2) for c, w in zip(centres, weights))
    return reference_from_arrays(ref_id, name or ref_id, ref_id, nu, k, "mock")


# ------------------------------------------------------------------ citations
def test_every_source_is_formattable_and_honest_about_verification():
    for src in SOURCES.values():
        assert src.apa() and src.bibtex().startswith("@")
    assert SOURCES["nea_atmospheric_spectroscopy"].doi == "10.26133/NEA36" and SOURCES["nea_atmospheric_spectroscopy"].verified
    assert SOURCES["nist_webbook"].doi == "10.18434/T4D303"
    assert not SOURCES["mock"].verified is False and "not a measurement" in SOURCES["mock"].title
    assert SOURCES["hapi"].verified is False                       # unverified entries must say so


def test_citation_log_records_provenance_once():
    log = CitationLog()
    log.record("nea_planetary_systems", "WASP-39 b", query="select ... from ps", rows=1)
    log.record("nea_planetary_systems", "WASP-39 b")               # duplicate: ignored
    log.record("mock", "mock observation")
    assert len(log.records) == 2 and log.keys() == ("nea_planetary_systems", "mock")
    md = log.references_markdown()
    assert "## References" in md and "WASP-39 b" in md and "accessed" in md and "select ... from ps" in md
    assert "@misc{mock" in log.bibtex()
    with pytest.raises(KeyError):
        log.record("not-a-source")
    assert "No external data" in CitationLog().references_markdown()


# --------------------------------------------------------------------- library
def _jcamp(yunits, y, xunits="1/CM", extra="", x_range=(500, 4000)):
    x = np.linspace(x_range[0], x_range[1], len(y))
    lines = [f"##TITLE=Test gas", "##JCAMP-DX=4.24", "##DATA TYPE=INFRARED SPECTRUM", "##ORIGIN=NIST Mass Spec Data Center",
             "##OWNER=NIST", "##CAS REGISTRY NO=7446-09-5", "##MOLFORM=O2 S", "##STATE=GAS",
             f"##XUNITS={xunits}", f"##YUNITS={yunits}", "##XFACTOR=1.0", "##YFACTOR=1.0",
             f"##FIRSTX={x[0]}", f"##LASTX={x[-1]}", f"##NPOINTS={len(y)}", f"##DELTAX={x[1]-x[0]}", extra,
             "##XYDATA=(X++(Y..Y))"]
    for i in range(0, len(y), 5):
        lines.append(f"{x[i]:.4f} " + " ".join(f"{v:.5f}" for v in y[i:i + 5]))
    return "\n".join(lines) + "\n##END=\n"


def test_import_jcamp_transmittance_and_quantitative(tmp_path):
    nu = np.linspace(500, 4000, 400)
    t = 1 - 0.6 * np.exp(-0.5 * ((nu - 1362) / 12) ** 2)
    p = tmp_path / "so2.jdx"; p.write_text(_jcamp("TRANSMITTANCE", t))
    ref = import_jcamp(p)
    assert ref.name == "Test gas" and ref.formula == "O2S" and ref.meta["cas"] == "7446-09-5"
    assert not ref.quantitative and abs(ref.nu_cm1[np.argmax(ref.k)] - 1362) < 15
    assert ref.ref_id == "cas:7446-09-5" and ref.source_key == "nist_webbook"
    assert ref.k.max() == pytest.approx(-np.log(0.4), rel=0.05)           # -ln T at the band centre

    q = tmp_path / "xs.jdx"; q.write_text(_jcamp("ABSORPTION COEFFICIENT (1/(ppm*m))", 1 - t))
    assert import_jcamp(q).quantitative

    um = tmp_path / "um.jdx"; um.write_text(_jcamp("ABSORBANCE", 1 - t, xunits="MICROMETERS", x_range=(2.5, 25.0)))
    r = import_jcamp(um)
    assert r.coverage_cm1 == pytest.approx((400.0, 4000.0), rel=1e-3)           # 2.5-25 um -> 4000-400 cm-1

    bad = tmp_path / "bad.jdx"; bad.write_text(_jcamp("REFLECTANCE", t))
    with pytest.raises(LibraryError):
        import_jcamp(bad)
    with pytest.raises(LibraryError):
        import_jcamp(tmp_path / "missing.jdx")


def test_nist_webbook_lookup_is_injectable_and_preserves_provenance():
    payload = _jcamp("TRANSMITTANCE", np.linspace(1.0, 0.4, 16))
    seen = []

    def fake_fetch(url):
        seen.append(url)
        return payload

    url = nist_webbook_url("C7446095", index=2)
    assert "JCAMP=C7446095" in url and "Index=2" in url and "IR-SPEC" in url
    assert "Type=UVVis" in nist_webbook_url("C287923", spectrum="uvvis")
    fetched_url, text = fetch_nist_jcamp("7446-09-5", 2, fetch=fake_fetch)
    assert fetched_url == seen[0] and text == payload
    ref = import_nist_jcamp("7446-09-5", 2, fetch=fake_fetch)
    assert ref.source_key == "nist_webbook" and ref.ref_id == "nist:7446-09-5:2"
    assert ref.meta["nist_index"] == 2 and ref.meta["query"] == seen[1]
    with pytest.raises(LibraryError):
        fetch_nist_jcamp("not-a-cas", fetch=fake_fetch)
    with pytest.raises(LibraryError):
        fetch_nist_jcamp("7446095", fetch=lambda _: "not JCAMP")


def test_nist_uvvis_jcamp_is_imported_as_a_shape(tmp_path):
    x = np.linspace(250.0, 900.0, 32)
    y = 1.0 + 2.0 * np.exp(-0.5 * ((x - 520.0) / 40.0) ** 2)
    payload = _jcamp("LOG EPSILON", y, xunits="NM", x_range=(250.0, 900.0))
    ref = import_nist_jcamp("C287923", spectrum="uvvis", fetch=lambda _: payload)
    assert ref.meta["spectrum"] == "uvvis" and ref.y_kind == "uv_log_epsilon"
    assert not ref.quantitative and ref.coverage_cm1 == pytest.approx((1e7 / 900.0, 1e7 / 250.0), rel=1e-3)


def test_hapi_adapter_marks_sampled_cross_section_quantitative():
    nu = np.linspace(1000.0, 1100.0, 16)
    xs = np.exp(-0.5 * ((nu - 1050.0) / 8.0) ** 2) * 1e-21
    ref = hapi_cross_section_reference("CO2", nu, xs, 800.0, pressure_atm=0.1)
    assert ref.source_key == "hapi" and ref.quantitative and ref.y_kind == "cross_section"
    assert ref.meta["units"] == "cm2 molecule-1" and ref.meta["temperature_k"] == 800.0
    with pytest.raises(LibraryError):
        hapi_cross_section_reference("CO2", nu[:-1], xs, 800.0)


def test_exomol_transition_import_is_shape_only_and_supports_compressed_files(tmp_path):
    states = tmp_path / "co.states"
    states.write_text("\n".join(f"{i} {i * 100.0:.3f} 1" for i in range(1, 18)) + "\n")
    transitions = tmp_path / "co.trans.bz2"
    import bz2
    transitions.write_bytes(bz2.compress("\n".join(f"{i + 1} {i} 1e-3" for i in range(1, 10)).encode()))
    ref = import_exomol_transitions(transitions, states, "CO", temperature_k=1200.0)
    assert ref.source_key == "exomol" and not ref.quantitative and ref.y_kind == "line_strength"
    assert ref.nu_cm1.size == 9 and ref.coverage_cm1 == pytest.approx((100.0, 100.0))
    assert "no line profile" in ref.meta["assumptions"]


def test_library_roundtrip_and_engine_references_match_sliders(tmp_path, engine):
    lib = ReferenceLibrary.from_engine(engine)
    assert set(lib.ids()) == set(engine.gas_index) and all(r.y_kind == "tau_per_multiplier" for r in lib)
    m = engine.default_multipliers()
    nu = np.linspace(500, 4000, 500)
    tau_from_refs = sum(m[engine.gas_index[r.ref_id]] * r.on(nu) for r in lib)
    np.testing.assert_allclose(tau_from_refs, engine.tau_on(nu) @ m, rtol=0.02, atol=2e-3)   # same physics
    lib.add(_so2_reference(engine))
    lib.save(tmp_path)
    back = ReferenceLibrary.load(tmp_path)
    assert back.ids() == lib.ids() and back.get("SO2").source_key == "mock"
    np.testing.assert_allclose(back.get("H2O").k, lib.get("H2O").k)
    assert ReferenceLibrary.load(tmp_path / "nothing-here").ids() == []


# ---------------------------------------------------------------- identification
@pytest.mark.parametrize("unit", list(Unit))
def test_decompose_finds_unmodelled_molecule_and_rejects_decoy(engine, unit):
    obs = mock_data.make_mock_observation(engine, unit, TRUTH, ("SO2", 3.0), noise=0.003)
    lib = ReferenceLibrary.from_engine(engine)
    lib.add(_so2_reference(engine))
    lib.add(_bands("NH3", [950, 1627, 3337], [1.0, 0.4, 0.3]))             # decoy: not in the data
    res = decompose(obs, lib)
    by = {c.ref_id: c for c in res.components}
    assert by["SO2"].detected and by["SO2"].amplitude == pytest.approx(3.0, rel=0.1) and by["SO2"].significance > 10
    for gas, truth in TRUTH.items():
        assert by[gas].detected and by[gas].amplitude == pytest.approx(truth, rel=0.15), gas
    assert not by["NH3"].detected and by["NH3"].significance < 3               # decoy is not claimed
    assert res.reduced_chi2 < 2.5
    assert any("RELATIVE" in w for w in res.warnings)                          # honesty about units


def test_rank_references_screens_a_large_library_before_mixture_fit():
    target = _bands("TARGET", [1150, 2300, 3410], [1.0, 0.6, 0.35], width=5.0)
    decoy = _bands("DECOY", [800, 1750, 2900], [1.0, 0.8, 0.5], width=5.0)
    nu = np.linspace(500, 4000, 1200)
    t = np.exp(-1.7 * target.on(nu))
    obs = ObservedSpectrum(nu, t, Unit.WAVENUMBER, np.full(nu.size, 0.01))
    ranked = rank_references(obs, ReferenceLibrary([decoy, target]))
    assert [r.ref_id for r in ranked] == ["TARGET", "DECOY"]
    assert ranked[0].score > 10 and ranked[0].delta_chi2 == pytest.approx(ranked[0].score ** 2)
    assert ranked[0].n_points == nu.size


def test_decompose_uses_observation_reference_overlap_and_explains_no_overlap():
    nu = np.linspace(10000.0, 19500.0, 240)
    k = np.exp(-0.5 * ((nu - 15000.0) / 500.0) ** 2) + 0.2
    uv = reference_from_arrays("UV", "UV target", "X", nu, k, "nist_webbook")
    obs = ObservedSpectrum(nu, np.exp(-0.8 * k), Unit.WAVENUMBER)
    result = decompose(obs, ReferenceLibrary([uv]))
    assert result.window_cm1[0] >= 10000.0 and result.components[0].detected

    ir = ReferenceLibrary([_bands("IR", [1000.0], [1.0])])
    with pytest.raises(IdentifyError, match="do not overlap"):
        decompose(obs, ir)


def test_missing_molecule_leaves_a_poor_fit_which_the_attribution_calls_unexplained(engine):
    obs = mock_data.make_mock_observation(engine, Unit.WAVENUMBER, TRUTH, ("SO2", 3.0), noise=0.003)
    feats = find_features(obs, min_snr=5.0)
    library = ReferenceLibrary.from_engine(engine)
    library.remove("SO2")
    res = decompose(obs, library, features=feats)    # SO2 NOT in the library
    assert res.reduced_chi2 > 5
    so2_feat = [a for a in res.attributions if abs(a.nu_peak_cm1 - 1362) < 15]
    assert so2_feat and so2_feat[0].unexplained > 0.3                          # nothing explains the 1362 band


def test_overlapping_molecules_are_separated_flagged_and_attributed():
    nu = np.linspace(900, 1100, 800)
    a = _bands("A", [1000.0], [1.0], width=6.0)
    b = _bands("B", [1002.0], [1.0], width=6.0)
    tau = 1.0 * a.on(nu) + 0.5 * b.on(nu)
    rng = np.random.default_rng(4)
    obs = ObservedSpectrum(nu, np.exp(-tau) + rng.normal(0, 0.002, nu.size), Unit.WAVENUMBER)
    lib = ReferenceLibrary([a, b])
    feat = find_features(obs, min_snr=5.0)
    res = decompose(obs, lib, features=feat)
    by = {c.ref_id: c for c in res.components}
    for rid, truth in (("A", 1.0), ("B", 0.5)):
        assert abs(by[rid].amplitude - truth) < 3 * by[rid].uncertainty + 0.05      # unbiased within its own error
    assert res.similarity[0, 1] > 0.8 and res.amp_correlation[0, 1] < -0.5          # degenerate pair is visible
    assert any("overlap strongly" in w for w in res.warnings)
    assert feat and res.attributions[0].overlapped and set(res.attributions[0].shares) == {"A", "B"}
    assert res.attributions[0].shares["A"] > res.attributions[0].shares["B"]


def test_decompose_input_validation(engine):
    obs = mock_data.make_mock_observation(engine, Unit.WAVENUMBER, TRUTH, None)
    with pytest.raises(IdentifyError):
        decompose(obs, ReferenceLibrary())
    far = ObservedSpectrum(np.linspace(8000, 9000, 50), np.ones(50), Unit.WAVENUMBER)
    with pytest.raises(IdentifyError, match="do not overlap"):
        decompose(far, ReferenceLibrary.from_engine(engine))
    given = ObservedSpectrum(obs.x, obs.y, obs.unit, np.full(obs.x.size, 0.003))   # file-supplied sigma path
    r = decompose(given, ReferenceLibrary.from_engine(engine))
    assert not r.sigma_estimated and r.reduced_chi2 < 2
