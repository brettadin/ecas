import urllib.request
import urllib.parse

TESTS = {
    "NASA Exoplanet Archive": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        + urllib.parse.urlencode({
            "query": "select pl_name from pscomppars where pl_name like 'WASP-39%'",
            "format": "csv"}),
    "MAST (JWST and TESS data)": "https://mast.stsci.edu/api/v0/",
    "HITRAN": "https://hitran.org/",
    "SIMBAD": "https://simbad.cds.unistra.fr/simbad/",
}

for name, url in TESTS.items():
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            print(f"OK    {name} (HTTP {r.status})")
    except Exception as exc:
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")