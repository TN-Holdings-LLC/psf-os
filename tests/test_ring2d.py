"""psf_os.ring2d (W3 prototype): the ring sentinel's ledger and its verdict beside the ledger (report only)."""
from psf_os import ring2d as RG


def _row(date, dead_sites, n=20, dev="ibm_x"):
    a = [0.3] * n
    for k in dead_sites:
        a[k] = 0.05
    ring = list(range(100, 100 + n))
    return dict(date=date, device=dev, status="DEAD" if dead_sites else "CLEAN", ring=ring, median=0.3,
                dead=[ring[k] for k in dead_sites], a=a)


def test_persistent_counts_jobs_and_days_and_forms_paths():
    h = [_row("d1", [5, 6, 7]), _row("d1", [5, 6, 7, 8]), _row("d2", [5, 6, 7]), _row("d2", [12, 13, 14])]
    p = RG.persistent(h, min_jobs=3, min_days=2)["ibm_x"]
    assert p["qubits"] == [105, 106, 107] and p["paths"] == [[105, 106, 107]]
    assert p["jobs"] == 4 and p["days"] == 2
    assert RG.persistent(h, min_jobs=3, min_days=3)["ibm_x"]["qubits"] == []


def test_verdict_beside_ledger_sets_known_sites_aside_only():
    r = _row("d3", [5, 6, 7, 8])
    v = RG.verdict_beside_ledger(r, [105, 106, 107, 108])
    assert v["status_all"] == "DEAD" and v["status_new"] == "CLEAN" and v["new_dead"] == []
    r2 = _row("d3", [5, 6, 7, 8, 14, 15, 16])  # a new stretch is still found
    v2 = RG.verdict_beside_ledger(r2, [105, 106, 107, 108])
    assert v2["status_new"] == "DEAD" and v2["new_dead"] == [114, 115, 116]


def test_known_dead_counts_only_earlier_days():
    from psf_os import report2d as X
    led = {"ibm_x": {"105": ["d1", "d1", "d2"], "106": ["d1", "d2", "d3"], "107": ["d1", "d1", "d1"]}}
    assert X.known_dead(led, "ibm_x", "d3") == [105]          # 106 has only two jobs before d3; 107 one day
    assert X.known_dead(led, "ibm_x", "d4") == [105, 106]
    assert X.known_dead(led, "ibm_x", "d2") == []
    assert X.known_dead(led, "other", "d9") == []
