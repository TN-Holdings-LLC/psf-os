"""psf_os.entrance (2026-11-13): family by fixed tests, placement, controls, special hook, cost and score.
The regression against the declared jobs (same gate hash) is the notebook's notes/os_entrance_2026-11-13/
regress_entrance.py; the TOQB_DIR test below checks chain_of against the notebook's builder on its device records."""
import glob
import json
import os
import sys
import types

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy

from psf_os import chain as CH
from psf_os import device as Dv
from psf_os import entrance as E
from psf_os import mirror2d as M
from psf_os import strip as S


def _chain_circuit(n, steps=1):
    qc = QuantumCircuit(n)
    for k in range(1, n, 2):
        qc.x(k)
    for _ in range(steps):
        for p in (0, 1):
            for k in range(p, n - 1, 2):
                qc.rzz(0.3, k, k + 1)
                qc.sx(k)
    return qc


def _graph_circuit(n, edges):
    qc = QuantumCircuit(n)
    for a, b in edges:
        qc.h(a)
        qc.cz(a, b)
    return qc


def test_graph_tests_and_refusals():
    assert E.path_order({0, 1, 2, 3}, {(0, 1), (1, 2), (2, 3)}) == [0, 1, 2, 3]
    assert E.path_order({0, 1, 2}, {(0, 2), (1, 2)}) == [0, 2, 1]
    assert E.path_order({0, 1, 2}, {(0, 1), (1, 2), (0, 2)}) is None
    assert E.cycle_order({0, 1, 2, 3}, {(0, 1), (1, 2), (2, 3), (0, 3)}) == [0, 1, 2, 3]
    dev = Dv.fetch_fake("FakeFez")
    grid = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]       # 2 x 3 square grid
    with pytest.raises(E.Refused, match="4-cycles"):
        E.recognise(dev, [("g", _graph_circuit(6, grid))])
    with pytest.raises(E.Refused, match="degree 4"):
        E.recognise(dev, [("s", _graph_circuit(5, [(0, 1), (0, 2), (0, 3), (0, 4)]))])
    with pytest.raises(E.Refused, match="family"):
        E.recognise(dev, [("c", _chain_circuit(4))], family="cube")


def test_chain_is_hub_builds_placement_and_job(tmp_path):
    """A logical chain through the entrance = on_chain(chain_of) + the compiled mirror, as gpu/hub_build built it."""
    from psf_os import gatehash, user2d as U
    dev = Dv.fetch_fake("FakeFez")
    items = [("a_s1", _chain_circuit(12, 1)), ("a_s2", _chain_circuit(12, 2))]
    b = E.build(dev, items, str(tmp_path / "e"), shots=500)
    labels, how = CH.chain_of(dev, 12)
    g = b["groups"][0]
    assert g["family"] == "chain" and g["placement"]["labels"] == labels and b["special"] is None
    assert b["mirror"] == "compiled" and b["sentinel_fallback"] is None and b["twin"] is False
    # the same job by hand, the builders' way
    old = M.mirror
    with E.compiled_mirror(dev):
        U.build_batch(dev, [(t, E.on_labels(qc, dict(enumerate(labels)), dev.num_qubits)) for t, qc in items],
                      str(tmp_path / "h"), shots=500)
    assert M.mirror is old                                    # the patch is scoped
    h1 = gatehash.qpy_hash(str(tmp_path / "e" / "job" / "circuits.qpy"))["gate_sha256"]
    h2 = gatehash.qpy_hash(str(tmp_path / "h" / "job" / "circuits.qpy"))["gate_sha256"]
    assert h1 == h2
    c = b["cost"]["items"]["a_s2"]
    assert c["cz"] == 2 * 11 * 2 and c["logical_2q_layers"] == 4 and c["inflation"] == c["cz_layers"] / 4


def test_groups_with_different_graphs_are_placed_each_on_its_own():
    dev = Dv.fetch_fake("FakeFez")
    r = E.check(dev, [("n8", _chain_circuit(8)), ("n12", _chain_circuit(12))], shots=500)
    assert [g["n"] for g in r["groups"]] == [8, 12]
    assert all(g["family"] == "chain" for g in r["groups"])


def test_strip_family_and_port():
    dev = Dv.fetch_fake("FakeFez")
    phys, bonds = S.strip(dev.all_couplers(), 2, 10)
    assert len(phys) == 24 and len(bonds) == 24 and len(S.colour(bonds)) == 3     # home VS's 24-qubit strip (2 rows x 11 columns: c = 10)
    phys3, bonds3 = S.strip(dev.all_couplers(), 3, 15)
    assert (len(phys3), len(bonds3)) == (56, 61)                                  # notes/ohm_honeycomb table
    rec = E.recognise(dev, [("h", _graph_circuit(24, bonds))])
    assert rec["family"] == "strip" and rec["phys"] == phys


def test_labels_family_passes_heron_labels_unchanged():
    dev = Dv.fetch_fake("FakeFez")
    star = [(2, 3), (3, 4), (3, 16)]          # row qubit 3 with its bridge: degree 3, not a path
    assert all(e in dev.all_couplers() for e in star)
    rec = E.recognise(dev, [("t", _graph_circuit(20, star))])
    assert rec["family"] == "labels"
    m, how = E.placement(dev, rec)
    assert m is None and how["rule"] == "labels"


def test_ring_places_a_plain_cycle_or_refuses_with_the_reason():
    dev = Dv.fetch_fake("FakeFez")
    n = 12
    qc = _graph_circuit(n, [(k, (k + 1) % n) for k in range(n)])
    rec = E.recognise(dev, [("r", qc)])
    assert rec["family"] == "ring"
    try:
        m, how = E.placement(dev, rec)
    except E.Refused as e:
        assert "pendant" in str(e) or "ring of" in str(e)
        return
    cyc = [m[q] for q in rec["order"]]
    assert all(tuple(sorted((cyc[i], cyc[(i + 1) % n]))) in dev.all_couplers() for i in range(n))


def test_special_hook_with_a_fake_quirk(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    calls = {}

    def rank(dev_name, cands, ledger, before):
        calls.update(dev=dev_name, n=len(cands), ledger=ledger, before=before)
        return list(reversed(cands)), dict(fake=True)
    fake = types.SimpleNamespace(rank=rank)
    items = [("q", _chain_circuit(6))]
    r = E.check(dev, items, special="quirk_ledger", quirk=fake, ledger={"x": 1}, before="2026-11-13")
    sp = r["groups"][0]["special"]
    assert calls["dev"] == dev.name and calls["ledger"] == {"x": 1} and calls["n"] > 1
    assert sp["name"] == "quirk_ledger" and sp["chosen"] != sp["first_by_rule"] and sp["report"] == dict(fake=True)
    b = E.build(dev, items, str(tmp_path / "s"), special="quirk_ledger", quirk=fake, shots=300)
    chosen = sorted(b["groups"][0]["mapping"].values())
    rec = json.load(open(tmp_path / "s" / "runs" / "p2d_q.json"))
    assert sorted(rec["labels"]) == chosen                    # user2d kept the pinned copy
    from psf_os import lattice2d as L
    assert L.place.__name__ == "place" and L.place.__module__ == "psf_os.lattice2d"   # restored
    sys.modules.pop("psf_os.quirk", None)
    if not os.path.exists(os.path.join(os.path.dirname(E.__file__), "quirk.py")):
        with pytest.raises(E.Refused, match="not present"):
            E.check(dev, items, special="quirk_ledger")


def test_options_are_recorded_and_checked(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("o", _chain_circuit(6))]
    with pytest.raises(E.Refused, match="mirror"):
        E.build(dev, items, str(tmp_path / "x"), mirror="other")
    with pytest.raises(E.Refused, match="sentinel_fallback"):
        E.build(dev, items, str(tmp_path / "y"), sentinel_fallback="any")
    b = E.build(dev, items, str(tmp_path / "t"), twin=True, sentinel_fallback="same_qubits", shots=300,
                answers={"o": [0.5] * 6}, answers_out=str(tmp_path / "ans.json"))
    assert b["twin"] and b["sentinel_fallback"] == dict(asked="same_qubits", used=False)
    tw = json.load(open(b["twin_answers"]))
    ans = json.load(open(tmp_path / "ans.json"))
    labels = b["groups"][0]["placement"]["labels"][:6]
    assert set(ans["o"]) == {str(q) for q in labels} and set(tw["o"]) == set(ans["o"])
    assert all(abs(v) <= 1 + 1e-12 for v in tw["o"].values())
    with pytest.raises(E.Refused, match="exists"):
        E.build(dev, items, str(tmp_path / "t"))


def test_verdict_split_and_twin_line():
    e = np.array([0.8, -0.6, 0.4, 0.2])
    v = E.verdict(0.5 * e, e)
    assert abs(v["r"] - 0.5) < 1e-12 and abs(v["shrink"] - 0.25) < 1e-12 and abs(v["rest"]) < 1e-12
    assert v["reads"] == "READS" and abs(v["ratio"] - 0.25) < 1e-12
    v2 = E.verdict(0.25 * e, e)
    assert v2["reads"] == "DOES NOT READ" and v2["r"] < E.READ_LINE
    assert abs(E.READ_LINE - 0.2929) < 1e-4


def test_cost_and_boundary_hint(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    bpath = tmp_path / "b" / "boundary.json"
    bpath.parent.mkdir()
    json.dump(dict(rows=[dict(result="hub-x", cz_layers=48, verdict="READS, AGREES 3 of 40"),
                         dict(result="hub-x", cz_layers=72, verdict="DOES NOT READ, AGREES 0 of 40"),
                         dict(result="ohm-y", cz_layers=90, verdict="READS, DIFFERS"),
                         dict(result="p6", cz_layers=None, verdict="READS")]), open(bpath, "w"))
    r = E.check(dev, [("c", _chain_circuit(10, 2))], boundary=str(bpath), boundary_prefix="hub")
    h = r["cost"]["boundary_hint"]
    assert h["read_up_to_layers"] == 48 and h["first_not_read_layers"] == 72 and h["rows"] == 2
    assert r["cost"]["fits"] and r["cost"]["items"]["c"]["cz_per_qubit"] == 2 * 9 * 2 / 10


def test_cli_check_and_build(tmp_path, capsys):
    from psf_os import cli
    dev = Dv.fetch_fake("FakeFez")
    dev.save(str(tmp_path / "dev"))
    c1, c2 = _chain_circuit(8, 1), _chain_circuit(8, 2)
    c1.name, c2.name = "k1", "k2"
    with open(tmp_path / "c.qpy", "wb") as f:
        qpy.dump([c1, c2], f)
    assert cli.main(["entrance", "check", "--device", str(tmp_path / "dev"), "--circuit", str(tmp_path / "c.qpy")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["groups"][0]["family"] == "chain" and out["groups"][0]["tags"] == ["k1", "k2"]
    assert cli.main(["entrance", "build", "--device", str(tmp_path / "dev"), "--circuit", str(tmp_path / "c.qpy"),
                     "--out", str(tmp_path / "o"), "--shots", "300"]) == 0
    assert json.load(open(tmp_path / "o" / "build.json"))["special"] is None
    grid = _graph_circuit(4, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)])
    with open(tmp_path / "g.qpy", "wb") as f:
        qpy.dump([grid], f)
    assert cli.main(["entrance", "check", "--device", str(tmp_path / "dev"), "--circuit", str(tmp_path / "g.qpy")]) == 3


TOQB = os.environ.get("TOQB_DIR")


@pytest.mark.skipif(not TOQB, reason="needs TOQB_DIR (the notebook's device records and gpu/hub_build.py)")
def test_chain_of_equals_the_notebook_builder_on_every_record():
    sys.path.insert(0, os.path.join(TOQB, "gpu"))
    sys.path.insert(0, TOQB)
    import hub_build as HB
    devs = sorted(glob.glob(os.path.join(TOQB, "results", "*", "fetch_in", "*", "dev")))
    assert devs
    for d in devs:
        dev = Dv.Device.load(d)
        for n in (24, 40, 60, 91):
            def get(f):
                try:
                    return f(dev, n)
                except Exception as e:
                    return type(e).__name__
            assert get(CH.chain_of) == get(HB.chain_of), (d, n)
