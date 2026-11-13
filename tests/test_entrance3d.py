"""psf_os.entrance3d and the entrance's 3D family (2026-11-13g): recognition of 4 x M x N lattices, the generic fold
path checked by statevector (fSim and RZZ, permuted axes, a 4 x M square grid), the kicked-Ising path equal to
grid3d.prepare, the job, refusals; and the chain candidates of the special condition (quirk hook) with fakes.
The declared 3D jobs are checked record by record in the notebook (notes/os_entrance_2026-11-13/regress_entrance.py)."""
import json
import types

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from psf_os import chain as CH
from psf_os import device as Dv
from psf_os import entrance as E
from psf_os import entrance3d as E3
from psf_os import grid3d as G


def fsim(t, p):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * p)]])


def _lattice_circuit(dims, gate="mixed", steps=2, seed=1):
    import itertools
    n = int(np.prod(dims))
    idx = {s: i for i, s in enumerate(itertools.product(*(range(d) for d in dims)))}
    edges = []
    for s, i in idx.items():
        for ax in range(3):
            t = list(s)
            t[ax] += 1
            if tuple(t) in idx:
                edges.append((i, idx[tuple(t)]))
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(rng.uniform(0, 3), q)
    for st in range(steps):
        for a, b in edges:
            if gate == "fsim" or (gate == "mixed" and (a + b + st) % 2):
                qc.append(UnitaryGate(fsim(0.7, 0.4), label="fsim"), [a, b])
            else:
                qc.rzz(0.37, a, b)
        for q in range(n):
            qc.rx(0.3, q)
    return qc


def _fidelity(dev, qc):
    """|<placed | logical embedded>|^2 with the bridges back at |0>, by statevector on the touched qubits."""
    nodes, edges = E.interaction([("x", qc)])
    r3 = E3.recognise(nodes, edges)
    pq, placed, used, info = E3.physical_generic(dev, qc, r3["M"], r3["N"], r3["site_of"])
    m = {u: i for i, u in enumerate(used)}
    small = QuantumCircuit(len(used))
    for ins in pq.data:
        small.append(ins.operation, [m[pq.find_bit(q).index] for q in ins.qubits])
    sv, lv = Statevector(small).data, Statevector(qc).data
    perm = [m[placed[r3["site_of"][k]]] for k in range(qc.num_qubits)]
    big = np.zeros(2 ** len(used), complex)
    for i, amp in enumerate(lv):
        j = 0
        for k in range(qc.num_qubits):
            if (i >> k) & 1:
                j |= 1 << perm[k]
        big[j] = amp
    return abs(np.vdot(big, sv)) ** 2, info


def test_recognise_lattices_and_axes():
    nodes, edges = E.interaction([("x", _lattice_circuit((4, 2, 2), steps=1))])
    r = E3.recognise(nodes, edges)
    assert (r["M"], r["N"]) == (2, 2) and all(r["site_of"][k] == k for k in r["site_of"])
    nodes, edges = E.interaction([("x", _lattice_circuit((2, 4, 2), steps=1))])
    r = E3.recognise(nodes, edges)
    assert r["perm"][0] == 1 and sorted(r["site_of"].values()) == list(range(16))
    nodes, edges = E.interaction([("x", _lattice_circuit((3, 4, 1), steps=1))])     # a 3 x 4 square grid
    r = E3.recognise(nodes, edges)
    assert (r["M"], r["N"]) == (3, 1)
    nodes, edges = E.interaction([("x", _lattice_circuit((3, 3, 1), steps=1))])
    assert E3.recognise(nodes, edges) is None


@pytest.mark.parametrize("dims,gate", [((4, 2, 2), "fsim"), ((4, 2, 2), "rzz"), ((4, 2, 2), "mixed"),
                                       ((2, 2, 4), "mixed"), ((4, 3, 1), "fsim")])
def test_generic_fold_equals_the_logical_circuit(dims, gate):
    dev = Dv.fetch_fake("FakeFez")
    f, info = _fidelity(dev, _lattice_circuit(dims, gate))
    assert f > 1 - 1e-9, (dims, gate, f)
    assert info["swaps"] > 0
    if gate == "rzz":
        assert info["bridge_moves"] == 0          # diagonal gates go through the bridge by CX copies
    else:
        assert info["bridge_moves"] > 0


def test_kicked_ising_goes_through_grid3d_unchanged():
    from psf_os import gatehash  # noqa: F401
    from psf_os.compile import RuleCompiler
    dev = Dv.fetch_fake("FakeFez")
    rc = RuleCompiler(dev)
    qc = G.logical(4, 2, 2, 2, 0.8)
    nodes, edges = E.interaction([("k", qc)])
    r3 = E3.recognise(nodes, edges)
    assert E3.kicked_ising(qc, 2, 2, r3["site_of"]) == (2, 0.8)
    for fold, merge in (("v3.1", True), ("v3.2", "carry")):
        a = E3.prepare(dev, qc, r3, fold=fold, rc=rc)
        b = G.prepare(dev, 4, 2, 2, 2, 0.8, layout="fold", rc=rc, merge=merge)
        assert a["path"] == "kicked_ising"
        assert [(i.operation.name, tuple(a["cu"].find_bit(q).index for q in i.qubits)) for i in a["cu"].data] == \
               [(i.operation.name, tuple(b["cu"].find_bit(q).index for q in i.qubits)) for i in b["cu"].data]
    line = E3.prepare(dev, qc, r3, fold="line", rc=rc)
    assert line["where"]["layout"].startswith("line")
    with pytest.raises(E3.Refused3D, match="line"):
        E3.prepare(dev, _lattice_circuit((4, 2, 2), "fsim", steps=1), r3, fold="line", rc=rc)


def test_entrance_builds_a_generic_3d_job(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("f1", _lattice_circuit((4, 2, 2), "fsim", steps=1)), ("f2", _lattice_circuit((4, 2, 2), "fsim", 2))]
    r = E.check(dev, items, shots=300)
    g = r["groups"][0]
    assert g["family"] == "grid3d" and g["placement"]["paths"] == ["generic"]
    c = r["cost"]["items"]["f2"]
    assert c["swaps"] > 0 and c["bridge_moves"] > 0 and c["inflation"] > 1 and c["qubits"] == 16
    b = E.build(dev, items, str(tmp_path / "j"), shots=300, answers={"f1": [0.1] * 16, "f2": [0.2] * 16},
                answers_out=str(tmp_path / "a.json"))
    rec = json.load(open(tmp_path / "j" / "runs" / "p2d_f1.json"))
    assert rec["labels"] == list(range(16)) and rec["grid3d"]["generic"] and rec["entrance_path"] == "generic"
    assert b["circuits"] > 6 and set(json.load(open(tmp_path / "a.json"))["f1"]) == {str(k) for k in range(16)}
    with pytest.raises(E.Refused, match="twin"):
        E.check(dev, items, twin=True)
    with pytest.raises(E.Refused, match="3D"):
        E.check(dev, items, special="quirk_ledger", quirk=types.SimpleNamespace(rank=None))


def test_too_large_and_other_grids_are_refused():
    dev = Dv.fetch_fake("FakeFez")
    with pytest.raises(E.Refused, match="M <= 8 rows"):
        E.recognise(dev, [("b", _lattice_circuit((4, 5, 5), "rzz", steps=1))])
    with pytest.raises(E.Refused, match="4-cycles"):
        E.recognise(dev, [("s", _lattice_circuit((3, 3, 1), "rzz", steps=1))])


def test_chain_special_ranks_the_chain_rule_starts(tmp_path):
    from psf_os import lattice2d as L
    dev = Dv.fetch_fake("FakeFez")
    n = 8
    labels, _ = CH.chain_of(dev, n)
    _, _, m, _ = L.place(dev, [tuple(sorted(e)) for e in zip(labels, labels[1:])], labels)
    default = [m[q] for q in labels]
    other = [20 * 2 + c for c in range(n)]                # a chain on row 2
    fake_chain = types.SimpleNamespace(candidates=lambda d, k: [dict(name="start A", qubits=default),
                                                                dict(name="start B", qubits=other)])
    fake_quirk = types.SimpleNamespace(rank=lambda name, cs, ledger, before: (list(reversed(cs)), dict(fake=1)))
    qc = QuantumCircuit(n)
    for k in range(n - 1):
        qc.cz(k, k + 1)
    r = E.check(dev, [("c", qc)], special="quirk_ledger", quirk=fake_quirk, quirk_chain=fake_chain)
    sp = r["groups"][0]["special"]
    assert sp["kind"] == "chain starts" and sp["chosen"] == "start B" and sp["first_by_rule"] == "start A"
    assert sp["default_is_rule_choice"] is True
    assert [r["groups"][0]["mapping"][str(k)] for k in range(n)] == other
    b = E.build(dev, [("c", qc)], str(tmp_path / "s"), special="quirk_ledger", quirk=fake_quirk,
                quirk_chain=fake_chain, shots=300)
    rec = json.load(open(tmp_path / "s" / "runs" / "p2d_c.json"))
    assert sorted(rec["labels"]) == sorted(other) and b["special"] == "quirk_ledger"
    # no special: unchanged (the chain rule's own placement)
    r0 = E.check(dev, [("c", qc)])
    assert r0["groups"][0]["special"] is None and r0["groups"][0]["placement"]["labels"] == labels
