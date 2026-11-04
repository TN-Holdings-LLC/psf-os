"""psf_os.user2d: the user's 2D circuit in, a job with floor, mirror and sentinel out, the answer card back."""
import json
import os

import pytest

from psf_os import cli
from psf_os import device as Dv
from psf_os import ising2d as I
from psf_os import mirror2d as M
from psf_os import sim
from psf_os import user2d as U
from psf_os.compile import RuleCompiler


def test_p6_as_a_user_circuit_lands_like_p6(tmp_path):
    dev = Dv.fetch_fake("FakeKingston")
    t = 5
    rep = U.build(dev, I.circuit(t), str(tmp_path / "u"), sentinel_spec=dict(kind="echo_clifford", D=24))
    dr, dc, m, _ = I.place(dev)
    assert (rep["copy"]["drow"], rep["copy"]["dcol"]) == (dr, dc)
    assert rep["mapping"] == {str(q): m[q] for q in I.QUBITS}
    assert rep["cz"] == 2 * len(I.EDGES) * t
    assert all(rep["checks"].values())
    assert rep["circuits"] == 3 + 4 and rep["qpu_seconds_model"] < 60


def test_non_edge_gate_is_refused(tmp_path):
    from qiskit import QuantumCircuit
    dev = Dv.fetch_fake("FakeKingston")
    qc = QuantumCircuit(10)
    qc.h(3)
    qc.cz(3, 5)           # 3 and 5 are two apart on the row: not a coupler
    with pytest.raises(U.Refused, match="not an edge"):
        U.build(dev, qc, str(tmp_path / "x"))
    qc2 = QuantumCircuit(10, 1)
    qc2.measure(3, 0)
    qc2.x(3)
    with pytest.raises(U.Refused, match="after its measurement"):
        U.analyse(qc2, dev)


def test_floor_keeps_the_cz_and_returns_to_zero():
    from qiskit import QuantumCircuit
    dev = Dv.fetch_fake("FakeKingston")
    rc = RuleCompiler(dev)
    qc = QuantumCircuit(30)
    for q in (3, 4, 5, 16, 23):
        qc.ry(0.7, q)
    qc.rzz(1.0, 3, 4)
    qc.cx(4, 5)
    qc.cz(3, 16)
    qc.rzz(0.4, 16, 23)
    qc.rx(0.3, 23)
    labels, edges, logical = U.analyse(qc, dev)
    _, _, m, _ = __import__("psf_os.lattice2d", fromlist=["place"]).place(dev, edges, labels)
    phys = [m[q] for q in labels]
    cu = rc.pm.run(U.with_measure(U.relabel(logical, m, dev.num_qubits), phys))
    cf = U.floor_of(cu)
    assert M.cz_count(cf) == M.cz_count(cu) > 0
    assert {i.operation.name for i in cf.data} <= {"cz", "rz", "barrier", "measure"}
    (row,) = sim.run_circuits([cf], 1000, seed=3)
    assert set(row["counts"]) == {"0" * len(labels)}


def test_cli_run2d_and_card_on_a_local_run(tmp_path):
    from qiskit import QuantumCircuit, qpy
    dev = Dv.fetch_fake("FakeKingston")
    dev.save(str(tmp_path / "dev"))
    qc = QuantumCircuit(30)
    for q in (3, 4, 5, 16, 23):
        qc.h(q)
    for a, b in ((3, 4), (4, 5), (3, 16), (16, 23)):
        qc.cz(a, b)
    for q in (3, 4, 5, 16, 23):
        qc.h(q)
    with open(tmp_path / "c.qpy", "wb") as f:
        qpy.dump(qc, f)
    out = tmp_path / "run"
    assert cli.main(["run2d", "--device", str(tmp_path / "dev"), "--circuit", str(tmp_path / "c.qpy"),
                     "--out", str(out), "--dry-clifford-sentinel"]) == 0
    rep = json.load(open(out / "report.json"))
    sim.submit(rep["job"], "noiseless", seed=5)  # every cell is Clifford here (the sentinel at its Clifford point)
    c = U.card(rep["job"], rep["runs"], answers={str(q): 1.0 for q in rep["labels"]})
    assert abs(c["floor_mean"] - 1) < 1e-9 and abs(c["mirror_mean"] - 1) < 1e-9
    assert c["mirror_flags"]["qubits"] == []
    assert c["answered"]["qubits"] == len(rep["labels"])
    assert c["sentinel"] == "CLEAN"
