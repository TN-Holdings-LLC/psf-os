"""psf_os.report2d (the extended answer card, report only) and psf_os.gatehash (a QPY-version-free circuit hash)."""
import json

import numpy as np
import pytest

from psf_os import cli
from psf_os import device as Dv
from psf_os import gatehash as G
from psf_os import report2d as X
from psf_os import sim
from psf_os import user2d as U


def test_flags_unreadable_rule_r_neighbours_and_m_without():
    labels = [10, 11, 12, 13, 14, 15]
    edges = [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]
    floor = [0.95, 0.96, -0.003, 0.94, 0.70, 0.95]  # 12 unreadable; 14 below rule R's line
    z = [0.5, 0.6, 0.0, 0.5, 0.2, 0.7]
    f = X.flags(labels, edges, floor, z, answers={str(q): 0.6 for q in labels})
    assert f["unreadable"]["qubits"] == [12]
    assert f["rule_R"]["dropped"] == [12, 14]
    assert f["neighbours"] == [11, 13, 15]
    assert f["flagged"] == [11, 12, 13, 14, 15]
    assert f["M_all"] == pytest.approx(np.mean(z))
    assert f["M_without_flagged"] == pytest.approx(0.5) and f["qubits_without_flagged"] == 1
    assert f["difference_without_flagged"] == pytest.approx(-0.1)


def test_rule_r_is_p6s_formula():
    z0 = np.array([0.9, 0.91, 0.92, 0.5, 0.93, 0.95, 0.2])
    keep, rr = X.rule_r(z0)
    med = np.median(z0)
    mad = np.median(np.abs(z0 - med))
    assert rr["line"] == pytest.approx(med - max(5 * 1.4826 * mad, 0.05))
    assert list(keep) == list(z0 >= rr["line"])


def test_retention_block_matches_daily_score_definitions():
    z, a = np.array([0.4, 0.5, 0.3]), np.array([0.5, 0.6, 0.4])
    r = X.retention(z, a, layers=20)
    slope = (z * a).sum() / (a * a).sum()
    assert r["retention"] == pytest.approx(z.mean() / a.mean())
    assert r["norm_error"] == pytest.approx(abs(z.mean() - a.mean()) / a.mean())
    assert r["slope"] == pytest.approx(slope)
    assert r["mse_after_scale"] == pytest.approx(np.mean((z - slope * a) ** 2))
    assert r["decay_per_layer"] == pytest.approx(-np.log(slope) / 20)
    small = X.retention(np.array([0.01, 0.02]), np.array([0.05, 0.06]))
    assert small["retention"] is None and "small answer" in small["retention_note"]


def _clifford(labels, edges):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(max(labels) + 1)
    for q in labels:
        qc.h(q)
    for a, b in edges:
        qc.cz(a, b)
    for q in labels:
        qc.h(q)
    return qc


def test_extended_card_keeps_the_card_and_adds_blocks(tmp_path, capsys):
    from qiskit import qpy
    dev = Dv.fetch_fake("FakeKingston")
    dev.save(str(tmp_path / "dev"))
    a = _clifford((3, 4, 5, 16, 23), ((3, 4), (4, 5), (3, 16), (16, 23)))
    b = _clifford((60, 61, 62), ((60, 61), (61, 62)))
    for name, qc in (("a1", a), ("b2", b)):
        with open(tmp_path / f"{name}.qpy", "wb") as f:
            qpy.dump(qc, f)
    out = tmp_path / "run"
    assert cli.main(["run2d", "--device", str(tmp_path / "dev"), "--circuit", str(tmp_path / "a1.qpy"),
                     str(tmp_path / "b2.qpy"), "--out", str(out), "--dry-clifford-sentinel"]) == 0
    capsys.readouterr()
    rep = json.load(open(out / "report.json"))
    sim.submit(rep["job"], "noiseless", seed=7)
    zb = {str(x["q"]): x["z"] for x in U.card(rep["job"], rep["runs"], tag="b2")["qubits"]}
    zb = {q: (v if abs(v) > 0.2 else 0.5) for q, v in zb.items()}  # a stand-in answer: the measured values themselves
    ans = {"b2": zb}
    plain = U.cards(rep["job"], rep["runs"], ans)
    ext = X.extended_cards(rep["job"], rep["runs"], ans)
    for t in plain:
        assert {k: ext[t][k] for k in plain[t]} == plain[t]  # the card itself unchanged
        assert ext[t]["flags"]["flagged"] == [] and ext[t]["notes"]
        assert ext[t]["cz_layers"] >= 1
    assert "retention" not in ext["a1"]
    r = ext["b2"]["retention"]
    z = np.array([x["z"] for x in plain["b2"]["qubits"]])
    a = np.array([zb[str(x["q"])] for x in plain["b2"]["qubits"]])
    assert r["slope"] == pytest.approx((z * a).sum() / (a * a).sum())
    assert r["M"] == pytest.approx(plain["b2"]["answered"]["M"]) and r["M_exact"] == pytest.approx(a.mean())
    ansf = tmp_path / "ans.json"
    ansf.write_text(json.dumps(ans))
    assert cli.main(["result2d", "--job", rep["job"], "--runs", rep["runs"], "--answers", str(ansf),
                     "--extended"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"a1", "b2"} and "flags" in printed["a1"]


def test_gatehash_ignores_qpy_version_and_tiny_float_noise(tmp_path):
    from qiskit import QuantumCircuit, qpy

    def make(angle):
        qc = QuantumCircuit(3, 3)
        qc.rz(angle, 0)
        qc.sx(1)
        qc.cz(0, 1)
        qc.barrier()
        qc.rz(-np.pi / 2, 2)
        qc.measure([0, 1, 2], [0, 1, 2])
        return qc
    qc = make(0.1)
    h = G.circuits_hash([qc])
    files = {}
    for v in sorted({qpy.QPY_COMPATIBILITY_VERSION, qpy.QPY_VERSION}):
        p = tmp_path / f"c{v}.qpy"
        with open(p, "wb") as f:
            qpy.dump(qc, f, version=v)
        files[v] = p
    assert all(G.qpy_hash(str(p))["gate_sha256"] == h for p in files.values())
    if len(files) > 1:
        a, b = (open(p, "rb").read() for p in files.values())
        assert a != b  # the bytes differ, the gate hash does not
    assert G.circuits_hash([make(0.1 + 1e-13)]) == h
    assert G.circuits_hash([make(0.1 + 1e-6)]) != h


def test_run2d_circuits_are_the_same_build_after_build(tmp_path, capsys):
    from qiskit import qpy
    dev = Dv.fetch_fake("FakeKingston")
    dev.save(str(tmp_path / "dev"))
    a = _clifford((3, 4, 5, 16, 23), ((3, 4), (4, 5), (3, 16), (16, 23)))
    with open(tmp_path / "a1.qpy", "wb") as f:
        qpy.dump(a, f)
    hashes = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        assert cli.main(["run2d", "--device", str(tmp_path / "dev"), "--circuit", str(tmp_path / "a1.qpy"),
                         "--out", str(out)]) == 0
        hashes.append(G.qpy_hash(str(out / "job" / "circuits.qpy"))["gate_sha256"])
    capsys.readouterr()
    assert hashes[0] == hashes[1]
