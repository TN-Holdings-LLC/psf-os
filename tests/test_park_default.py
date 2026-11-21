"""park-default-2026-11-21: a qubit that waits in |1> in every branch is put in |0> for the wait (X before, X after,
fenced); |0> qubits, carriers and branch-dependent qubits are left alone; the noiseless distribution and the compiled
CZ count are unchanged; park=False and recorded builds (no "park" key) are as before, byte for byte."""
import json
import os

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from psf_os import device as Dv
from psf_os import feedforward as FF
from psf_os.footprint import exact_distribution, park_waits, wait_footprint


def _mixed():
    """q0 the read ancilla; q1 in |1> (parked); q2 in |0> (left); q3 in a superposition (a carrier, left); q4 |1> or
    |0> by the branch of the first read (left at the second wait)."""
    qc = QuantumCircuit(QuantumRegister(5, "q"), ClassicalRegister(4, "c"))
    qc.x(1)
    qc.h(3)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
        qc.x(4)
    qc.h(0)
    qc.measure(0, 1)
    with qc.if_test((qc.clbits[1], 1)):
        qc.x(0)
    qc.cz(1, 3)
    qc.h(3)
    qc.measure(1, 2)
    qc.measure(3, 3)
    return qc


def _same(d0, d1):
    return all(abs(d0.get(k, 0) - d1.get(k, 0)) < 1e-12 for k in set(d0) | set(d1))


def test_parks_only_the_qubit_in_one_in_every_branch():
    qc = _mixed()
    p, info = park_waits(qc)
    assert [w["qubits"] for w in info["parked"]] == [[1], [1]]       # q1 at both waits; q2, q3, q4 never
    assert info["waits"] == 2 and info["parked_waits"] == 2
    assert info["excited_before"] > info["excited_after"]
    assert _same(exact_distribution(qc), exact_distribution(p))
    xs = [ins for ins in p.data if ins.operation.name == "x" and p.find_bit(ins.qubits[0]).index == 1]
    assert len(xs) == 1 + 2 * 2                                          # the preparation and two pairs
    assert sum(1 for ins in p.data if ins.operation.name == "barrier") >= 4   # every pair inside fences


def test_nothing_to_park_returns_the_circuit_itself():
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
    qc.h(1)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    qc.h(1)
    qc.measure(1, 1)
    p, info = park_waits(qc)
    assert p is qc and info["parked"] == []


def test_branch_dependent_qubit_is_not_parked():
    qc = _mixed()
    p, _ = park_waits(qc)
    x4 = [ins for ins in p.data if ins.operation.name == "x" and p.find_bit(ins.qubits[0]).index == 4]
    assert x4 == []


def test_footprint_after_parking_has_no_excited_waiting_qubit_from_q1():
    p, _ = park_waits(_mixed())
    f = wait_footprint(p, conditionals=True)
    assert all(1 not in w["carriers"] for w in f["waits"])


def _ipe_like():
    """An IPE-shaped item FF accepts: q0 the ancilla, q1 the target prepared in |1> and read at the end."""
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(3, "c"))
    qc.x(1)
    for r in range(2):
        qc.h(0)
        qc.cz(0, 1)
        qc.h(0)
        qc.measure(0, r)
        with qc.if_test((qc.clbits[r], 1)):
            qc.x(0)
    qc.measure(1, 2)
    return qc


def test_build_parks_by_default_and_records_it(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("ip", _ipe_like(), dict(success={2: 1}))]
    out = str(tmp_path / "on")
    rep = FF.build(dev, items, out, shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24))
    rec = json.load(open(os.path.join(out, "runs", "ff_ip.json")))
    assert rec["park"]["park"] is True and rep["park"] is True
    parked = rec["park"]["arms"]["ff"]["parked"]
    assert parked and all(w["physical"] == [rec["chain"][1]] for w in parked)
    chk = json.load(open(os.path.join(out, "checks.json")))
    assert chk["items"]["ip"]["park"]["arms"]["ff"]["parked"] == parked
    out0 = str(tmp_path / "off")
    rep0 = FF.build(dev, items, out0, shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24), park=False)
    assert rep["items"]["ip"]["cz"] == rep0["items"]["ip"]["cz"]                 # the CZ count kept
    assert "park" not in rep0 and "park" not in json.load(open(os.path.join(out0, "runs", "ff_ip.json")))
    FF.run_local(out)
    assert FF.cards(out)["ip"]["ff"]["share"] == 1.0


def test_park_false_and_the_context_are_byte_for_byte_as_before(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("ip", _ipe_like(), dict(success={2: 1}))]
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    FF.build(dev, items, a, shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24), park=False)
    with FF.park_default(False):
        FF.build(dev, items, b, shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24))
    assert FF.PARK_DEFAULT is True
    for f in sorted(os.listdir(os.path.join(a, "runs"))):
        if f.endswith(".qpy"):
            assert open(os.path.join(a, "runs", f), "rb").read() == open(os.path.join(b, "runs", f), "rb").read()
    ra = json.load(open(os.path.join(a, "runs", "ff_ip.json")))
    rb = json.load(open(os.path.join(b, "runs", "ff_ip.json")))
    assert ra == rb


def test_a_record_without_the_key_reads_as_false():
    assert FF.park_of_record({}) is False
    assert FF.park_of_record({"park": {"park": True, "arms": {}}}) is True
    assert FF.park_of_record(None) is False


def test_nothing_to_park_builds_as_without_parking(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [(f"tp{k}", FF.teleport_test(k), dict(success={k: 0})) for k in (1, 2)]
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    FF.build(dev, items, a, shots=500, sentinel_spec=dict(kind="echo_clifford", D=24), park=False)
    FF.build(dev, items, b, shots=500, sentinel_spec=dict(kind="echo_clifford", D=24), park=True)
    for k in (1, 2):
        ra = json.load(open(os.path.join(a, "runs", f"ff_tp{k}.json")))
        rb = json.load(open(os.path.join(b, "runs", f"ff_tp{k}.json")))
        if rb["park"]["arms"]["ff"]["parked"]:
            continue                                   # teleport_test may hold a |1> waiter; then not this case
        assert open(os.path.join(a, "runs", f"ff_tp{k}.qpy"), "rb").read() == \
            open(os.path.join(b, "runs", f"ff_tp{k}.qpy"), "rb").read()


def test_a_changed_distribution_is_refused(monkeypatch):
    import psf_os.footprint as FP
    real = FP.exact_distribution
    calls = {"n": 0}

    def fake(qc):
        calls["n"] += 1
        d = real(qc)
        return d if calls["n"] % 2 else {"x": 1.0}
    monkeypatch.setattr(FP, "exact_distribution", fake)
    with pytest.raises(ValueError, match="changed the noiseless distribution"):
        park_waits(_mixed())


def test_loop_start_records_park_and_verify_reads_absent_as_false(tmp_path, monkeypatch):
    from psf_os import loop as LP
    seen = []

    class Exp:
        name, k_max, budget_s = "x", 4, 240.0
    out = str(tmp_path / "n")
    LP.start(out, Exp(), dry_devdir=str(tmp_path))
    assert json.load(open(os.path.join(out, "run.json")))["park"] is True
    cfg = json.load(open(os.path.join(out, "run.json")))
    cfg.pop("park")
    json.dump(cfg, open(os.path.join(out, "run.json"), "w"))
    monkeypatch.setattr(LP, "_verify", lambda o, e, r=False: seen.append(FF.PARK_DEFAULT) or True)
    LP.verify(out, Exp())
    assert seen == [False] and FF.PARK_DEFAULT is True
