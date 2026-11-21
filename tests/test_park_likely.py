"""likely-park-2026-11-22 (a 0.3.1 candidate, off by default): footprint.park_likely parks every waiting qubit with
P(|1>) > 0.5 (entangled ones too) with an X pair kept by a partial fence at those waits only; build(park_likely=...)
applies it after `park`, checked as `park` is; without it everything is 0.3.0's, byte for byte."""
import json
import math
import os

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from psf_os import device as Dv
from psf_os import feedforward as FF
from psf_os.footprint import exact_distribution, park_likely, park_waits

SPEC = dict(kind="echo_clifford", D=24)


def _likely():
    """q0 the read ancilla (two reads, a conditional after each); q1, q2 entangled with P(|1>) = 0.8 each (parked);
    q3 with P(|1>) = 0.2 (not parked); q4 |0>."""
    qc = QuantumCircuit(QuantumRegister(5, "q"), ClassicalRegister(5, "c"))
    qc.ry(2 * math.asin(math.sqrt(0.8)), 1)
    qc.h(2)                                   # CX(1, 2) written as H CZ H (psf-os takes CZ only)
    qc.cz(1, 2)
    qc.h(2)
    qc.ry(2 * math.asin(math.sqrt(0.2)), 3)
    for r in range(2):
        qc.h(0)
        qc.measure(0, r)
        with qc.if_test((qc.clbits[r], 1)):
            qc.x(0)
    qc.h(1)
    qc.measure(1, 2)
    qc.measure(2, 3)
    qc.measure(3, 4)
    return qc


def test_parks_the_likely_value_with_a_partial_fence():
    qc = _likely()
    p, info = park_likely(qc)
    assert [w["qubits"] for w in info["parked"]] == [[1, 2], [1, 2]]
    assert all(abs(x - 0.8) < 1e-9 for w in info["parked"] for x in w["p1"])
    d0, d1 = exact_distribution(qc), exact_distribution(p)
    assert all(abs(d0.get(k, 0) - d1.get(k, 0)) < 1e-12 for k in set(d0) | set(d1))
    bars = [ins for ins in p.data if ins.operation.name == "barrier"]
    assert bars and all(sorted(p.find_bit(q).index for q in b.qubits) == [0, 1, 2] for b in bars)   # never q3, q4
    xs = lambda c, q: sum(1 for ins in c.data if ins.operation.name == "x" and c.find_bit(ins.qubits[0]).index == q)
    assert xs(p, 1) == 4 and xs(p, 2) == 4 and xs(p, 3) == 0


def test_nothing_likely_returns_the_circuit():
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
    qc.ry(0.3, 1)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    qc.measure(1, 1)
    p, info = park_likely(qc)
    assert p is qc and info["parked"] == []


def test_build_keeps_the_pair_and_the_cz(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("lk", _likely(), dict(success={4: 0}))]
    a, b = str(tmp_path / "off"), str(tmp_path / "on")
    r0 = FF.build(dev, items, a, shots=500, sentinel_spec=SPEC, park=True)
    r1 = FF.build(dev, items, b, shots=500, sentinel_spec=SPEC, park=True, park_likely=True)
    assert r0["items"]["lk"]["cz"] == r1["items"]["lk"]["cz"]
    rec = json.load(open(os.path.join(b, "runs", "ff_lk.json")))
    parked = rec["park_likely"]["arms"]["ff"]["parked"]
    assert parked and all(w["physical"] == [rec["chain"][1], rec["chain"][2]] for w in parked)
    assert "park_likely" not in json.load(open(os.path.join(a, "runs", "ff_lk.json")))
    assert json.load(open(os.path.join(b, "job", "manifest.json")))["psf_os_build"]["park_likely"] is True
    assert "park_likely" not in json.load(open(os.path.join(a, "job", "manifest.json")))["psf_os_build"]
    # the compiled pair survives: more one-qubit gates on the parked qubits than without it
    from qiskit import qpy
    c0 = qpy.load(open(os.path.join(a, "runs", "ff_lk.qpy"), "rb"))[0]
    c1 = qpy.load(open(os.path.join(b, "runs", "ff_lk.qpy"), "rb"))[0]
    n1q = lambda c: sum(1 for nm, _ in FF.walk(c) if nm in ("x", "sx", "rz"))
    assert n1q(c1) > n1q(c0)
    d0 = exact_distribution(FF.compact(c0))
    d1 = exact_distribution(FF.compact(c1))
    assert all(abs(d0.get(k, 0) - d1.get(k, 0)) < 1e-9 for k in set(d0) | set(d1))


def test_park_likely_names_items(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("la", _likely(), dict(success={4: 0})), ("lb", _likely(), dict(success={4: 0}))]
    out = str(tmp_path / "j")
    FF.build(dev, items, out, shots=500, sentinel_spec=SPEC, park=True, park_likely=["la"])
    assert "park_likely" in json.load(open(os.path.join(out, "runs", "ff_la.json")))
    assert "park_likely" not in json.load(open(os.path.join(out, "runs", "ff_lb.json")))


def test_off_by_default_byte_for_byte(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("lk", _likely(), dict(success={4: 0}))]
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    FF.build(dev, items, a, shots=500, sentinel_spec=SPEC, park=True)
    FF.build(dev, items, b, shots=500, sentinel_spec=SPEC, park=True, park_likely=False)
    for f in sorted(os.listdir(os.path.join(a, "runs"))):
        assert open(os.path.join(a, "runs", f), "rb").read() == open(os.path.join(b, "runs", f), "rb").read()


def test_the_always_one_case_is_left_to_park():
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
    p, _ = park_waits(qc)
    p2, info = park_likely(p)
    assert p2 is p and info["parked"] == []          # park took it already


def _likely_cx():
    """As _likely, with CX(1, 2) again after the waits (written H CZ H) and q1 read at once: the two CX cancel when
    nothing stands between them."""
    qc = QuantumCircuit(QuantumRegister(5, "q"), ClassicalRegister(5, "c"))
    qc.ry(2 * math.asin(math.sqrt(0.8)), 1)
    qc.h(2)
    qc.cz(1, 2)
    qc.h(2)
    qc.ry(2 * math.asin(math.sqrt(0.2)), 3)
    for r in range(2):
        qc.h(0)
        qc.measure(0, r)
        with qc.if_test((qc.clbits[r], 1)):
            qc.x(0)
    qc.h(2)
    qc.cz(1, 2)
    qc.h(2)
    qc.measure(1, 2)
    qc.measure(2, 3)
    qc.measure(3, 4)
    return qc


def test_a_pair_the_compiler_would_cancel_unfenced_is_refused(tmp_path):
    # unfenced, the compiler cancels the two CX (0 CZ); parking puts a barrier on q1, q2 between them (2 CZ): the
    # build refuses a changed CZ count, as park does
    import pytest
    dev = Dv.fetch_fake("FakeFez")
    with pytest.raises(FF.Refused, match="changed the compiled CZ count"):
        FF.build(dev, [("cx", _likely_cx(), dict(success={4: 0}))], str(tmp_path / "x"), shots=500, sentinel_spec=SPEC,
                 park=True, park_likely=True)
