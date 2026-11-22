"""likely-park-2026-11-22, released in 0.3.1 as the default: footprint.park_likely parks every waiting qubit with
P(|1>) > 0.5 (entangled ones too) with an X pair kept by a partial fence at those waits only; build(park_likely=None)
applies it after `park` by default, checked as `park` is; park_likely=False builds as 0.3.0, byte for byte; the safety
guard refuses an item it would change when the declared psf-os had the other default."""
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
    r0 = FF.build(dev, items, a, shots=500, sentinel_spec=SPEC, park=True, park_likely=False)
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


def test_the_default_parks_and_false_is_0_3_0(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("lk", _likely(), dict(success={4: 0}))]
    a, b, c, d = (str(tmp_path / x) for x in "abcd")
    FF.build(dev, items, a, shots=500, sentinel_spec=SPEC)                         # the 0.3.1 default
    FF.build(dev, items, b, shots=500, sentinel_spec=SPEC, park_likely=True)
    assert open(os.path.join(a, "runs", "ff_lk.qpy"), "rb").read() == open(os.path.join(b, "runs", "ff_lk.qpy"),
                                                                           "rb").read()
    assert json.load(open(os.path.join(a, "job", "manifest.json")))["psf_os_build"]["park_likely_set"] == "default"
    FF.build(dev, items, c, shots=500, sentinel_spec=SPEC, park_likely=False)
    with FF.park_default(True, likely=False):
        FF.build(dev, items, d, shots=500, sentinel_spec=SPEC)
    for f in sorted(os.listdir(os.path.join(c, "runs"))):
        assert open(os.path.join(c, "runs", f), "rb").read() == open(os.path.join(d, "runs", f), "rb").read()
    assert "park_likely" not in json.load(open(os.path.join(c, "runs", "ff_lk.json")))
    assert FF.PARK_LIKELY_DEFAULT is True


def test_park_default_false_builds_as_before_0_3_0(tmp_path):
    with FF.park_default(False):
        assert FF.PARK_DEFAULT is False and FF.PARK_LIKELY_DEFAULT is False
    assert FF.PARK_DEFAULT is True and FF.PARK_LIKELY_DEFAULT is True
    assert FF.park_likely_of_record({}) is False and FF.park_likely_of_record({"park_likely": {"park_likely": True}})


def test_the_guard_for_the_likely_default(tmp_path, monkeypatch):
    import pytest
    dev = Dv.fetch_fake("FakeFez")
    assert FF.parks_likely_by_default("0.3.1") and not FF.parks_likely_by_default("0.3.0")
    assert not FF.parks_likely_by_default("a6170e5")
    for declared in ("0.3.0", "a6170e5"):
        with pytest.raises(FF.Refused, match="park the likely value by default"):
            FF.build(dev, [("lk", _likely(), dict(success={4: 0}))], str(tmp_path / f"x{declared}"), shots=500,
                     sentinel_spec=SPEC, park=True, declared=declared)
    # an item parking the likely value leaves alone is built as either version builds it
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
    FF.build(dev, [("ip", qc, dict(success={2: 1}))], str(tmp_path / "ipe"), shots=500, sentinel_spec=SPEC,
             declared="0.3.0")
    FF.build(dev, [("lk", _likely(), dict(success={4: 0}))], str(tmp_path / "said"), shots=500, sentinel_spec=SPEC,
             park=True, declared="0.3.0", park_likely=False)                      # said: built
    monkeypatch.setenv("PSF_OS_DECLARED", "0.3.1")
    FF.build(dev, [("lk", _likely(), dict(success={4: 0}))], str(tmp_path / "rel"), shots=500, sentinel_spec=SPEC)


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


def test_version_and_the_guard_for_0_3_1(tmp_path, monkeypatch):
    import psf_os
    assert FF.parks_likely_by_default(psf_os.__version__)             # 0.3.1 and later (0.3.2: the same default)
    assert FF.parks_by_default("0.3.1") is True and FF.parks_by_default("0.3.0") is True
    monkeypatch.setenv("PSF_OS_DECLARED", "0.3.1")
    assert FF.resolve_park(None) == (True, "default")              # the same park default as 0.3.0: no refusal
    dev = Dv.fetch_fake("FakeFez")
    out = str(tmp_path / "j")
    FF.build(dev, [("lk", _likely(), dict(success={4: 0}))], out, shots=500, sentinel_spec=SPEC, park_likely=True)
    b = json.load(open(os.path.join(out, "job", "manifest.json")))["psf_os_build"]
    assert b["version"] == psf_os.__version__ and b["park"] is True and b["park_likely"] is True
    assert b["declared"] == "0.3.1"
