"""psf_os.feedforward, register conditions (2026-11-18, branch register-cond-2026-11-18; home instruction
"psf-os にレジスタの条件"): if_test on a whole ClassicalRegister (<= 32 bits, every bit measured before, value in range),
several registers that partition the bits in order, a look-up decoder of the distance-3 repetition code that is right
in every shot noiselessly for every single X error, the refusals, restrict / wire_signature / bundle_acceptance /
components with register tests, the off and floor arms, the compiled register test (register and bit order kept), and
the broadcast record of the manifest against a hand count. Single-bit tests are covered, unchanged, by
test_feedforward.py."""
import json
import os

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr

from psf_os import device as Dv
from psf_os import feedforward as FF

ERRORS = (None, 0, 1, 2)
SUCC = {2: 0, 3: 0, 4: 0}          # c[0..2] are clbits 2..4 (register s holds clbits 0, 1)


def _sq(n=3, ns=2, nc=1):
    s, c = ClassicalRegister(ns, "s"), ClassicalRegister(nc, "c")
    return QuantumCircuit(QuantumRegister(n, "q"), s, c), s, c


# ------------------------------------------------------------------------------------------------ the look-up decoder

def test_rep3_lookup_is_accepted_and_counted():
    for e in ERRORS:
        a = FF.analyse(FF.rep3_lookup(e))
        assert a == dict(n=5, clbits=5, mid=2, conditionals=5, cz=16, cz_in_blocks=0, register_conditionals=3,
                         condition_bits=3 * 2 + 2)
        off = FF.rep3_lookup(e, feed_forward=False)
        with pytest.raises(FF.Refused, match="one classical register"):
            FF.analyse(off)                                   # two registers without a register test: as before
        ao = FF.analyse(off, many_registers=True)
        assert ao["conditionals"] == 2 and "register_conditionals" not in ao


def test_rep3_lookup_noiseless_every_error_corrected():
    from qiskit_aer import AerSimulator
    for e in ERRORS:
        for ff in (True, False):
            cnt = AerSimulator().run(FF.rep3_lookup(e, ff), shots=1000, seed_simulator=7).result().get_counts()
            c = {}
            for k, v in cnt.items():
                k = k.replace(" ", "")
                c[k[:3]] = c.get(k[:3], 0) + v                # register c (c[2] c[1] c[0])
            if ff:
                assert c == {"000": 1000}, (e, c)
            elif e == 0:
                assert set(c) == {"010", "011"}                  # d0's X reaches the T: c[0] random, c[1] kept 1
                assert 400 < c["011"] < 600


def test_rep3_job_end_to_end(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [(f"r{'n' if e is None else e}", FF.rep3_lookup(e),
              dict(success=SUCC, off_circuit=FF.rep3_lookup(e, feed_forward=False))) for e in ERRORS]
    out = str(tmp_path / "j")
    rep = FF.build(dev, items, out, shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24))
    assert rep["circuits"] == 4 * 3 + 4
    rec = json.load(open(os.path.join(out, "runs", "ff_r0.json")))
    assert rec["register_conditionals"] == 3 and rec["condition_bits"] == 8 and rec["off_explicit"]
    # the compiled ff arm keeps the register tests: register s over clbits 0, 1, values 1, 3, 2 in that order
    from qiskit import qpy
    cs = qpy.load(open(os.path.join(out, "job", "circuits.qpy"), "rb"))
    man = json.load(open(os.path.join(out, "job", "manifest.json")))
    names = [c["name"] for c in man["cells"] for _ in range(c["points"])]
    ff = cs[names.index("ff_r0")]
    assert [(r.name, r.size) for r in ff.cregs] == [("s", 2), ("c", 3)]
    tests = [(type(i.operation.condition[0]).__name__, i.operation.condition[1]) for i in ff.data
             if i.operation.name == "if_else"]
    assert tests[:3] == [("ClassicalRegister", 1), ("ClassicalRegister", 3), ("ClassicalRegister", 2)]
    reg = [i.operation.condition[0] for i in ff.data if i.operation.name == "if_else"][0]
    assert [ff.find_bit(b).index for b in reg] == [0, 1]
    # floor: every bit 0, the register tests kept (diagonal blocks); off: the explicit arm (no register test)
    fl = cs[names.index("ff_r0_floor")]
    assert sum(1 for i in fl.data if i.operation.name == "if_else") == 5
    assert all(n not in FF.NON_DIAGONAL_COMPILED for n, _ in FF.walk(fl))
    off = cs[names.index("ff_r0_off")]
    assert sum(1 for i in off.data if i.operation.name == "if_else") == 2
    # broadcast record against the hand count: ff 5 tests / 8 bits, off 2 / 2, floor 5 / 8 per item; sentinel 0
    b = man["broadcast"]
    assert b["conditionals"] == 4 * (5 + 2 + 5) and b["condition_bits"] == 4 * (8 + 2 + 8)
    assert b["max_conditionals_per_circuit"] == 5 and len(b["per_circuit"]) == rep["circuits"]
    assert b["per_circuit"][names.index("ff_r1")] == dict(conditionals=5, condition_bits=8)
    assert b["per_circuit"][names.index("ff_r1_off")] == dict(conditionals=2, condition_bits=2)
    FF.run_local(out, seed=3)
    cards = FF.cards(out)
    for t in ("rn", "r0", "r1", "r2"):
        assert cards[t]["ff"]["share"] == 1.0 and cards[t]["floor"]["all_zero"] == 1.0
    assert cards["rn"]["off"]["share"] == 1.0
    assert cards["r0"]["off"]["share"] == 0.0                         # without the look-up c[1] keeps d0's X (above)


def test_default_off_arm_removes_every_conditional():
    off = FF.strip_conditionals(FF.rep3_lookup(0))
    assert not any(i.operation.name == "if_else" for i in off.data)
    assert [(r.name, r.size) for r in off.cregs] == [("s", 2), ("c", 3)]


def test_static_jobs_keep_their_manifest(tmp_path):
    """A job without conditionals has no broadcast key (its manifest is as before)."""
    dev = Dv.fetch_fake("FakeFez")
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    rep = FF.build(dev, [("s0", qc, dict(success={1: 0}))], str(tmp_path / "s"), shots=100,
                   sentinel_spec=dict(kind="echo_clifford", D=24))
    man = json.load(open(os.path.join(str(tmp_path / "s"), "job", "manifest.json")))
    assert "broadcast" not in man and rep["circuits"] == 3 + 4
    rep = FF.build(dev, [("t1", FF.teleport_test(1), dict(success={1: 0}))], str(tmp_path / "t"), shots=100,
                   sentinel_spec=dict(kind="echo_clifford", D=24))
    man = json.load(open(os.path.join(str(tmp_path / "t"), "job", "manifest.json")))
    assert man["broadcast"]["conditionals"] == 2 and man["broadcast"]["condition_bits"] == 2   # ff 1 + floor 1


# ------------------------------------------------------------------------------------------------ refusals

def test_refusals_say_why():
    big = ClassicalRegister(33, "big")
    qc = QuantumCircuit(QuantumRegister(2, "q"), big)
    for k in range(33):
        qc.measure(0, big[k])
    with qc.if_test((big, 1)):
        qc.x(1)
    with pytest.raises(FF.Refused, match="at most 32"):
        FF.analyse(qc)

    qc, s, c = _sq()
    qc.measure(0, s[0])
    with qc.if_test((s, 1)):                                          # s[1] not measured yet
        qc.x(2)
    qc.measure(1, s[1])
    qc.measure(2, c[0])
    with pytest.raises(FF.Refused, match=r"before its bits \[1\] are measured"):
        FF.analyse(qc)

    for value in (4, -1):
        # Qiskit refuses such a value at if_test; a circuit can still carry one (set afterwards, or read from a file)
        qc, s, c = _sq()
        qc.measure([0, 1], [0, 1])
        with qc.if_test((s, 1)):
            qc.x(2)
        qc.measure(2, 2)
        op = [i for i in qc.data if i.operation.name == "if_else"][0].operation
        try:
            op.condition = (s, value)
        except Exception:                                             # noqa: BLE001  (this Qiskit refuses it too)
            continue
        with pytest.raises(FF.Refused, match="outside 0 .. 3"):
            FF.analyse(qc)

    qc, s, c = _sq()
    qc.measure([0, 1], [0, 1])
    with qc.if_test(expr.bit_xor(s[0], s[1])):                       # an expression
        qc.x(2)
    qc.measure(2, 2)
    with pytest.raises(FF.Refused, match="expressions"):
        FF.analyse(qc)

    qc, s, c = _sq()
    qc.measure([0, 1], [0, 1])
    with qc.if_test((s, 1)):
        qc.measure(2, 2)                                              # a measurement inside
    with pytest.raises(FF.Refused, match="X or Z|inside a conditional"):
        FF.analyse(qc)
    with pytest.raises(FF.Refused, match="inside a conditional"):
        FF.analyse(qc, form="blocks")

    qc, s, c = _sq()
    qc.measure([0, 1], [0, 1])
    with qc.if_test((s, 1)):
        qc.reset(2)                                                   # a reset inside
    qc.measure(2, 2)
    with pytest.raises(FF.Refused, match="inside a conditional"):
        FF.analyse(qc, form="blocks")

    qc, s, c = _sq()
    qc.measure([0, 1], [0, 1])
    with qc.if_test((s, 1)):
        with qc.if_test((s, 2)):                                      # nested
            qc.x(2)
    qc.measure(2, 2)
    with pytest.raises(FF.Refused, match="inside a conditional"):
        FF.analyse(qc, form="blocks")

    qc, s, c = _sq()
    qc.measure([0, 1], [0, 1])
    with qc.if_test((s, 1)) as else_:
        qc.x(2)
    with else_:
        qc.z(2)
    qc.measure(2, 2)
    with pytest.raises(FF.Refused, match="no else"):
        FF.analyse(qc)


def test_registers_must_partition_the_bits_in_order():
    q = QuantumRegister(3, "q")
    a = ClassicalRegister(2, "a")
    qc = QuantumCircuit(q, a)
    alias = ClassicalRegister(bits=list(a), name="al")             # a register over bits of another
    qc.add_register(alias)
    qc.measure([0, 1], [0, 1])
    with qc.if_test((alias, 1)):
        qc.x(2)
    with pytest.raises(FF.Refused, match="overlap"):
        FF.analyse(qc)
    s, c = ClassicalRegister(2, "s"), ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c, s)
    qc2 = QuantumCircuit(q)
    qc2.add_bits([s[1], s[0]])                                       # the circuit's bit order differs
    qc2.add_register(s)
    qc2.add_bits(list(c))
    qc2.add_register(c)
    qc2.measure(0, s[0])
    qc2.measure(1, s[1])
    with qc2.if_test((s, 1)):
        qc2.x(2)
    qc2.measure(2, c[0])
    with pytest.raises(FF.Refused, match="one register after another"):
        FF.analyse(qc2)


# ------------------------------------------------------------------------------------------------ helpers

def test_walk_bits_lists_every_register_bit():
    qc = FF.rep3_lookup(1)
    tested = [cs for n, qs, cs in FF.walk_bits(qc) if n == "if_else"]
    assert all(0 in cs and 1 in cs for cs in tested[:3])
    assert -1 not in {b for cs in tested for b in cs}
    parts = FF.components(qc)
    assert parts == [([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])]


def test_restrict_rebuilds_the_register_and_keeps_the_signature():
    qc = FF.rep3_lookup(2)
    # restrict onto the same qubits and bits: a register of the same bits (aliasing c), same signature
    r = FF.restrict(qc, list(range(5)), list(range(5)))
    regs = [(x.name, [r.find_bit(b).index for b in x]) for x in r.cregs]
    assert regs == [("c", [0, 1, 2, 3, 4]), ("r0", [0, 1])]
    assert FF.wire_signature(r) == FF.wire_signature(qc)
    # bits given in another order: the register's bits follow (s[0] = old bit 0 -> new 4, s[1] = old 1 -> new 3)
    r2 = FF.restrict(qc, list(range(5)), [4, 3, 2, 1, 0])
    reg = [x for x in r2.cregs if x.name == "r0"][0]
    assert [r2.find_bit(b).index for b in reg] == [4, 3]
    sig = dict(FF.wire_signature(r2))
    assert any(s[0] == "if_else" and s[3] == ("register", (4, 3), 3) for s in sig[("c", 4)])
    # the register tests and their values survive a round trip through restrict
    back = FF.restrict(r2, list(range(5)), [4, 3, 2, 1, 0])
    assert FF.wire_signature(back) == FF.wire_signature(qc)
    # a wrong value is seen
    bad = FF.rep3_lookup(2)
    ops = [i for i in bad.data if i.operation.name == "if_else"]
    ops[0].operation.condition = (ops[0].operation.condition[0], 2)
    assert FF.wire_signature(bad) != FF.wire_signature(qc)
    # a register spanning bits outside the given set straddles it
    with pytest.raises(FF.Refused, match="straddles"):
        FF.restrict(qc, list(range(5)), [1, 2, 3, 4])


def test_bundle_of_lookups_and_the_refusal_across_items(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("la", FF.rep3_lookup(0), dict(success=SUCC)), ("lb", FF.rep3_lookup(2), dict(success=SUCC))]
    out = str(tmp_path / "b")
    rep = FF.build_bundle(dev, items, out, shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24))
    assert rep["circuits"] == 3 + 4
    from qiskit import qpy
    cs = qpy.load(open(os.path.join(out, "job", "circuits.qpy"), "rb"))
    assert [(r.name, r.size) for r in cs[0].cregs] == [("s0_s", 2), ("s0_c", 3), ("s1_s", 2), ("s1_c", 3)]
    FF.run_local(out, seed=5)
    cards = FF.cards(out)
    assert cards["la"]["ff"]["share"] == 1.0 and cards["lb"]["ff"]["share"] == 1.0
    # a register holding a bit of each item is refused
    q = QuantumRegister(4, "q")
    r = ClassicalRegister(2, "r")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, r, c)
    qc.measure(0, r[0])                                              # item 0's bit
    qc.measure(2, r[1])                                              # item 1's bit
    with qc.if_test((r, 3)):
        qc.x(1)
    qc.measure([1, 3], [2, 3])
    with pytest.raises(FF.Refused, match="spans sub-items"):
        FF.bundle_acceptance({"ff": qc}, [{0, 1}, {2, 3}], [{0, 2}, {1, 3}])


def test_broadcast_record_hand_count():
    a, b = FF.rep3_lookup(1), FF.teleport_test(3)
    rec = FF.broadcast_record([a, b, FF.strip_conditionals(a)])
    assert rec["per_circuit"] == [dict(conditionals=5, condition_bits=8), dict(conditionals=3, condition_bits=3),
                                  dict(conditionals=0, condition_bits=0)]
    assert rec["conditionals"] == 8 and rec["condition_bits"] == 11 and rec["max_conditionals_per_circuit"] == 5
