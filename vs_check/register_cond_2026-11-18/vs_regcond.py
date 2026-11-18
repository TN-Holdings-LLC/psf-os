"""Independent check of the register conditions (branch register-cond-2026-11-18) by the workplace VS Claude,
2026-11-18 (home instruction 18h, item 1). The functions under test are called; everything that decides PASS or FAIL
is written here with qiskit and qiskit_aer only (no psf-os helper, no test of the branch is reused).

  python vs_regcond.py DEVICE_DIR WORKDIR          (DEVICE_DIR: a notebook record, e.g. .../ibm_marrakesh/dev)

Sections: S1 refusals and an accepted 32-bit test; S2 the look-up decoder (logical, and compiled by build: register
and bit order kept); S3 restrict's register bit order; S4 bundle_acceptance across sub-items, and a bundle compiled
end to end; S5 the broadcast record against a hand count; S6 mutated circuits (a value shifted by one, the syndrome
bits exchanged).
"""
import json
import os
import shutil
import sys

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy
from qiskit.circuit import Clbit
from qiskit.circuit.classical import expr
from qiskit_aer import AerSimulator

from psf_os import feedforward as FF
from psf_os.device import Device

RESULTS = []
SIM = AerSimulator()


def report(section, name, ok, note=""):
    RESULTS.append(dict(section=section, name=name, ok=bool(ok), note=note))
    print(f"{section} {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note else ""), flush=True)


def counts(qc, shots=1000, seed=11):
    cnt = SIM.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
    return {k.replace(" ", ""): v for k, v in cnt.items()}


def bits_of(key, idx):
    """The value of clbit idx in a counts key with spaces removed (clbit 0 rightmost)."""
    return int(key[-1 - idx])


# ------------------------------------------------------------------------------------------------ S1 refusals

def s1():
    def refused(qc, why):
        try:
            FF.analyse(qc, form="blocks")
        except FF.Refused as e:
            report("S1", f"refused: {why}", True, str(e)[:70])
            return
        except Exception as e:  # noqa: BLE001 - a crash is not a refusal with a reason
            report("S1", f"refused: {why}", False, f"crashed instead: {type(e).__name__}: {e}"[:90])
            return
        report("S1", f"refused: {why}", False, "accepted")

    def base(w, measured=None):
        q, s, c = QuantumRegister(max(w, 2), "q"), ClassicalRegister(w, "s"), ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, s, c)
        for k in (range(w) if measured is None else measured):
            qc.measure(q[k], s[k])
        return qc, q, s, c

    qc, q, s, c = base(33)
    with qc.if_test((s, 1)):
        qc.x(q[0])
    qc.measure(q[0], c[0])
    refused(qc, "a 33-bit register")

    qc, q, s, c = base(32)
    with qc.if_test((s, 2 ** 32 - 1)):
        qc.x(q[0])
    qc.measure(q[0], c[0])
    try:
        a = FF.analyse(qc, form="blocks")
        report("S1", "accepted: a 32-bit register with value 2^32 - 1", a.get("register_conditionals") == 1,
               f"condition_bits {a.get('condition_bits')}")
    except Exception as e:  # noqa: BLE001
        report("S1", "accepted: a 32-bit register with value 2^32 - 1", False, f"{type(e).__name__}: {e}"[:90])

    qc, q, s, c = base(2, measured=[0])
    with qc.if_test((s, 1)):
        qc.x(q[0])
    qc.measure(q[0], c[0])
    refused(qc, "a register with a bit not yet measured")

    for v in (4, -1):
        qc, q, s, c = base(2)
        try:
            with qc.if_test((s, v)):
                qc.x(q[0])
        except Exception as e:  # noqa: BLE001 - qiskit itself may refuse the value
            report("S1", f"refused: value {v} for 2 bits", True, f"by qiskit at construction: {type(e).__name__}")
            continue
        qc.measure(q[0], c[0])
        refused(qc, f"value {v} for 2 bits")

    qc, q, s, c = base(2)
    with qc.if_test(expr.equal(s, 1)):
        qc.x(q[0])
    qc.measure(q[0], c[0])
    refused(qc, "an expression (s == 1)")

    qc, q, s, c = base(2)
    with qc.if_test(expr.bit_and(s[0], s[1])):
        qc.x(q[0])
    qc.measure(q[0], c[0])
    refused(qc, "an expression (s[0] & s[1])")

    qc, q, s, c = base(2)
    with qc.if_test((s, 1)):
        qc.measure(q[0], c[0])
    refused(qc, "a measurement in the block")

    qc, q, s, c = base(2)
    with qc.if_test((s, 1)):
        qc.reset(q[0])
    qc.measure(q[0], c[0])
    refused(qc, "a reset in the block")

    qc, q, s, c = base(2)
    with qc.if_test((s, 1)):
        with qc.if_test((s[0], 1)):
            qc.x(q[0])
    qc.measure(q[0], c[0])
    refused(qc, "nesting")

    qc, q, s, c = base(2)
    with qc.if_test((s, 1)) as other:
        qc.x(q[0])
    with other:
        qc.z(q[0])
    qc.measure(q[0], c[0])
    refused(qc, "an else branch")

    # registers that alias or are out of order
    q = QuantumRegister(2, "q")
    bits = [Clbit() for _ in range(3)]
    r1 = ClassicalRegister(bits=bits[:2], name="s")
    r2 = ClassicalRegister(bits=bits[1:], name="t")
    qc = QuantumCircuit(q, r1, r2)
    qc.measure(q[0], bits[0]); qc.measure(q[1], bits[1])
    with qc.if_test((r1, 1)):
        qc.x(q[0])
    qc.measure(q[0], bits[2])
    refused(qc, "overlapping registers")


# ------------------------------------------------------------------------------------------------ S2 look-up decoder

def expected_table():
    """From the code itself: a0 = d0 + d1 -> s[0], a1 = d1 + d2 -> s[1]; value = s[0] + 2 s[1]; an X on data k gives
    the syndrome below, and the decoder must flip data k for that value."""
    syn = {0: (1, 0), 1: (1, 1), 2: (0, 1)}
    return {s0 + 2 * s1: k for k, (s0, s1) in syn.items()}


def read_table(qc):
    """{value: data qubit index} of the register tests of the logical circuit, read from its instructions."""
    t = {}
    for ins in qc.data:
        op = ins.operation
        if op.name == "if_else" and isinstance(op.condition[0], ClassicalRegister):
            body = op.blocks[0]
            (x,) = [i for i in body.data if i.operation.name == "x"]
            q = qc.find_bit(ins.qubits[body.find_bit(x.qubits[0]).index]).index
            t[int(op.condition[1])] = {0: 0, 2: 1, 4: 2}[q]
    return t


def c_bits_ok(cnt, cpos):
    return all(all(bits_of(k, p) == 0 for p in cpos) for k in cnt)


def s2(devdir, work):
    t = read_table(FF.rep3_lookup(None))
    report("S2", "look-up table in the circuit equals the one derived from the parities", t == expected_table(), str(t))
    for e in (None, 0, 1, 2):
        cnt = counts(FF.rep3_lookup(e))
        report("S2", f"logical circuit, error {e}: c = 000 in every shot", c_bits_ok(cnt, (2, 3, 4)),
               f"{len(cnt)} outcome(s)")
    out = os.path.join(work, "s2_build")
    shutil.rmtree(out, ignore_errors=True)
    succ = {2: 0, 3: 0, 4: 0}
    items = [(f"r{'n' if e is None else e}", FF.rep3_lookup(e),
              dict(success=succ, off_circuit=FF.rep3_lookup(e, feed_forward=False))) for e in (None, 0, 1, 2)]
    FF.build(Device.load(devdir), items, out, shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24))
    check_job(out, "S2", "compiled")
    return out


def compact(c):
    used = sorted({c.find_bit(q).index for ins in c.data for q in ins.qubits})
    pos = {q: i for i, q in enumerate(used)}
    out = QuantumCircuit(QuantumRegister(len(used), "q"), *c.cregs)
    for ins in c.data:
        out.append(ins.operation, [out.qubits[pos[c.find_bit(q).index]] for q in ins.qubits],
                   [out.clbits[c.find_bit(b).index] for b in ins.clbits])
    return out


def job_circuits(out):
    m = json.load(open(os.path.join(out, "job", "manifest.json")))
    cs = qpy.load(open(os.path.join(out, "job", "circuits.qpy"), "rb"))
    names = [cell["name"] + (f"#{p}" if cell["points"] > 1 else "") for cell in m["cells"] for p in range(cell["points"])]
    roles = [cell["role"] for cell in m["cells"] for _ in range(cell["points"])]
    return m, list(zip(names, roles, cs))


def check_job(out, sec, label, cpos=(2, 3, 4), expect_ok=True):
    """Noiseless Aer of every compiled arm: ff right in every shot, floor all zero; the registers of a compiled
    register test are the logical ones (s of 2 bits, then c), and every register test reads s."""
    m, rows = job_circuits(out)
    bad = []
    for name, role, c in rows:
        if role == "sentinel":
            continue
        cc = compact(c)
        regs = [(r.name, len(r)) for r in cc.cregs]
        for ins in cc.data:
            op = ins.operation
            if op.name == "if_else" and isinstance(op.condition[0], ClassicalRegister):
                reg = op.condition[0]
                if [cc.find_bit(b).index for b in reg] != [0, 1]:
                    bad.append(f"{name}: register test on bits {[cc.find_bit(b).index for b in reg]}")
        cnt = counts(cc)
        if role == "floor":
            if any(set(k) != {"0"} for k in cnt):
                bad.append(f"{name}: floor not all zero")
        elif role == "user":
            if not c_bits_ok(cnt, cpos):
                bad.append(f"{name}: c not 000 in every shot")
        if regs[:2] != [("s", 2), ("c", 3)]:
            bad.append(f"{name}: registers {regs}")
    ok = (not bad) if expect_ok else bool(bad)
    report(sec, f"{label}: every ff arm right, every floor zero, register tests read bits [0, 1]" if expect_ok
           else f"{label}: caught", ok, "; ".join(bad)[:150] if bad else "none wrong")
    return bad


# ------------------------------------------------------------------------------------------------ S3 restrict

def s3():
    for perm in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 0, 4, 2, 3]):
        allok = True
        for e in (None, 0, 1, 2):
            qc = FF.rep3_lookup(e)
            r = FF.restrict(qc, list(range(5)), perm)
            # where the old bits went: old bit b -> new position perm.index(b)
            new = {b: perm.index(b) for b in range(5)}
            for ins in r.data:
                op = ins.operation
                if op.name == "if_else" and isinstance(op.condition[0], ClassicalRegister):
                    got = [r.find_bit(b).index for b in op.condition[0]]
                    if got != [new[0], new[1]]:
                        allok = False
            cnt = counts(r)
            allok &= c_bits_ok(cnt, [new[2], new[3], new[4]])
        report("S3", f"restrict with clbits {perm}: register bits in s's order, decoder right for every error", allok)
    try:
        FF.restrict(FF.rep3_lookup(0), list(range(5)), [1, 2, 3, 4])
        report("S3", "restrict leaving out a register bit is refused", False, "accepted")
    except FF.Refused as e:
        report("S3", "restrict leaving out a register bit is refused", True, str(e)[:60])


# ------------------------------------------------------------------------------------------------ S4 bundles

def s4(devdir, work):
    q = QuantumRegister(4, "q")
    bits = [Clbit() for _ in range(4)]
    r = ClassicalRegister(bits=[bits[0], bits[1]], name="r")      # bit 0 of item A, bit 1 of item B
    rest = ClassicalRegister(bits=bits[2:], name="z")
    qc = QuantumCircuit(q, r, rest)
    qc.measure(q[0], bits[0]); qc.measure(q[2], bits[1])
    with qc.if_test((r, 1)):
        qc.x(q[1])
    qc.measure(q[1], bits[2]); qc.measure(q[3], bits[3])
    try:
        FF.bundle_acceptance({"ff": qc}, [{0, 1}, {2, 3}], [{0, 2}, {1, 3}])
        report("S4", "bundle_acceptance refuses a register test reading another item's bit", False, "accepted")
    except FF.Refused as e:
        report("S4", "bundle_acceptance refuses a register test reading another item's bit", True, str(e)[:60])
    r = ClassicalRegister(bits=[bits[0], bits[1]], name="r")
    qc = QuantumCircuit(q, r, rest)
    qc.measure(q[0], bits[0]); qc.measure(q[1], bits[1])
    with qc.if_test((r, 1)):
        qc.x(q[0])
    qc.measure(q[2], bits[2]); qc.measure(q[3], bits[3])
    try:
        FF.bundle_acceptance({"ff": qc}, [{0, 1}, {2, 3}], [{0, 1}, {2, 3}])
        report("S4", "bundle_acceptance accepts a register test inside one item", True)
    except FF.Refused as e:
        report("S4", "bundle_acceptance accepts a register test inside one item", False, str(e)[:60])
    out = os.path.join(work, "s4_bundle")
    shutil.rmtree(out, ignore_errors=True)
    succ = {2: 0, 3: 0, 4: 0}
    items = [(f"r{e}", FF.rep3_lookup(e), dict(success=succ, off_circuit=FF.rep3_lookup(e, feed_forward=False)))
             for e in (0, 2)]
    FF.build_bundle(Device.load(devdir), items, out, shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24))
    m, rows = job_circuits(out)
    bad = []
    for name, role, c in rows:
        if role != "user":
            continue
        cc = compact(c)
        order = [b for rg in cc.cregs for b in rg]
        regtests = [[cc.find_bit(b).index for b in i.operation.condition[0]] for i in cc.data
                    if i.operation.name == "if_else" and isinstance(i.operation.condition[0], ClassicalRegister)]
        if order != list(cc.clbits) or len(set(order)) != len(order):
            bad.append(f"{name}: registers do not partition the bits")
        if sorted(map(tuple, regtests)) != sorted([(0, 1)] * 3 + [(5, 6)] * 3):
            bad.append(f"{name}: register tests on {regtests}")
        cnt = counts(cc)
        if not c_bits_ok(cnt, (2, 3, 4, 7, 8, 9)):
            bad.append(f"{name}: a c bit not 0")
    report("S4", "bundle of two look-ups compiled: registers partition the bits, tests on (0,1) and (5,6), all right",
           not bad, "; ".join(bad)[:150])


# ------------------------------------------------------------------------------------------------ S5 broadcast record

def hand_count(c):
    n = bits = 0
    for ins in c.data:
        op = ins.operation
        if op.name == "if_else":
            n += 1
            bits += len(op.condition[0]) if isinstance(op.condition[0], ClassicalRegister) else 1
    return n, bits


def ibm_broadcasts(c):
    """IBM's reading (a broadcast = a transfer of unique classical data; tests on an unchanged register are one)."""
    stamp, sent, nb, nbits, t = {}, {}, 0, 0, 0
    for ins in c.data:
        t += 1
        op = ins.operation
        if op.name == "measure":
            stamp[c.find_bit(ins.clbits[0]).index] = t
        elif op.name == "if_else":
            tgt = op.condition[0]
            cb = [c.find_bit(b).index for b in ([tgt] if isinstance(tgt, Clbit) else list(tgt))]
            if any(stamp.get(b, 0) > sent.get(b, -1) for b in cb):
                nb += 1
                nbits += len(cb)
                for b in cb:
                    sent[b] = t
    return nb, nbits


def s5(out):
    m, rows = job_circuits(out)
    rec = m.get("broadcast")
    if rec is None:
        report("S5", "manifest has a broadcast record", False)
        return
    hand = [hand_count(c) for _, _, c in rows]
    got = [(p["conditionals"], p["condition_bits"]) for p in rec["per_circuit"]]
    report("S5", "broadcast record per circuit equals the hand count", got == hand,
           f"job: {rec['conditionals']} conditionals, {rec['condition_bits']} bits; hand {sum(h[0] for h in hand)}, "
           f"{sum(h[1] for h in hand)}")
    ibm = [ibm_broadcasts(c) for _, _, c in rows]
    ff = [(n, h, i) for (n, _, _), h, i in zip(rows, hand, ibm) if n == "ff_rn"]
    report("S5", "beside: IBM's broadcasts (unchanged register = one) for the ff arm of error None", True,
           f"record {ff[0][1] if ff else '-'} (conditionals, bits); IBM reading {ff[0][2] if ff else '-'}")
    return dict(record=got, hand=hand, ibm=ibm)


# ------------------------------------------------------------------------------------------------ S6 mutated circuits

def rep3_mutant(e, kind):
    """rep3_lookup with one damage: 'value' (the test that corrects d0 compares with 2 instead of 1), or 'swap' (the
    two syndrome measurements write each other's bit)."""
    qc = FF.rep3_lookup(e)
    out = qc.copy_empty_like()
    for ins in qc.data:
        op = ins.operation
        if kind == "value" and op.name == "if_else" and isinstance(op.condition[0], ClassicalRegister) \
                and int(op.condition[1]) == 1:
            op = op.replace_blocks(op.blocks)
            op.condition = (op.condition[0], 2)
        if kind == "swap" and op.name == "measure" and qc.find_bit(ins.clbits[0]).index in (0, 1):
            b = qc.find_bit(ins.clbits[0]).index
            out.append(op, ins.qubits, [out.clbits[1 - b]])
            continue
        out.append(op, ins.qubits, ins.clbits)
    return out


def s6(devdir, work):
    for kind in ("value", "swap"):
        logical_wrong = [e for e in (None, 0, 1, 2) if not c_bits_ok(counts(rep3_mutant(e, kind)), (2, 3, 4))]
        report("S6", f"mutant '{kind}': the logical circuit is wrong for some error", bool(logical_wrong),
               f"wrong for errors {logical_wrong}")
        out = os.path.join(work, f"s6_{kind}")
        shutil.rmtree(out, ignore_errors=True)
        succ = {2: 0, 3: 0, 4: 0}
        items = [(f"r{'n' if e is None else e}", rep3_mutant(e, kind),
                  dict(success=succ, off_circuit=FF.rep3_lookup(e, feed_forward=False))) for e in (None, 0, 1, 2)]
        try:
            FF.build(Device.load(devdir), items, out, shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24))
        except FF.Refused as e:
            report("S6", f"mutant '{kind}': refused by build", True, str(e)[:70])
            continue
        check_job(out, "S6", f"mutant '{kind}' compiled", expect_ok=False)
        # psf-os's own path: run_local and cards
        FF.run_local(out, seed=20261118)
        cards = FF.cards(out)
        low = {t: round(c["ff"]["share"], 3) for t, c in cards.items() if c["ff"]["share"] < 1}
        report("S6", f"mutant '{kind}': psf-os run_local + cards show a share below 1", bool(low), str(low))


def main():
    devdir, work = sys.argv[1:3]
    os.makedirs(work, exist_ok=True)
    s1()
    out = s2(devdir, work)
    s3()
    s4(devdir, work)
    bc = s5(out)
    s6(devdir, work)
    n_bad = sum(not r["ok"] for r in RESULTS)
    print(f"{len(RESULTS)} checks, {n_bad} FAIL")
    if len(sys.argv) > 3:
        json.dump(dict(results=RESULTS, broadcast=bc), open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
