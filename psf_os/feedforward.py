"""Feed-forward jobs (2026-11-16; owner: the test that no after-the-fact processing can reach). A new job kind beside
user2d: circuits with mid-circuit measurements and conditional Pauli gates applied in real time. user2d is not changed
and still refuses any gate after a measurement.

Accepted (narrow on purpose; anything else is refused with the reason; form "pauli", the default):
  * qubits 0 .. n-1 on a line: two-qubit gates are CZ between neighbours (k, k+1) only;
  * one-qubit gates (h, x, y, z, s, sdg, t, tdg, sx, sxdg, rx, ry, rz, p, id);
  * measurements anywhere, one classical register; each classical bit written once (a definite record per shot);
  * `if_test` blocks on one classical bit already written, value 0 or 1, holding only X or Z on one qubit, no else;
  * (2026-11-18, branch register-cond-2026-11-18) `if_test` on a whole ClassicalRegister of at most 32 bits, every bit
    already written, against a value 0 <= value < 2^width: a table look-up on one register is several such tests with
    different values. The circuit may then hold several classical registers that partition its bits in order (no
    bit in two registers). Expressions and arithmetic stay refused (IBM runs neither). The block rules do not change.
    Every job with conditionals records its broadcasts in the manifest (`broadcast_record`), never enforced;
  * `delay` on one qubit at the top level (2026-11-17: to hold a schedule, e.g. notes/next_2026-11-17/pin_delay.py in
    the notebook); kept through placement, compilation, the off and floor arms, and the identity noiselessly; refused
    inside a conditional block.
Form "blocks" (2026-11-16 evening, for repeat-until-success): the blocks may hold a fixed sub-circuit of one-qubit
gates and neighbour CZ; never a measurement, a reset or another conditional inside a block (IBM's dynamic circuits do
not run those: "Nested conditionals are not allowed", "Having reset or measurements inside conditionals is not
supported"; measure after the block, unconditionally, and reset by a conditional X). CY and CRY are accepted and
lowered to CZ and one-qubit gates at the entrance (`lower`). `rus_chain(L, theta, m)` writes the neuron chain so.

Placement: the n-site chain by rule C (psf_os.chain.place: the coupling graph and the failed marks only, no error
number), logical qubit k on chain[k]. Compilation: the map-free Target with the control-flow op if_else added (no
property, no duration), Qiskit's level-1 pass manager, layout fixed, no routing (as compile.RuleCompiler).

Placement given from outside (2026-11-17): build(..., chains={tag: [qubits]}) puts an item on a given chain, accepted
only if it is a joined line of usable qubits on working couplers (given_chain); otherwise rule C as above.
Bundles (2026-11-17, the packing test): build_bundle puts independent sub-items side by side on one rule-C chain
(bundle_place: one site skipped between items, more while a coupler would join two), one ff, one off and one floor
circuit for all of them and one classical register (each sub-item's bits at an offset); accepted only if no
instruction spans two sub-items and each sub-item compiles exactly as it does alone on the same qubits; cards() gives
one card per sub-item from the marginal counts of its bits; run_local simulates independent parts separately.

Arms of one item, all in the same job, and one ring sentinel per job:
  ff     the circuit;
  off    the same circuit with every conditional block removed (the pulses are the ff arm's but the conditionals);
  floor  the compiled ff circuit with every sx and x removed, also inside the conditional blocks: CZ, RZ, the
         measurements and the (now empty or diagonal) conditionals stay, so every bit is ideally 0 (readout and idle).
No mirror: a mirror needs the inverse of the circuit, and a measurement has none (README of
notes/feedforward_2026-11-16 in the notebook says what off and floor cover instead).

Answer card per item (every shot counted, none discarded): the success share (the bits named by `success` equal their
values), its binomial 2 SE, the bound the item states (e.g. 0.75 for the teleport test), (share - bound) / SE; the off
arm's share; the floor's all-zero share and per-bit share of 1. No verdict words here: those are a declaration's.

    from psf_os import feedforward as FF
    items = [(f"tp{k}", FF.teleport_test(k), dict(success={k: 0}, bound=0.75)) for k in (1, 2, 4)]
    rep = FF.build(dev, items, "OUT", shots=4000)
    FF.run_local("OUT")              # noiseless Aer, writes raw_counts.json as a device run would (dry runs only)
    cards = FF.cards("OUT")
"""
from __future__ import annotations

import json
import math
import os
import re

KIND = "feedforward"
ONE_QUBIT = ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "id")
CONDITIONAL = ("x", "z")
NON_DIAGONAL_COMPILED = ("sx", "x")
LOCAL_MAX_QUBITS = 24          # run_local: statevector up to this many touched qubits for non-Clifford circuits


class Refused(ValueError):
    """The circuit is not a feed-forward circuit this job kind reads; the message says why."""


# ------------------------------------------------------------------------------------------------ the test circuit

def teleport_test(k, feed_forward=True):
    """k one-bit teleports along a line, then a non-Clifford T (notebook notes/feedforward_2026-11-16/ff_desk.py, here
    with one classical register c: c[0..k-1] the mid-circuit bits, c[k] the final bit). Qubit i+1 starts in |+>,
    CZ(i, i+1), qubit i is measured in the X basis and, if 1, X is applied to qubit i+1 at once. Each hop applies H to
    the carried state, so psi = H^k T^dag |+> arrives as T^dag |+>; T and H turn it into |0>. Ideal: every shot ends in
    c[k] = 0 with feed-forward; without it the X byproducts sit before the T and no fixed rule on the recorded bits
    gets the final bit right in more than 3/4 of the shots."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    if k < 1:
        raise ValueError("k >= 1")
    q, c = QuantumRegister(k + 1, "q"), ClassicalRegister(k + 1, "c")
    qc = QuantumCircuit(q, c)
    qc.h(0)
    qc.tdg(0)
    for _ in range(k):
        qc.h(0)
    for i in range(k):
        qc.h(i + 1)
        qc.cz(i, i + 1)
        qc.h(i)
        qc.measure(i, i)
        if feed_forward:
            with qc.if_test((c[i], 1)):
                qc.x(i + 1)
    qc.t(k)
    qc.h(k)
    qc.measure(k, k)
    return qc


def rep3_lookup(error=None, feed_forward=True):
    """The distance-3 bit-flip repetition code with a look-up decoder in real time (2026-11-18, the register
    condition's test). Line d0 a0 d1 a1 d2 (qubits 0 .. 4). d0 holds T^dag|+>, encoded into d0 d1 d2 through the
    ancillas (CX(d0, a0) CX(a0, d1) CX(d0, a0), likewise d1 -> d2: neighbours only, the ancillas back to 0); then X on
    data `error` (None, 0, 1 or 2); the parities a0 = d0 + d1, a1 = d1 + d2 measured into register s (s[0], s[1]); the
    look-up: s == 1 -> X d0, s == 3 -> X d1, s == 2 -> X d2 (three tests on one register); the ancillas reset by a
    conditional X on their own bit (single-bit tests); decoded back onto d0; then T and H on d0: noiselessly every shot
    reads c = 000 (c[0] = d0, c[1] = d1, c[2] = d2). feed_forward=False removes the look-up and keeps the ancilla
    resets (an off arm to give explicitly): an X on d0 then reaches d0 before the T, so c[0] is right in half the
    shots, and an X on d1 or d2 ends on c[1] or c[2]. The default off arm (strip_conditionals) removes the resets too,
    and an ancilla left at 1 then also acts in the decoding (with an X on d1 it happens to undo it)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    if error not in (None, 0, 1, 2):
        raise ValueError("error is None, 0, 1 or 2")
    q = QuantumRegister(5, "q")
    s, c = ClassicalRegister(2, "s"), ClassicalRegister(3, "c")
    qc = QuantumCircuit(q, s, c)
    d, a = (0, 2, 4), (1, 3)

    def cx(ctl, tgt):
        qc.h(tgt)
        qc.cz(ctl, tgt)
        qc.h(tgt)

    def copy(i):                         # d[i+1] ^= d[i] through a[i], a[i] back to 0
        cx(d[i], a[i])
        cx(a[i], d[i + 1])
        cx(d[i], a[i])
    qc.h(d[0])
    qc.tdg(d[0])
    copy(0)
    copy(1)
    if error is not None:
        qc.x(d[error])
    cx(d[0], a[0])
    cx(d[1], a[0])
    cx(d[1], a[1])
    cx(d[2], a[1])
    qc.measure(a[0], s[0])
    qc.measure(a[1], s[1])
    if feed_forward:
        for value, k in ((1, 0), (3, 1), (2, 2)):
            with qc.if_test((s, value)):
                qc.x(d[k])
    for i in (0, 1):
        with qc.if_test((s[i], 1)):
            qc.x(a[i])
    copy(1)
    copy(0)
    qc.t(d[0])
    qc.h(d[0])
    for i in range(3):
        qc.measure(d[i], c[i])
    return qc


def rus_q(theta):
    """The repeat-until-success neuron's angle: tan q = tan^2 theta (Cao, Guerreschi, Aspuru-Guzik 2017)."""
    return math.atan(math.tan(theta) ** 2)


def _rus_attempt(qc, a, o, theta, ctrl):
    if ctrl is None:
        qc.ry(2 * theta, a)
    else:
        qc.cry(2 * theta, ctrl, a)
    qc.cy(a, o)
    qc.sdg(a)
    if ctrl is None:
        qc.ry(-2 * theta, a)
    else:
        qc.cry(-2 * theta, ctrl, a)


def rus_chain(L, theta, m=3, feed_forward=True):
    """L repeat-until-success neurons in a row on a line a0, o0, a1, o1, ... (ancilla 2j, output 2j+1; neuron j > 0
    has its ancilla angle switched on by output j-1), m tries per neuron (notebook notes/feedforward_2026-11-16/
    rus_neuron_desk.py), written for IBM's dynamic circuits: the conditional blocks hold gates only (undo RY(pi/2) on
    the output, reset the ancilla by X, the next attempt) and every try's measurement sits after its block,
    unconditionally (after a success the ancilla is |0> and reads 0 again). One register c: neuron j's tries in
    c[j m .. j m + m - 1], the last output in c[L m]. Returns (circuit, spec) with the exact value of P(last output 1)
    when every neuron succeeds, (sin^2 q)^L."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    if L < 1 or m < 1:
        raise ValueError("L >= 1 and m >= 1")
    q, c = QuantumRegister(2 * L, "q"), ClassicalRegister(L * m + 1, "c")
    qc = QuantumCircuit(q, c)
    for j in range(L):
        a, o, ctrl = 2 * j, 2 * j + 1, (None if j == 0 else 2 * j - 1)
        _rus_attempt(qc, a, o, theta, ctrl)
        qc.measure(a, j * m)
        for r in range(1, m):
            if feed_forward:
                with qc.if_test((c[j * m + r - 1], 1)):
                    qc.ry(math.pi / 2, o)
                    qc.x(a)
                    _rus_attempt(qc, a, o, theta, ctrl)
            qc.measure(a, j * m + r)
    qc.measure(2 * L - 1, L * m)
    spec = dict(kind="rus", L=L, theta=theta, m=m, out_bit=L * m,
                tries=[list(range(j * m, j * m + m)) for j in range(L)], exact=math.sin(rus_q(theta)) ** (2 * L),
                form="blocks")
    return qc, spec


def rus_exact_statevector(L, theta):
    """P(last output 1) of the ideal chain (every neuron succeeds) by statevector: output 0 turned by RY(2q), output j
    by a controlled RY(2q) from output j-1."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(L)
    qc.ry(2 * rus_q(theta), 0)
    for j in range(1, L):
        qc.cry(2 * rus_q(theta), j - 1, j)
    return float(Statevector(qc).probabilities([L - 1])[1])


# ------------------------------------------------------------------------------------------------ analysis

def _cond_bit(op, qc):
    cond = getattr(op, "condition", None)
    if cond is None or len(cond) != 2:
        raise Refused("a conditional must test one classical bit")
    bit, val = cond
    from qiskit.circuit import Clbit
    if not isinstance(bit, Clbit):
        raise Refused("a conditional must test one classical bit, not a register or an expression")
    if int(val) not in (0, 1):
        raise Refused("a conditional bit is compared with 0 or 1")
    return qc.find_bit(bit).index, int(val)


REGISTER_MAX_BITS = 32          # IBM, Execute dynamic circuits: "The operand used in an if_test statement must be 32
                                # or fewer bits"


def condition_of(op, qc):
    """(bit indices, value, register or None) of a conditional's test (2026-11-18, register conditions): one classical
    bit compared with 0 or 1 (as before, by _cond_bit), or a whole ClassicalRegister of at most 32 bits compared with a
    value 0 <= value < 2^width (bits in the register's order, bit 0 the least significant). An expression, arithmetic
    or anything else is refused (IBM: "Arithmetic operations are not supported")."""
    from qiskit.circuit import ClassicalRegister, Clbit
    cond = getattr(op, "condition", None)
    if isinstance(cond, tuple) and len(cond) == 2 and isinstance(cond[0], Clbit):
        b, v = _cond_bit(op, qc)
        return (b,), v, None
    if not (isinstance(cond, tuple) and len(cond) == 2 and isinstance(cond[0], ClassicalRegister)):
        raise Refused("a conditional tests one classical bit or one classical register against a value; expressions "
                      "and arithmetic are not run on IBM devices")
    reg, val = cond
    w = len(reg)
    if not 1 <= w <= REGISTER_MAX_BITS:
        raise Refused(f"a conditional's register '{reg.name}' has {w} bits; IBM takes at most {REGISTER_MAX_BITS}")
    try:
        val = int(val)
    except (TypeError, ValueError):
        raise Refused(f"register '{reg.name}' is compared with {val!r}, not an integer") from None
    if not 0 <= val < 2 ** w:
        raise Refused(f"register '{reg.name}' ({w} bits) is compared with {val}, outside 0 .. {2 ** w - 1}")
    return tuple(qc.find_bit(b).index for b in reg), val, reg


def _tests_a_register(qc):
    return any(ins.operation.name == "if_else" and type(ins.operation.condition[0]).__name__ == "ClassicalRegister"
               for ins in qc.data)


def _check_registers(qc, many=False):
    """One register holding every bit (as before), or (2026-11-18) several registers that partition the bits in order:
    disjoint, every bit in exactly one, and the circuit's bit order the registers' bits one after the other. Then a
    counts key read with its spaces removed has clbit k at position -1-k, on Aer and in IBM's joined data alike, and
    no bit is reported twice (an aliasing register would be). Several registers are accepted only when a conditional
    tests a register (or many=True: an explicit off arm of such an item); every other circuit is refused as before."""
    if len(qc.cregs) == 1:
        if qc.num_clbits != len(qc.cregs[0]):
            raise Refused("every classical bit must belong to the one register")
        return
    if not qc.cregs or not (many or _tests_a_register(qc)):
        raise Refused(f"one classical register holding every measured bit (got {len(qc.cregs)})")
    order = [b for r in qc.cregs for b in r]
    if len(set(order)) != len(order):
        raise Refused("classical registers overlap (a bit in two registers would be reported twice)")
    if order != list(qc.clbits):
        raise Refused("the classical registers must hold every bit, one register after another in the circuit's bit "
                      "order")


FORMS = ("pauli", "blocks")
DELAY_IN_BLOCK = ("a delay inside a conditional block is not read (its timing would depend on the branch); put "
                  "delays at the top level only")
IBM_RULES = ("IBM, Execute dynamic circuits (2026): 'Nested conditionals are not allowed'; 'Having reset or "
             "measurements inside conditionals is not supported'")


def _check_block(body, qmap):
    """The 'blocks' form: one-qubit gates and neighbour CZ only (no measurement, reset or nested conditional)."""
    cz, gates = 0, 0
    for bi in body.data:
        name = bi.operation.name
        qs = [qmap[body.find_bit(x).index] for x in bi.qubits]
        if name == "barrier":
            continue
        if name == "if_else":
            raise Refused(f"a conditional inside a conditional is not run on IBM devices ({IBM_RULES})")
        if name == "delay":
            raise Refused(DELAY_IN_BLOCK)
        if name in ("measure", "reset"):
            raise Refused(f"'{name}' inside a conditional is not run on IBM devices ({IBM_RULES}); measure after "
                          "the block, unconditionally")
        if len(qs) == 1 and name in ONE_QUBIT:
            gates += 1
        elif len(qs) == 2 and name == "cz" and abs(qs[0] - qs[1]) == 1:
            cz += 1
        else:
            raise Refused(f"a conditional block holds one-qubit gates and neighbour CZ only (got '{name}' on {qs})")
    return cz, gates


def analyse(qc, form="pauli", many_registers=False):
    """dict(n, clbits, mid, conditionals, cz, cz_in_blocks) or Refused. Checks the accepted form (module docstring):
    form "pauli" (conditional blocks hold one X or Z) or "blocks" (fixed sub-circuits of one-qubit gates and
    neighbour CZ; no measurement, reset or nesting inside, as IBM's dynamic circuits require). CY and CRY are lowered
    to CZ and one-qubit gates first (`lower`). With a register test (2026-11-18) also register_conditionals and
    condition_bits; many_registers=True accepts several registers without one (an explicit off arm)."""
    if form not in FORMS:
        raise Refused(f"form {form!r} is not one of {FORMS}")
    for ins in qc.data:
        # 2026-11-18: a register test with a value out of range cannot be rebuilt by `lower` (Qiskit refuses it), so
        # every test is read first; for one-bit tests this is the same reading as below
        if ins.operation.name == "if_else":
            condition_of(ins.operation, qc)
    qc = lower(qc)
    cz_blocks = 0
    _check_registers(qc, many_registers)
    written, measured_last, cz, conds, mids = set(), {}, 0, 0, 0
    reg_conds, cond_bits = 0, 0
    for pos, ins in enumerate(qc.data):
        op = ins.operation
        name = op.name
        qs = [qc.find_bit(x).index for x in ins.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            b = qc.find_bit(ins.clbits[0]).index
            if b in written:
                raise Refused(f"classical bit {b} is written twice (each bit must be one record per shot)")
            written.add(b)
            measured_last[qs[0]] = pos
            continue
        if name == "if_else":
            bits, _, reg = condition_of(op, qc)
            if reg is None:
                if bits[0] not in written:
                    raise Refused(f"a conditional tests bit {bits[0]} before it is measured")
            else:
                late = [b for b in bits if b not in written]
                if late:
                    raise Refused(f"a conditional tests register '{reg.name}' before its bits {late} are measured")
                reg_conds += 1
            cond_bits += len(bits)
            if len(op.blocks) != 1 or op.blocks[0] is None:
                raise Refused("a conditional has no else branch here")
            body = op.blocks[0]
            if form == "pauli":
                for bi in body.data:
                    if bi.operation.name == "delay":
                        raise Refused(DELAY_IN_BLOCK)
                    if bi.operation.name not in CONDITIONAL or len(bi.qubits) != 1:
                        raise Refused(f"a conditional block holds only X or Z on one qubit (got "
                                      f"'{bi.operation.name}'; form 'blocks' takes fixed sub-circuits)")
            else:
                c, _ = _check_block(body, {j: qs[j] for j in range(len(qs))})
                cz_blocks += c
            conds += 1
            continue
        if getattr(op, "condition", None) is not None:
            raise Refused("conditions are written as if_test blocks")
        if name == "delay":
            # a top-level delay holds a schedule; it is the identity noiselessly and is not counted as a gate
            if len(qs) != 1:
                raise Refused("a delay acts on one qubit")
            continue
        if name in ("reset", "initialize", "while_loop", "for_loop", "switch_case", "store", "box"):
            raise Refused(f"'{name}' is not read by the feed-forward job")
        if len(qs) == 1:
            if name not in ONE_QUBIT:
                raise Refused(f"one-qubit gate '{name}' is not in the accepted set {ONE_QUBIT}")
            continue
        if len(qs) == 2:
            if name != "cz":
                raise Refused(f"two-qubit gates are CZ only (got '{name}'; write CX as H CZ H)")
            if abs(qs[0] - qs[1]) != 1:
                raise Refused(f"CZ({qs[0]}, {qs[1]}): only neighbours on the line (k, k+1)")
            cz += 1
            continue
        raise Refused(f"'{name}' on {len(qs)} qubits is not read")
    if len(written) != qc.num_clbits:
        raise Refused(f"{qc.num_clbits - len(written)} classical bits are never written")
    ops = [ins.operation.name for ins in qc.data]
    last_op = max((i for i, nm in enumerate(ops) if nm not in ("measure", "barrier", "delay")), default=-1)
    mids = sum(1 for i, nm in enumerate(ops) if nm == "measure" and i < last_op)   # an operation follows it
    out = dict(n=qc.num_qubits, clbits=qc.num_clbits, mid=mids, conditionals=conds, cz=cz, cz_in_blocks=cz_blocks)
    if reg_conds:
        # 2026-11-18: only when a register is tested, so single-bit items keep exactly the fields they had
        out.update(register_conditionals=reg_conds, condition_bits=cond_bits)
    return out


def lower(qc):
    """CY and CRY written as CZ and one-qubit gates, also inside conditional blocks (exact identities, checked in the
    tests): CY(c, t) = Sdg(t) H(t) CZ(c, t) H(t) S(t); CRY(theta)(c, t) = RY(theta/2)(t) CX RY(-theta/2)(t) CX with
    each CX(c, t) = H(t) CZ(c, t) H(t). Everything else is kept."""
    out = qc.copy_empty_like()
    for ins in qc.data:
        op, qs = ins.operation, ins.qubits
        if op.name == "if_else":
            out.append(op.replace_blocks([None if b is None else lower(b) for b in op.blocks]), qs, ins.clbits)
        elif op.name == "cy":
            c, t = qs
            out.sdg(t); out.h(t); out.cz(c, t); out.h(t); out.s(t)          # noqa: E702
        elif op.name == "cry":
            c, t = qs
            th = op.params[0]
            out.ry(th / 2, t)
            out.h(t); out.cz(c, t); out.h(t)                                 # noqa: E702
            out.ry(-th / 2, t)
            out.h(t); out.cz(c, t); out.h(t)                                 # noqa: E702
        else:
            out.append(op, qs, ins.clbits)
    return out


def strip_conditionals(qc):
    """The off arm: the same circuit without its conditional blocks."""
    out = qc.copy_empty_like()
    for ins in qc.data:
        if ins.operation.name != "if_else":
            out.append(ins.operation, ins.qubits, ins.clbits)
    return out


# ------------------------------------------------------------------------------------------------ placement, compile

def target(dev):
    """The map-free Target (psf_os.device) with the control-flow op if_else: no property, no duration."""
    from qiskit.circuit import IfElseOp
    from psf_os.device import map_free_target
    t = map_free_target(dev)
    t.add_instruction(IfElseOp, name="if_else")
    return t


def place(dev, n):
    from psf_os import chain as CH
    try:
        return CH.place(dev, n)
    except CH.Refused as e:
        raise Refused(f"chain of {n}: {e}") from None


def on_device(qc, chain, width):
    """Logical qubit k on chain[k] of a width-qubit device circuit (classical register kept, blocks remapped)."""
    from qiskit import QuantumCircuit, QuantumRegister
    out = QuantumCircuit(QuantumRegister(width, "q"), *qc.cregs)
    out.compose(qc, qubits=list(chain), clbits=list(range(qc.num_clbits)), inplace=True)
    return out


class Compiler:
    def __init__(self, dev):
        from qiskit.transpiler import generate_preset_pass_manager
        self.dev = dev
        self.target = target(dev)
        self.pm = generate_preset_pass_manager(1, target=self.target, initial_layout=list(range(dev.num_qubits)),
                                               routing_method="none", seed_transpiler=0)

    def run(self, qc):
        return self.pm.run(qc)


def _map_blocks(qc, fn):
    """A copy of qc with fn(instruction) -> keep? applied at top level and inside every control-flow block."""
    out = qc.copy_empty_like()
    for ins in qc.data:
        op = ins.operation
        if op.name == "if_else":
            nb = [None if b is None else _map_blocks(b, fn) for b in op.blocks]
            op = op.replace_blocks(nb)
            out.append(op, ins.qubits, ins.clbits)
        elif fn(ins):
            out.append(op, ins.qubits, ins.clbits)
    return out


def floor_of(compiled):
    """The compiled circuit with every sx and x removed, also inside conditional blocks (ideal: every bit 0)."""
    return _map_blocks(compiled, lambda ins: ins.operation.name not in NON_DIAGONAL_COMPILED)


def walk(qc):
    """Every instruction, top level and inside blocks: (name, qubit indices of the top circuit)."""
    out = []

    def rec(c, qmap):
        for ins in c.data:
            qs = [qmap[c.find_bit(x).index] for x in ins.qubits]
            out.append((ins.operation.name, tuple(qs)))
            if ins.operation.name == "if_else":
                for b in ins.operation.blocks:
                    if b is not None:
                        rec(b, {j: qs[j] for j in range(len(qs))})
    rec(qc, {j: j for j in range(qc.num_qubits)})
    return out


BROADCAST_NOTE = ("recorded, not enforced (2026-11-18): IBM, Execute dynamic circuits: the limit 'is a function of the "
                  "number of broadcasts and the number of broadcasted bits in a job (not a circuit)'; 'if you "
                  "broadcast 60 bits each time, then the job can have approximately 300 broadcasts ... If you "
                  "broadcast just one bit each time, however, then the job can have 2400 broadcasts'. conditionals / "
                  "condition_bits: every if_else (top level and in blocks) and the bits its test reads (1 for a bit, "
                  "the width for a register), an upper bound. broadcasts / broadcast_bits: IBM's count as read by the "
                  "workplace VS (notebook notes/seats_2026-11-18/broadcasts.py): a test is a new broadcast when a bit "
                  "it reads was measured after that bit was last broadcast, and then all its bits are broadcast; "
                  "tests of unchanged bits add nothing (our reading of the text, not IBM's code)")


def broadcast_record(circuits):
    """Per circuit and for the job: the number of conditionals and the bits their tests read (an upper bound), and
    the broadcasts and broadcast bits by IBM's description as read in BROADCAST_NOTE (2026-11-18, after the workplace
    VS's check: the look-up reads one register five times but broadcasts it once). A record for the manifest; nothing
    is refused by it."""
    def tested(op, cc):
        cb = op.condition[0]
        return [cc.find_bit(x).index for x in (cb if type(cb).__name__ == "ClassicalRegister" else [cb])]

    def count(cc):
        n = bits = 0
        for ins in cc.data:
            if ins.operation.name == "if_else":
                n += 1
                bits += len(tested(ins.operation, cc))
                for b in ins.operation.blocks:
                    if b is not None:
                        bn, bb = count(b)
                        n, bits = n + bn, bits + bb
        return n, bits

    def ibm(cc):
        stamp, sent, bc, bb = {}, {}, 0, 0
        for t, ins in enumerate(cc.data, 1):
            if ins.operation.name == "measure":
                stamp[cc.find_bit(ins.clbits[0]).index] = t
            elif ins.operation.name == "if_else":
                bits = tested(ins.operation, cc)
                if any(stamp.get(b, 0) > sent.get(b, -1) for b in bits):
                    bc, bb = bc + 1, bb + len(bits)
                    for b in bits:
                        sent[b] = t
        return bc, bb
    per = []
    for c in circuits:
        n, bits = count(c)
        bc, bb = ibm(c)
        per.append(dict(conditionals=n, condition_bits=bits, broadcasts=bc, broadcast_bits=bb))
    return dict(per_circuit=per, conditionals=sum(p["conditionals"] for p in per),
                condition_bits=sum(p["condition_bits"] for p in per),
                broadcasts=sum(p["broadcasts"] for p in per), broadcast_bits=sum(p["broadcast_bits"] for p in per),
                max_conditionals_per_circuit=max((p["conditionals"] for p in per), default=0),
                max_broadcasts_per_circuit=max((p["broadcasts"] for p in per), default=0), note=BROADCAST_NOTE)


def cz_count(qc):
    return sum(1 for n, _ in walk(qc) if n == "cz")


def checks(dev, compiled, chain):
    """The acceptance of a compiled arm: every two-qubit gate on a working chain coupler, every touched qubit on the
    chain and usable, measurements kept."""
    work = {tuple(sorted(e)) for e in dev.working_couplers()}
    links = {tuple(sorted(e)) for e in zip(chain, chain[1:])}
    touched = {q for _, qs in walk(compiled) for q in qs}
    two = [qs for n, qs in walk(compiled) if len(qs) == 2 and n not in ("barrier",)]
    return dict(on_chain=touched <= set(chain), qubits_ok=all(dev.qubit_ok(q) for q in touched),
                couplers_on_chain=all(tuple(sorted(p)) in links for p in two),
                couplers_working=all(tuple(sorted(p)) in work for p in two),
                measures=sum(1 for n, _ in walk(compiled) if n == "measure") > 0)


# ------------------------------------------------------------------------------------------------ build

def _sentinel(dev, runs, shots, spec, ring=40, cover=None):
    """The job's ring sentinel. cover (after-2026-11-20; default None: the ring as before): the test's physical
    qubits; the ring rule then prefers, at the same length, an outline holding all of them (place.rule_choice prefer).
    Returns dict(asked, covered, ring) when cover is given (the qubits of cover on the sentinel's ring), else None."""
    import numpy as np
    from psf_os import circuits as C
    from psf_os import sentinel
    from psf_os.compile import RuleCompiler, measured_qubits
    rc = RuleCompiler(dev, prefer=cover)
    mq = measured_qubits(rc.compile(C.model_circuit(list(np.zeros(C.n_model_angles(ring, 1))), ring, 1)))
    user_ring = [int(mq[j]) for j in range(ring)]
    sentinel.build(dev, runs, user_ring, shots, rc, spec=spec)
    if not cover:
        return None
    asked = sorted({int(q) for q in cover})
    return dict(asked=asked, covered=sorted(set(asked) & set(user_ring)), all=set(asked) <= set(user_ring))


def sentinel_ring(dev, ring=40, cover=None):
    """The physical qubits of the job's ring sentinel in ring order (as _sentinel places it; cover as there)."""
    import numpy as np
    from psf_os import circuits as C
    from psf_os.compile import RuleCompiler, measured_qubits
    rc = RuleCompiler(dev, prefer=cover)
    mq = measured_qubits(rc.compile(C.model_circuit(list(np.zeros(C.n_model_angles(ring, 1))), ring, 1)))
    return [int(mq[j]) for j in range(ring)]


def ring_chains(dev, n, ring=40):
    """after-2026-11-20 (home's candidate 4, the other way round): every chain of n consecutive sites of the default
    sentinel ring whose neighbours are joined by working couplers and whose qubits are usable, in ring order of the
    first site. A test placed on one (build's chains={tag: chain}) is watched by the sentinel on its own qubits.
    Rule C's chains on a Heron start at a corner (qubit 0 has one coupler, qubit 1 two), where no ring can pass."""
    r = sentinel_ring(dev, ring)
    work = {tuple(sorted(e)) for e in dev.working_couplers()}
    out = []
    for s in range(len(r)):
        ch = [r[(s + k) % len(r)] for k in range(n)]
        if len(set(ch)) == n and all(dev.qubit_ok(q) for q in ch) and all(
                tuple(sorted((ch[k], ch[k + 1]))) in work for k in range(n - 1)):
            out.append(ch)
    return out


def given_chain(dev, chain, n):
    """A placement given from outside (2026-11-17: the single job of the packing test puts each item on the qubits
    the packed job gives it). Accepted only as rule C's chains are: n distinct usable qubits, each consecutive pair a
    working coupler of the device record (the coupling map and the failed marks only; no error number is read)."""
    chain = [int(q) for q in chain]
    work = {tuple(sorted(e)) for e in dev.working_couplers()}
    if len(chain) != n:
        raise Refused(f"given chain has {len(chain)} qubits, the circuit {n}")
    if len(set(chain)) != n or any(not 0 <= q < dev.num_qubits for q in chain):
        raise Refused(f"given chain must hold {n} distinct qubits of the device: {chain}")
    bad = [q for q in chain if not dev.qubit_ok(q)]
    if bad:
        raise Refused(f"given chain holds failed qubits {bad}")
    gaps = [p for p in zip(chain, chain[1:]) if tuple(sorted(p)) not in work]
    if gaps:
        raise Refused(f"given chain is not joined by working couplers at {gaps}")
    return dict(chain=chain, rule="given", start=None)


def _prepare(tag, qc, spec):
    """One item as the entrance reads it: (analysis, lowered circuit, lowered off arm, success, spec)."""
    spec = dict(spec or {})
    try:
        a = analyse(qc, form=spec.get("form", "pauli"))
    except Refused as e:
        raise Refused(f"{tag}: {e}") from None
    qc = lower(qc)
    if spec.get("kind") == "rus":
        bits = [b for t in spec["tries"] for b in t] + [spec["out_bit"]]
        if any(not 0 <= b < a["clbits"] for b in bits) or len(set(bits)) != len(bits):
            raise Refused(f"{tag}: the rus spec's bits do not fit the circuit")
        succ = {}
    else:
        succ = {int(b): int(v) for b, v in spec.get("success", {}).items()}
        if not succ or any(b >= a["clbits"] or v not in (0, 1) for b, v in succ.items()):
            raise Refused(f"{tag}: success must name classical bits of the circuit and their values 0 / 1")
    off = spec.get("off_circuit")
    if off is not None:
        # an explicit off arm (2026-11-17): for a circuit without conditionals (e.g. repeat-until-success written
        # coherently) the comparison without control is a different circuit (one try), given by the caller
        try:
            ao = analyse(off, form=spec.get("form", "pauli"), many_registers=len(qc.cregs) > 1)
        except Refused as e:
            raise Refused(f"{tag} (off arm): {e}") from None
        if (ao["n"], ao["clbits"]) != (a["n"], a["clbits"]):
            raise Refused(f"{tag}: the off arm has {ao['n']} qubits / {ao['clbits']} bits, the circuit "
                          f"{a['n']} / {a['clbits']}")
        off = lower(off)
    else:
        off = strip_conditionals(qc)
    return a, qc, off, succ, spec


def _item_extra(tag, a, succ, spec):
    """The per-item fields of a record (the same for a single item and for a sub-item of a bundle)."""
    extra = dict(success={str(b): v for b, v in succ.items()}, bound=spec.get("bound"), clbits=a["clbits"],
                 mid=a["mid"], conditionals=a["conditionals"], logical_cz=a["cz"])
    if spec.get("off_circuit") is not None:
        extra["off_explicit"] = True
    if spec.get("kind") == "rus":
        extra["rus"] = {k: spec[k] for k in ("L", "theta", "m", "out_bit", "tries", "exact")}
        extra["logical_cz_in_blocks"] = a["cz_in_blocks"]
    if a.get("register_conditionals"):
        extra.update(register_conditionals=a["register_conditionals"], condition_bits=a["condition_bits"])
    return extra


CHECKS_FILE = "checks.json"


def _check_row(tag, qc, off, arms):
    """One item's record-only checks (after-2026-11-20): CZ written / compiled per arm, the footprint as sent and as
    written (psf_os.footprint.footprint_or_none: counted=False above its bounds)."""
    import sys
    from psf_os.footprint import footprint_or_none
    written = {"ff": cz_count(qc), "off": cz_count(off), "floor": cz_count(qc)}
    got = {arm: cz_count(c) for arm, c in arms.items()}
    row = dict(cz_written=written, cz_compiled=got, cz_same=got == written,
               footprint_sent={arm: footprint_or_none(arms[arm]) for arm in ("ff", "off")},
               footprint_written=footprint_or_none(qc))
    if got != written:
        print(f"note: {tag}: compiled CZ {got} differ from the circuit as written {written} (recorded in "
              f"{CHECKS_FILE}; strict_cz=True refuses)", file=sys.stderr)
    return row


def _write_checks(out, rows):
    json.dump(dict(note="record-only checks (psf-os after-2026-11-20): never part of the job; the job and its "
                        "records are as without them", items=rows),
              open(os.path.join(out, CHECKS_FILE), "w"), indent=1)


# park-default-2026-11-21: new builds park the qubits that wait in |1> (psf_os.footprint.park_waits); a build that
# remakes a recorded night must not (its records hold no "park" key: read as False). PARK_DEFAULT is what build()
# uses when park=None; park_default(value) sets it for a block (psf_os.loop.verify uses it).
PARK_DEFAULT = True
# release-0.3.1: parking the likely value (footprint.park_likely, behind a partial fence) is build()'s default too.
# A recorded build without the "park_likely" key was made without it (read as False).
PARK_LIKELY_DEFAULT = True


class park_default:
    """Context manager: build()'s park default inside the block (park=None resolves to `value`) and its park_likely
    default (park_likely=None resolves to `likely`; when `likely` is not given, the same as `value`: park_default(False)
    builds as before 0.3.0)."""

    def __init__(self, value, likely=None):
        self.value = bool(value)
        self.likely = self.value if likely is None else bool(likely)

    def __enter__(self):
        global PARK_DEFAULT, PARK_LIKELY_DEFAULT
        self.saved = (PARK_DEFAULT, PARK_LIKELY_DEFAULT)
        PARK_DEFAULT, PARK_LIKELY_DEFAULT = self.value, self.likely
        return self

    def __exit__(self, *exc):
        global PARK_DEFAULT, PARK_LIKELY_DEFAULT
        PARK_DEFAULT, PARK_LIKELY_DEFAULT = self.saved
        return False


PARK_SINCE = (0, 3, 0)


def parks_by_default(version):
    """Whether the psf-os named by `version` parks by default: a version "X.Y.Z" (or "vX.Y.Z") at 0.3.0 or later.
    Anything else (an older version, a commit such as a164fc5) names a psf-os that did not park."""
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", str(version or "").strip())
    return bool(m) and tuple(int(x) for x in m.groups()) >= PARK_SINCE


def resolve_park(park, declared=None):
    """build()'s park setting (release-0.3.0, the safety guard): an explicit park wins; otherwise PARK_DEFAULT, refused
    when a declared psf-os (`declared`, else the environment variable PSF_OS_DECLARED) has the other default (e.g. a
    tool declared on a164fc5, which built IPE unparked, built with 0.3.0, which parks): the build must then say
    park=True or park=False."""
    if park is not None:
        return bool(park), "explicit"
    d = declared or os.environ.get("PSF_OS_DECLARED")
    if d and parks_by_default(d) != PARK_DEFAULT:
        raise Refused(f"the declared psf-os {d} {'parks' if parks_by_default(d) else 'does not park'} by default and "
                      f"this psf-os {'parks' if PARK_DEFAULT else 'does not'}: pass park=False (the form the declared "
                      "version built) or park=True explicitly")
    return PARK_DEFAULT, "default"


PARK_LIKELY_SINCE = (0, 3, 1)


def parks_likely_by_default(version):
    """Whether the psf-os named by `version` parks the likely value by default: a version at 0.3.1 or later (the
    released 0.3.1; its candidate a6170e5, also "0.3.1", had it off, and every build of it named park_likely
    explicitly). Anything else, a commit or an older version, did not."""
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", str(version or "").strip())
    return bool(m) and tuple(int(x) for x in m.groups()) >= PARK_LIKELY_SINCE


def resolve_park_likely(park_likely, declared=None):
    """(setting, how, guarded): an explicit park_likely (True, False or tags) wins; otherwise PARK_LIKELY_DEFAULT.
    guarded is True when a declared psf-os (`declared`, else PSF_OS_DECLARED) has the other default: build() then
    refuses any item that parking the likely value would change (an item it leaves alone is built as either version
    builds it)."""
    if park_likely is not None:
        return park_likely, "explicit", False
    d = declared or os.environ.get("PSF_OS_DECLARED")
    guarded = bool(d) and parks_likely_by_default(d) != PARK_LIKELY_DEFAULT
    return PARK_LIKELY_DEFAULT, "default", guarded


def park_likely_of_record(rec):
    """The park_likely setting a recorded build was made with: its "park_likely" key, False when absent."""
    v = (rec or {}).get("park_likely", False)
    return bool(v) if not isinstance(v, dict) else bool(v.get("park_likely", True))


def park_of_record(rec):
    """The park setting a recorded build was made with: its "park" key, False when absent (every build before
    park-default-2026-11-21); an item skipped by park_skip reads False."""
    v = (rec or {}).get("park", False)
    return bool(v) if not isinstance(v, dict) else bool(v.get("park", True))


def build(dev, items, out, shots=4000, max_seconds=60, sentinel_spec=None, chains=None, pack_fixed=False,
          strict_cz=False, footprint=False, sentinel_over_test=False, fence=False, record_checks="record",
          ff_only=(), park=None, declared=None, park_skip=(), park_likely=None, allow_long_delay=False):
    """items: [(tag, QuantumCircuit, spec)], spec either dict(success={clbit: value}, bound=float | None) (a success
    test such as teleport_test) or the dict rus_chain returns (kind "rus": out_bit, tries per neuron, exact, form
    "blocks"). One job: per item its ff, off and floor arms (for "rus" the off arm is also read post-selected:
    the "post" view, no extra circuit), one ring sentinel. Nothing is written if any item is refused. Returns the build report (also
    out/build.json).
    chains (2026-11-17): {tag: [physical qubits]} places those items on the given chains (given_chain checks them);
    items not named are placed by rule C as before.
    pack_fixed (v0.2 prototype, 2026-11-18; default False: the build above, unchanged): every item's ff arm stays its
    own circuit, while the fixed arms (off and floor: no conditional) of all items are packed into one off circuit and
    one floor circuit (build_pack_fixed).
    strict_cz (2026-11-20, after-2026-11-20; default False: as before, byte for byte): refused when an arm's compiled
    CZ count differs from the CZ count of the circuit as written for that arm (the compiler cancels a CX pair with
    nothing between them; on 2026-11-19 three of the repetition code's four items went 16 -> 14 CZ). A barrier keeps
    a pair (the level-1 pass manager does not cancel across barriers). With strict_cz the records also carry
    cz_written.
    footprint (2026-11-20; default False: as before): every record also carries wait_footprint, the waiting qubits not
    in a computational basis state at each mid-circuit measurement of the item as written (psf_os.footprint; logical
    qubits, noiseless, every branch), and the build report its max and total per item.
    sentinel_over_test (after-2026-11-20, home's candidate 4; default False: the ring as before, byte for byte): the
    ring sentinel's placement prefers, at the same ring length, an outline holding every item's chain (on 2026-11-19 the
    ring did not hold the qubits that carried IPE); the manifest and the build report record sentinel_cover (asked,
    covered, all). Nothing is refused when the cover is partial: it is recorded.
    fence (after-2026-11-20, the workplace VS's finding; default False: as before, byte for byte): every item's ff and
    off arms get a barrier on every qubit before each mid-circuit measurement group, before each run of conditionals
    that does not follow a read, and after each run of conditionals (psf_os.footprint.fence_waits, conditionals=True:
    the workplace VS's その6) before compiling, so that the compiler cannot merge gates across a
    wait (on the compiled IPE the target otherwise waited in a superposition). The records carry fence=True.
    record_checks (after-2026-11-20, home's three conditions; default "record"): every build also writes OUT/checks.json,
    beside the job and outside it (job/, runs/ and build.json are byte for byte as without it, so every recorded
    build reproduces as before): per item and arm the CZ as written and as compiled (a difference is noted on stderr,
    never refused: strict_cz=True refuses), and the wait footprint as sent (each arm compiled, span) and as written,
    or counted=False with the reason when the circuit is above the footprint's bounds (the build never waits on it).
    record_checks="off" writes nothing.
    ff_only (loop-paired-2026-11-21; default (): every item has its three arms, byte for byte as before): the tags
    named here get their ff arm only (no off, no floor circuit in the job). For the paired loop: the incumbent's ff
    beside the candidate's three arms in one job; the incumbent's off and floor were measured when it was adopted.
    The records of such an item carry ff_only=True.
    park (park-default-2026-11-21; None = PARK_DEFAULT, True unless set by park_default): in the ff and off arms,
    every qubit that waits in |1> in every branch, while a mid-circuit measurement is read or a run of conditionals is
    decided, is put in |0> for the wait (psf_os.footprint.park_waits: X before the wait's fence, X after it; the waits
    fenced as fence_waits(conditionals=True)). Checked at build: the exact noiseless distribution of the classical
    bits unchanged (park_waits), and the compiled CZ count of each arm the same as without parking; otherwise refused.
    An item with nothing to park is built exactly as with park=False. With park True every record carries "park"
    (where and what was parked, excited before and after); with park=False nothing is added, byte for byte as
    before. A recorded build is remade with park_of_record(record) (no key: False).
    declared (release-0.3.0; else the environment variable PSF_OS_DECLARED): the psf-os version a declaration fixed;
    with park=None the build is refused when that version's park default differs from this one's (resolve_park). The
    manifest names the version, the park setting and the declared version (psf_os_build).
    park_likely (likely-park-2026-11-22; release-0.3.1: None = PARK_LIKELY_DEFAULT, True; park_likely=False builds as
    0.3.0, byte for byte; a recorded build without the key is remade with False): True, or the tags of the items to
    treat, parks the likely value after `park` (footprint.park_likely: an X pair around each
    wait on every waiting qubit with P(|1>) > 0.5, kept by a partial fence on the parked qubits and the wait's own
    qubits, at those waits only). Checked as `park` is: the noiseless distribution (park_likely) and the compiled CZ
    of each arm against the unparked arms; the records carry "park_likely", the manifest's psf_os_build the setting.
    park_skip (release-0.3.0; default (): every item as `park` says): with park on, the items named here are built
    unparked (their records say "park": {"park": false, "skipped": "park_skip"}), so that a test can hold the parked
    and the unparked form of the same circuit side by side in one job.
    footprint (changed the same day, after the workplace VS's finding): the records carry the footprint of each arm
    as sent (compiled, span=True: the whole wait) as `sent` and its max / total first, and the item as written beside
    (`written`)."""
    from psf_os import job as J
    from psf_os import user2d as U
    if pack_fixed:
        if chains:
            raise Refused("pack_fixed places the items itself (bundle_place); chains cannot be given with it")
        return build_pack_fixed(dev, items, out, shots=shots, max_seconds=max_seconds, sentinel_spec=sentinel_spec,
                                park=park, declared=declared, park_likely=park_likely,
                                allow_long_delay=allow_long_delay)
    if os.path.exists(out):
        raise Refused(f"{out} exists (nothing is overwritten)")
    tags = [t for t, _, _ in items]
    if not items or len(set(tags)) != len(tags) or not all(U._tag_ok(t) for t in tags):
        raise Refused(f"tags must be distinct and use letters, digits, '_', '.', '-' only: {tags}")
    chains = dict(chains or {})
    ff_only = set(ff_only or ())
    if ff_only - set(tags):
        raise Refused(f"ff_only names unknown items {sorted(ff_only - set(tags))}")
    if set(chains) - set(tags):
        raise Refused(f"chains given for unknown items {sorted(set(chains) - set(tags))}")
    if record_checks not in ("record", "off"):
        raise Refused(f"record_checks must be 'record' or 'off' (got {record_checks!r})")
    park, park_how = resolve_park(park, declared)
    park_likely, likely_how, likely_guard = resolve_park_likely(park_likely, declared)
    if not park and likely_how == "default":
        park_likely = False                    # release-0.3.1: by default, the likely value is parked only with park
    park_skip = set(park_skip or ())
    if park_skip - set(tags):
        raise Refused(f"park_skip names unknown items {sorted(park_skip - set(tags))}")
    comp = Compiler(dev)
    prepared = []
    check_rows = {}
    for tag, qc, spec in items:
        a, qc, off, succ, spec = _prepare(tag, qc, spec)
        if fence:
            from psf_os.footprint import fence_waits
            qc, off = fence_waits(qc, conditionals=True), fence_waits(off, conditionals=True)
        park_rec, unparked = None, None
        if park and tag in park_skip:
            park_rec = {"park": False, "skipped": "park_skip"}
        elif park:
            park_rec, unparked, qc, off = _park_item(tag, qc, off)
        likely_rec = None
        named = park_likely is True or (bool(park_likely) and park_likely is not True and tag in set(park_likely))
        # release-0.3.1: never on a park_skip item (the unparked twin stays unparked); by default only where park is on
        if named and tag not in park_skip and (park or likely_how == "explicit"):
            likely_rec, before_likely, qc, off = _park_likely_item(tag, qc, off)
            if before_likely is not None and likely_guard:
                raise Refused(f"{tag}: the declared psf-os {declared or os.environ.get('PSF_OS_DECLARED')} does not "
                              "park the likely value by default and this psf-os does, and it would change this item: "
                              "pass park_likely=False (the form the declared version built) or park_likely=True")
            if before_likely is not None and unparked is None:
                unparked = before_likely
        try:
            pl = given_chain(dev, chains[tag], a["n"]) if tag in chains else place(dev, a["n"])
        except Refused as e:
            raise Refused(f"{tag}: {e}") from None
        chain = pl["chain"]
        arms = {}
        for arm, logical in (("ff", qc), ("off", off)):
            arms[arm] = comp.run(on_device(logical, chain, dev.num_qubits))
        arms["floor"] = floor_of(arms["ff"])
        if unparked is not None:
            for arm, logical in (("ff", unparked[0]), ("off", unparked[1])):
                if logical is None:
                    continue
                c0 = comp.run(on_device(logical, chain, dev.num_qubits))
                if cz_count(c0) != cz_count(arms[arm]):
                    raise Refused(f"{tag}: parking changed the compiled CZ count of the {arm} arm "
                                  f"({cz_count(c0)} -> {cz_count(arms[arm])}): not built")
            if park_rec is not None and "arms" in park_rec:
                park_rec = _park_physical(park_rec, chain)
        chk = {arm: checks(dev, c, chain) for arm, c in arms.items()}
        if not all(all(v.values()) for v in chk.values()):
            raise Refused(f"{tag}: acceptance failed: {chk}")
        if not any(n == "if_else" for n, _ in walk(arms["ff"])) and a["conditionals"]:
            raise Refused(f"{tag}: the compiler removed the conditionals")
        if record_checks == "record":
            check_rows[tag] = _check_row(tag, qc, off, arms)
            if park_rec is not None:
                check_rows[tag]["park"] = park_rec
            if likely_rec is not None:
                check_rows[tag]["park_likely"] = likely_rec
        if strict_cz:
            written = {"ff": cz_count(qc), "off": cz_count(off), "floor": cz_count(qc)}
            got = {arm: cz_count(c) for arm, c in arms.items()}
            if got != written:
                raise Refused(f"{tag}: compiled CZ {got} differ from the circuit as written {written} (strict_cz: a "
                              "cancelled pair? put a barrier between the gates to keep)")
        fp = None
        if footprint:
            from psf_os.footprint import wait_footprint
            f = wait_footprint(qc)
            fp = dict(max=f["max"], total=f["total"], carriers=[w["carriers"] for w in f["waits"]])
            if True:
                # the footprint as sent (compiled, span: the whole wait), first; as written beside (the workplace
                # VS, 2026-11-20: the compiler can merge gates across a wait, so the written count can be smaller)
                sent = {}
                for arm in ("ff", "off"):
                    g = wait_footprint(arms[arm], span=True, conditionals=True)
                    sent[arm] = dict(max=g["max"], total=g["total"], carriers=[w["carriers"] for w in g["waits"]])
                fp = dict(sent=sent, written=fp, max=sent["ff"]["max"], total=sent["ff"]["total"])
        if likely_rec is not None:
            likely_rec = _park_physical(likely_rec, chain)
        prepared.append((tag, a, succ, spec, pl, arms, chk, written if strict_cz else None, fp, park_rec, likely_rec))
    runs = os.path.join(out, "runs")
    os.makedirs(runs)
    rep_items = {}
    for tag, a, succ, spec, pl, arms, chk, written, fp, park_rec, likely_rec in prepared:
        bound = spec.get("bound")
        chain = pl["chain"]
        labels = list(range(a["n"]))
        mapping = {k: chain[k] for k in labels}
        common = dict(labels=labels, mapping=mapping, shots=shots)
        extra = dict(tag=tag, chain=chain, chain_rule=dict(rule=pl["rule"], start=pl["start"]), success=
                     {str(b): v for b, v in succ.items()}, bound=bound, clbits=a["clbits"], mid=a["mid"],
                     conditionals=a["conditionals"], logical_cz=a["cz"])
        if spec.get("off_circuit") is not None:
            extra["off_explicit"] = True
        if spec.get("kind") == "rus":
            extra["rus"] = {k: spec[k] for k in ("L", "theta", "m", "out_bit", "tries", "exact")}
            extra["logical_cz_in_blocks"] = a["cz_in_blocks"]
        if a.get("register_conditionals"):
            extra.update(register_conditionals=a["register_conditionals"], condition_bits=a["condition_bits"])
        for arm, kind in (("ff", KIND), ("off", KIND + "_off"), ("floor", KIND + "_floor")):
            if tag in ff_only and arm != "ff":
                continue
            c = arms[arm]
            U._record(dev, runs, f"ff_{tag}" + ("" if arm == "ff" else f"_{arm}"), "user" if arm == "ff" else arm,
                      kind, [c], [dict(input="f", setting="z")],
                      extra=dict(extra, ff_arm=arm, cz=cz_count(c), checks=chk[arm],
                                 **({"cz_written": written[arm]} if written else {}),
                                 **({"fence": True} if fence else {}),
                                 **({"ff_only": True} if tag in ff_only else {}),
                                 **({"wait_footprint": fp} if fp is not None else {}),
                                 **({"park": park_rec} if park_rec is not None else {}),
                                 **({"park_likely": likely_rec} if likely_rec is not None else {})), valid=True,
                      **common)
        c = arms["ff"]
        rep_items[tag] = dict(n=a["n"], chain=chain, cz=cz_count(c), off_cz=cz_count(arms["off"]),
                              floor_cz=cz_count(arms["floor"]), depth=c.depth(),
                              mid=a["mid"], conditionals=a["conditionals"], success=succ, bound=bound,
                              **({"rus": extra["rus"], "cz_in_blocks": cz_count(c) - cz_count(arms["off"])}
                                 if "rus" in extra else {}),
                              **({"footprint_max": fp["max"], "footprint_total": fp["total"]} if fp is not None
                                 else {}),
                              **({"park": park_rec} if park_rec is not None else {}))
    cover = _sentinel(dev, runs, shots, sentinel_spec,
                      cover=sorted({q for p in prepared for q in p[4]["chain"]}) if sentinel_over_test else None)
    man = J.assemble(dev, runs, os.path.join(out, "job"), max_seconds=max_seconds, allow_long_delay=allow_long_delay,
                     **({"extra": dict(sentinel_cover=cover)} if cover is not None else {}),
                     build=dict(_build_rec(park, park_how, declared),
                                **({"park_likely": park_likely if park_likely is True else sorted(park_likely)}
                                   if park_likely else {}),
                                **({"park_likely_set": likely_how} if park_likely and likely_how == "default"
                                   else {})))
    if record_checks == "record":
        _write_checks(out, check_rows)
    rep = dict(device=dev.name, kind=KIND, items=rep_items, circuits=len(man["pubs"]),
               qpu_seconds_model=man["qpu_seconds_model"], shots=shots,
               note="the time model does not include mid-circuit measurement and feed-forward latency (microseconds "
                    "per shot), small against its per-shot term")
    if cover is not None:
        rep["sentinel_cover"] = cover
    if park:
        rep["park"] = True
    json.dump(rep, open(os.path.join(out, "build.json"), "w"), indent=1, default=str)
    return rep


def _park_item(tag, qc, off):
    """(record, unparked (qc, off) or None, qc, off) for build(park=True): park_waits on both arms. Nothing to park:
    the arms unchanged and unparked None. Above the footprint's bounds: the arms unchanged, the record says why."""
    from psf_os.footprint import park_waits
    rec, new = {"park": True, "arms": {}}, {}
    for arm, c in (("ff", qc), ("off", off)):
        try:
            c2, info = park_waits(c)
        except ValueError as e:
            if "changed the noiseless distribution" in str(e):
                raise Refused(f"{tag}: {e}") from None
            c2, info = c, dict(parked=[], skipped=str(e))
        new[arm] = c2
        rec["arms"][arm] = info
    if new["ff"] is qc and new["off"] is off:
        return rec, None, qc, off
    return rec, (qc, off), new["ff"], new["off"]


def _build_rec(park, how, declared):
    return dict(park=bool(park), park_set=how,
                **({"declared": declared or os.environ.get("PSF_OS_DECLARED")}
                   if (declared or os.environ.get("PSF_OS_DECLARED")) else {}))


def _park_prep(tags, prep, park, likely=False, likely_guard=False, declared=None):
    """(prep, [record per item or None], unparked prep or None) for the builders of several items (build_bundle,
    build_pack_fixed): each item's ff and off arms parked as build parks a single item (_park_item), then, with
    `likely` (release-0.3.1: True or tags), the likely value parked (_park_likely_item); a likely record rides in the
    item's record under "park_likely"."""
    if not park and not likely:
        return prep, [None] * len(prep), None
    out, recs, changed = [], [], False
    for t, p in zip(tags, prep):
        a, qc, off, succ, spec = p
        rec = None
        if park:
            rec, unparked, qc, off = _park_item(t, qc, off)
            changed |= unparked is not None
        if likely is True or (likely and likely is not True and t in set(likely)):
            lrec, before, qc, off = _park_likely_item(t, qc, off)
            if before is not None and likely_guard:
                raise Refused(f"{t}: the declared psf-os {declared or os.environ.get('PSF_OS_DECLARED')} does not park "
                              "the likely value by default and this psf-os does, and it would change this item: pass "
                              "park_likely=False (the form the declared version built) or park_likely=True")
            changed |= before is not None
            rec = dict(rec or {}, park_likely=lrec)
        recs.append(rec)
        out.append((a, qc, off, succ, spec))
    return out, recs, (prep if changed else None)


def _park_likely_item(tag, qc, off):
    """(record, the arms before or None, qc, off) for build(park_likely=...): footprint.park_likely on both arms."""
    from psf_os.footprint import park_likely
    rec, new = {"park_likely": True, "arms": {}}, {}
    for arm, c in (("ff", qc), ("off", off)):
        try:
            c2, info = park_likely(c)
        except ValueError as e:
            if "changed the noiseless distribution" in str(e):
                raise Refused(f"{tag}: {e}") from None
            c2, info = c, dict(parked=[], skipped=str(e))
        new[arm] = c2
        rec["arms"][arm] = info
    if new["ff"] is qc and new["off"] is off:
        return rec, None, qc, off
    return rec, (qc, off), new["ff"], new["off"]


def _park_physical(rec, chain):
    """The parked qubits also as physical qubits (chain[k] for logical k), in a likely record riding along too."""
    for info in (rec.get("arms") or {}).values():
        for w in info.get("parked", []):
            w["physical"] = [chain[q] for q in w["qubits"]]
    if isinstance(rec.get("park_likely"), dict):
        _park_physical(rec["park_likely"], chain)
    return rec


# ------------------------------------------------------------------------------------------------ bundled items
# 2026-11-17 (the packing test, notebook notes/next_2026-11-17/README.md "Bundled items"): independent items side by
# side in one circuit per arm. Each sub-item is read as a single item is (its own spec, its own off arm); the bundle is
# one ff, one off and one floor circuit plus the job's sentinel, whatever the number of sub-items.

BUNDLE_GAP = 1


def _gaps(gap, n_items):
    """The sites skipped after each item but the last: an int >= 1 for every pair, or a list of n_items - 1 ints >= 1."""
    if isinstance(gap, bool):
        raise Refused(f"gap must be an integer >= 1 or a list of them (got {gap})")
    if isinstance(gap, int):
        if gap < 1:
            raise Refused(f"gap must be an integer >= 1 (got {gap})")
        return [gap] * (n_items - 1)
    g = list(gap)
    if len(g) != n_items - 1 or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in g):
        raise Refused(f"gap as a list needs {n_items - 1} integers >= 1 (got {gap})")
    return g


def bundle_place(dev, sizes, gap=BUNDLE_GAP):
    """One rule-C chain of N sites and a segment per item: items laid in order on consecutive sites; after an item
    `gap` sites are skipped (an int for every pair of neighbours, or a list of len(sizes) - 1 ints, one per pair:
    2026-11-20, a third item further out than the second), and further sites while the next item's first site would be joined by any coupler of the
    device (working or failed) to a qubit of an earlier item. N is the smallest length that holds every item.
    Returns dict(chain, segments (chain positions), rule, start); Refused if no rule-C chain holds them."""
    from psf_os import chain as CH
    adj = {}
    for a, b in dev.all_couplers():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    gaps = _gaps(gap, len(sizes))
    N = sum(sizes) + sum(gaps)
    while N <= dev.num_qubits:
        try:
            pl = CH.place(dev, N)
        except CH.Refused as e:
            raise Refused(f"bundle of sizes {sizes}: no rule-C chain of {N} sites ({e})") from None
        ch = list(pl["chain"])
        segs, used, pos = [], set(), 0
        for k, n in enumerate(sizes):
            while pos + n <= N and any(q in used or adj.get(q, set()) & used for q in ch[pos:pos + n]):
                pos += 1
            if pos + n > N:
                break
            segs.append(list(range(pos, pos + n)))
            used |= set(ch[pos:pos + n])
            pos += n + (gaps[k] if k < len(gaps) else 0)
        if len(segs) == len(sizes):
            return dict(chain=ch, segments=segs, rule=pl["rule"], start=pl["start"])
        N += 1
    raise Refused(f"bundle of sizes {sizes}: does not fit the device")


def walk_bits(qc):
    """Every instruction, top level and inside blocks: (name, qubit indices, clbit indices including the condition's
    bits) of the top circuit. A condition on a register (2026-11-18) lists every bit of the register; before, it gave
    the bit index -1 (nothing produced one: registers were refused at the entrance)."""
    out = []

    def rec(c, qmap, cmap):
        for ins in c.data:
            op = ins.operation
            qs = [qmap[c.find_bit(x).index] for x in ins.qubits]
            cs = [cmap[c.find_bit(x).index] for x in ins.clbits]
            if op.name == "if_else":
                cb = op.condition[0]
                if type(cb).__name__ == "Clbit":
                    cs = cs + [cmap[c.find_bit(cb).index]]
                elif type(cb).__name__ == "ClassicalRegister":
                    cs = cs + [cmap[c.find_bit(x).index] for x in cb]
                else:
                    cs = cs + [-1]
                out.append((op.name, tuple(qs), tuple(cs)))
                for b in op.blocks:
                    if b is not None:
                        rec(b, {j: qs[j] for j in range(len(qs))},
                            {j: cmap[c.find_bit(x).index] for j, x in enumerate(ins.clbits)})
            else:
                out.append((op.name, tuple(qs), tuple(cs)))
    rec(qc, {j: j for j in range(qc.num_qubits)}, {j: j for j in range(qc.num_clbits)})
    return out


def restrict(qc, qubits, clbits):
    """The instructions of qc on the given qubits (in that order) and clbits (in that order), as a circuit of
    len(qubits) qubits and one register c of len(clbits) bits. Refused if an instruction straddles the given set.
    A condition on a register (2026-11-18) is rebuilt on a register of the same bits, in the register's order, of the
    restricted circuit (an aliasing register over c's bits: for local simulation and comparison only, never sent)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import IfElseOp
    qi = {q: k for k, q in enumerate(qubits)}
    ci = {c: k for k, c in enumerate(clbits)}
    out = QuantumCircuit(QuantumRegister(len(qubits), "q"), ClassicalRegister(len(clbits), "c"))
    aliases = {}
    for ins in qc.data:
        qs = [qc.find_bit(x).index for x in ins.qubits]
        cs = [qc.find_bit(x).index for x in ins.clbits]
        op = ins.operation
        cond, rbits = None, None
        if op.name == "if_else":
            cb = op.condition[0]
            if type(cb).__name__ == "ClassicalRegister":
                rbits = [qc.find_bit(x).index for x in cb]
            else:
                cond = qc.find_bit(cb).index
        inq = [q in qi for q in qs]
        inc = [c in ci for c in cs + ([cond] if cond is not None else []) + (rbits or [])]
        if not any(inq) and not any(inc):
            continue
        if not all(inq) or not all(inc):
            raise Refused(f"{op.name} on qubits {qs} / bits {cs} straddles the given set")
        if op.name == "if_else":
            if rbits is None:
                test = out.clbits[ci[cond]]
            else:
                key = tuple(ci[b] for b in rbits)
                if key not in aliases:
                    aliases[key] = ClassicalRegister(bits=[out.clbits[k] for k in key], name=f"r{len(aliases)}")
                    out.add_register(aliases[key])
                test = aliases[key]
            op = IfElseOp((test, op.condition[1]), op.blocks[0].copy(),
                          op.blocks[1].copy() if len(op.blocks) > 1 and op.blocks[1] is not None else None)
        out.append(op, [out.qubits[qi[q]] for q in qs], [out.clbits[ci[c]] for c in cs])
    return out


def wire_signature(qc):
    """Per wire (qubit or clbit; a condition's bit counts as a wire of the conditional), the instructions on it in
    order, with their parameters and the wires they act on (blocks included, recursively). Two circuits with the same
    signature differ at most in the order in which independent instructions are listed."""
    def one(c, ins):
        op = ins.operation
        qs = tuple(c.find_bit(x).index for x in ins.qubits)
        cs = tuple(c.find_bit(x).index for x in ins.clbits)
        if op.name == "if_else":
            cb = op.condition[0]
            if type(cb).__name__ == "ClassicalRegister":
                # 2026-11-18: a register's test, by its bits in the register's order and the value
                cond = ("register", tuple(c.find_bit(x).index for x in cb), int(op.condition[1]))
            else:
                cond = (c.find_bit(cb).index, int(op.condition[1]))
            return ("if_else", qs, cs, cond, tuple(wire_signature(b) if b is not None else None for b in op.blocks))
        return (op.name, qs, cs, tuple(round(float(p), 12) if isinstance(p, (int, float)) else str(p)
                                       for p in op.params))
    wires = {}
    for ins in qc.data:
        s = one(qc, ins)
        if s[0] == "if_else":
            tested = list(s[3][1]) if s[3][0] == "register" else [s[3][0]]
        else:
            tested = []
        ws = [("q", i) for i in s[1]] + [("c", i) for i in s[2]] + [("c", i) for i in tested]
        for w in ws:
            wires.setdefault(w, []).append(s)
    return tuple(sorted((w, tuple(v)) for w, v in wires.items()))


def bundle_acceptance(arms, qubit_sets, bit_sets):
    """Refused if any instruction or barrier of any arm (blocks included) acts on the qubits of two sub-items, reads or
    writes bits of two sub-items, or touches qubits of one sub-item and bits of another. A conditional on a register
    (2026-11-18) counts every bit of the register, so a register holding bits of another sub-item is refused; a test
    whose bits cannot be listed (-1) is refused as before."""
    for arm, c in arms.items():
        for name, qs, cs in walk_bits(c):
            if -1 in cs:
                raise Refused(f"{arm}: a conditional on a whole register")
            owners = {i for i, s in enumerate(qubit_sets) if set(qs) & s} | \
                     {i for i, s in enumerate(bit_sets) if set(cs) & s}
            if len(owners) > 1:
                raise Refused(f"{arm}: {name} on qubits {list(qs)} / bits {list(cs)} spans sub-items {sorted(owners)}")


CONDITIONAL_NEIGHBOURS_WHY = (
    "two or more items with conditional operations in one circuit lose what each keeps alone: 2026-11-17 "
    "(results/packing-night-2026-11-17: ten items packed, the control arm kept 0.20 less, mean d -0.197, 2 SE 0.007) "
    "and 2026-11-18 (results/parallel-night-2026-11-18, P-B: two items, the first lost 0.06 in B2 and B3, a delay did not "
    "remove it; the home VS's independent check confirmed the reading)")


def build_bundle(dev, items, out, tag="pack", shots=4000, max_seconds=60, sentinel_spec=None,
                 allow_conditional_neighbours=False, gap=BUNDLE_GAP, bit_order=None, behind_fixed_first=False,
                 sentinel_over_test=False, record_checks="record", park=None, declared=None, park_likely=None,
                 allow_long_delay=False):
    """items: [(sub_tag, QuantumCircuit, spec)] as for build. One job of one bundle: the sub-items placed by
    bundle_place, every sub-item on its segment and its bits at an offset of one classical register; ff = the
    sub-items' circuits, off = their off arms (explicit or the conditionals removed), floor = floor_of(ff), one ring
    sentinel. Acceptance: checks() of each arm on the bundle's chain, the conditionals kept, bundle_acceptance
    (independence), and per sub-item and arm the same wire_signature as the sub-item compiled alone on the same
    physical qubits. Records: one per arm (tag `tag`), each with the table bundle = {sub_tag: dict(segment, qubits,
    bit_offset, clbits, the sub-item's spec fields)}; cards() reads one card per sub-item from its own bits.
    v0.2 prototype (2026-11-18): refused when two or more sub-items hold conditional operations
    (CONDITIONAL_NEIGHBOURS_WHY), unless allow_conditional_neighbours=True (tests of the loss itself, such as P-B), which
    the manifest records; several conditionals inside one sub-item (e.g. a table-lookup decoder) are not a neighbour and
    are only counted (manifest: conditionals_per_item).
    Interference tests (2026-11-18 evening): gap = the sites skipped between items (bundle_place; default 1, as
    before); bit_order = the sub-item tags in the order their bits take in the register (default: the items' order, as
    before), so that placement (items' order on the chain) and bit order can be varied apart; both recorded in the
    manifest when not at their defaults.
    Receiver prototype (after-2026-11-20; off by default, for the decoy test's row 1 only: "the loss sits on the
    register's first bits; a fixed first item shields"): behind_fixed_first=True allows two or more conditional
    sub-items when the sub-item whose bits come FIRST in the register (bit_order[0], or the first item) holds no
    conditional operation and at least one bit (a fixed item, e.g. an item's off arm); refused otherwise, as before.
    The manifest records conditional_neighbours.allowed_by = "behind_fixed_first" and the shield's tag. Not to be used
    before a declared night has read row 1.
    sentinel_over_test (after-2026-11-20; default False): as build's, over the bundle's sub-items' qubits.
    park, declared (release-0.3.0): as build's. Each sub-item's ff and off arms are parked as build parks a single
    item (at its own waits; a wait of another sub-item is not looked at) before the sub-items are laid side by side;
    the compiled CZ counts of the bundle's arms must equal the unparked bundle's (else refused); the bundle table
    carries each sub-item's "park", the manifest the setting (psf_os_build). park=False: as before.
    record_checks (after-2026-11-20; default "record"): as build's, OUT/checks.json for the bundle's three arms."""
    if record_checks not in ("record", "off"):
        raise Refused(f"record_checks must be 'record' or 'off' (got {record_checks!r})")
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from psf_os import job as J
    from psf_os import user2d as U
    if os.path.exists(out):
        raise Refused(f"{out} exists (nothing is overwritten)")
    tags = [t for t, _, _ in items]
    if len(items) < 2 or len(set(tags)) != len(tags) or not all(U._tag_ok(t) for t in tags + [tag]):
        raise Refused(f"a bundle holds two or more distinct sub-items with plain tags: {tags}")
    if bit_order is not None and sorted(bit_order) != sorted(tags):
        raise Refused(f"{tag}: bit_order must name every sub-item once: {bit_order} against {tags}")
    try:
        _gaps(gap, len(items))
    except Refused as e:
        raise Refused(f"{tag}: {e}") from None
    prep = [_prepare(t, qc, spec) for t, qc, spec in items]
    park, park_how = resolve_park(park, declared)
    park_likely, likely_how, likely_guard = resolve_park_likely(park_likely, declared)
    if not park and likely_how == "default":
        park_likely = False                    # release-0.3.1: by default, the likely value is parked only with park
    prep, park_recs, unparked_prep = _park_prep(tags, prep, park, park_likely, likely_guard, declared)
    with_cond = [t for t, p in zip(tags, prep) if p[0]["conditionals"]]
    shield = None
    if len(with_cond) >= 2 and not allow_conditional_neighbours and behind_fixed_first:
        first = (list(bit_order) if bit_order is not None else tags)[0]
        fp = prep[tags.index(first)][0]
        if first in with_cond or fp["clbits"] < 1:
            raise Refused(f"{tag}: behind_fixed_first needs the sub-item whose bits come first ({first}) to hold no "
                          f"conditional operation and at least one bit (conditionals {fp['conditionals']}, bits "
                          f"{fp['clbits']})")
        shield = first
    if len(with_cond) >= 2 and not allow_conditional_neighbours and shield is None:
        raise Refused(f"{tag}: sub-items {with_cond} hold conditional operations; {CONDITIONAL_NEIGHBOURS_WHY}. Build them "
                      "as single items (build), or pass allow_conditional_neighbours=True for a test of the loss itself")
    man_extra = dict(conditionals_per_item={t: p[0]["conditionals"] for t, p in zip(tags, prep)},
                     conditional_neighbours=dict(items=with_cond, allowed=bool(allow_conditional_neighbours),
                                                 used=len(with_cond) >= 2))
    if shield is not None:
        man_extra["conditional_neighbours"].update(allowed_by="behind_fixed_first", shield=shield)
    if gap != BUNDLE_GAP:
        man_extra["bundle_gap"] = gap
    if bit_order is not None and list(bit_order) != tags:
        man_extra["bit_order"] = list(bit_order)
    sizes = [p[0]["n"] for p in prep]
    pl = bundle_place(dev, sizes, gap=gap)
    chain, segs = pl["chain"], pl["segments"]
    N = len(chain)
    order = list(bit_order) if bit_order is not None else tags
    start, o = {}, 0
    for t in order:
        start[t] = o
        o += prep[tags.index(t)][0]["clbits"]
    lo = [start[t] for t in tags]                                  # bit offset of item i (items' order)
    hi = [lo[i] + prep[i][0]["clbits"] for i in range(len(prep))]
    nbits = o
    logical = {}
    multi = any(len(p[1].cregs) > 1 for p in prep)
    if multi and order != tags:
        raise Refused(f"{tag}: bit_order cannot be changed for sub-items with several registers")
    for which, k in (("ff", 1), ("off", 2)):
        if multi:
            # 2026-11-18: a sub-item with several registers (a register condition) keeps them, renamed per sub-item and
            # laid one after another in bit order, so compose maps every register condition onto a register of the
            # bundle (one register "c" would make compose add an aliasing register: bits reported twice)
            regs = [ClassicalRegister(len(r), f"s{i}_{r.name}") for i, p in enumerate(prep) for r in p[1].cregs]
            c1 = QuantumCircuit(QuantumRegister(N, "q"), *regs)
        else:
            c1 = QuantumCircuit(QuantumRegister(N, "q"), ClassicalRegister(nbits, "c"))
        for i, p in enumerate(prep):
            c1.compose(p[k], qubits=segs[i], clbits=list(range(lo[i], hi[i])), inplace=True)
        logical[which] = c1
    comp = Compiler(dev)
    arms = {w: comp.run(on_device(logical[w], chain, dev.num_qubits)) for w in ("ff", "off")}
    arms["floor"] = floor_of(arms["ff"])
    if unparked_prep is not None:
        for w, k in (("ff", 1), ("off", 2)):
            c0 = comp.run(on_device(_bundle_logical(unparked_prep, k, segs, lo, hi, N, nbits, multi), chain,
                                    dev.num_qubits))
            if cz_count(c0) != cz_count(arms[w]):
                raise Refused(f"{tag}: parking changed the compiled CZ count of the {w} arm "
                              f"({cz_count(c0)} -> {cz_count(arms[w])}): not built")
    chk = {arm: checks(dev, c, chain) for arm, c in arms.items()}
    if not all(all(v.values()) for v in chk.values()):
        raise Refused(f"{tag}: acceptance failed: {chk}")
    if any(p[0]["conditionals"] for p in prep) and not any(n == "if_else" for n, _ in walk(arms["ff"])):
        raise Refused(f"{tag}: the compiler removed the conditionals")
    qsets = [set(chain[k] for k in s) for s in segs]
    bsets = [set(range(lo[i], hi[i])) for i in range(len(prep))]
    bundle_acceptance(arms, qsets, bsets)
    table = {}
    for i, (t, p) in enumerate(zip(tags, prep)):
        a, qc, off, succ, spec = p
        phys = [chain[k] for k in segs[i]]
        bits = list(range(lo[i], hi[i]))
        alone = {w: comp.run(on_device(lg, phys, dev.num_qubits)) for w, lg in (("ff", qc), ("off", off))}
        alone["floor"] = floor_of(alone["ff"])
        for w in ("ff", "off", "floor"):
            mine = restrict(arms[w], phys, bits)
            ref = restrict(alone[w], phys, list(range(a["clbits"])))
            if wire_signature(mine) != wire_signature(ref):
                raise Refused(f"{tag}: sub-item {t} ({w}) differs from the sub-item compiled alone on the same qubits")
        table[t] = dict(segment=segs[i], qubits=phys, bit_offset=lo[i], tag=t, chain=phys,
                        **_item_extra(t, a, succ, spec), cz=cz_count(restrict(arms["ff"], phys, bits)),
                        **({"park": _park_physical(park_recs[i], phys)} if park_recs[i] is not None else {}))
    runs = os.path.join(out, "runs")
    os.makedirs(runs)
    labels = list(range(N))
    common = dict(labels=labels, mapping={k: chain[k] for k in labels}, shots=shots)
    extra = dict(tag=tag, chain=chain, chain_rule=dict(rule=pl["rule"], start=pl["start"], bundle_gap=gap),
                 clbits=nbits, bundle=table)
    for arm, kind in (("ff", KIND), ("off", KIND + "_off"), ("floor", KIND + "_floor")):
        c = arms[arm]
        U._record(dev, runs, f"ff_{tag}" + ("" if arm == "ff" else f"_{arm}"), "user" if arm == "ff" else arm,
                  kind, [c], [dict(input="f", setting="z")],
                  extra=dict(extra, ff_arm=arm, cz=cz_count(c), checks=chk[arm]), valid=True, **common)
    cover = _sentinel(dev, runs, shots, sentinel_spec,
                      cover=sorted(q for t in tags for q in table[t]["qubits"]) if sentinel_over_test else None)
    if cover is not None:
        man_extra["sentinel_cover"] = cover
    man = J.assemble(dev, runs, os.path.join(out, "job"), max_seconds=max_seconds, extra=man_extra,
                     allow_long_delay=allow_long_delay,
                     build=dict(_build_rec(park, park_how, declared),
                                **({"park_likely": park_likely if park_likely is True else sorted(park_likely),
                                    "park_likely_set": likely_how} if park_likely else {})))
    if record_checks == "record":
        row = _check_row(tag, logical["ff"], logical["off"], arms)
        if park:
            row["park"] = {t: table[t].get("park") for t in tags}
        _write_checks(out, {tag: row})
    rep = dict(device=dev.name, kind=KIND, bundle=tag, chain=chain, segments=segs,
               items={t: dict(n=sizes[i], chain=table[t]["qubits"], bit_offset=lo[i], clbits=prep[i][0]["clbits"],
                              cz=table[t]["cz"]) for i, t in enumerate(tags)},
               circuits=len(man["pubs"]), qpu_seconds_model=man["qpu_seconds_model"], shots=shots,
               cz=dict(ff=cz_count(arms["ff"]), off=cz_count(arms["off"]), floor=cz_count(arms["floor"])))
    if cover is not None:
        rep["sentinel_cover"] = cover
    json.dump(rep, open(os.path.join(out, "build.json"), "w"), indent=1, default=str)
    return rep


def _bundle_logical(prep, k, segs, lo, hi, N, nbits, multi):
    """build_bundle's composition of the sub-items' arms (k = 1 ff, k = 2 off): several registers kept and renamed,
    or one register with each sub-item at its bit offset."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    if multi:
        regs = [ClassicalRegister(len(r), f"s{i}_{r.name}") for i, p in enumerate(prep) for r in p[1].cregs]
        c1 = QuantumCircuit(QuantumRegister(N, "q"), *regs)
    else:
        c1 = QuantumCircuit(QuantumRegister(N, "q"), ClassicalRegister(nbits, "c"))
    for i, p in enumerate(prep):
        c1.compose(p[k], qubits=segs[i], clbits=list(range(lo[i], hi[i])), inplace=True)
    return c1


def _packed_logical(prep, k, segs, offs, N):
    """The items' circuits (k = 1 ff, k = 2 off) side by side on an N-site chain, one register per build_bundle's
    rule (several registers kept and renamed when any item has more than one)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    if any(len(p[1].cregs) > 1 for p in prep):
        regs = [ClassicalRegister(len(r), f"s{i}_{r.name}") for i, p in enumerate(prep) for r in p[1].cregs]
        c1 = QuantumCircuit(QuantumRegister(N, "q"), *regs)
    else:
        c1 = QuantumCircuit(QuantumRegister(N, "q"), ClassicalRegister(offs[-1], "c"))
    for i, p in enumerate(prep):
        c1.compose(p[k], qubits=segs[i], clbits=list(range(offs[i], offs[i + 1])), inplace=True)
    return c1


def exact_distribution(qc, tol=1e-15):
    """Exact distribution of the classical register of a dynamic circuit on its touched qubits (measurements anywhere,
    if_else on one bit or a register, with or without else), every measurement branch followed with a state vector:
    {bitstring c[n-1]..c[0]: probability}. For acceptance checks of small items only."""
    import numpy as np
    from qiskit.quantum_info import Operator
    qc = compact(qc)
    n, nc = qc.num_qubits, qc.num_clbits
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    branches = [(1.0, psi, [0] * nc)]

    def apply(psi, qs, U):
        t = psi.reshape([2] * n)
        axes = [n - 1 - q for q in qs]
        k = len(qs)
        Ut = U.reshape([2] * (2 * k))
        t = np.tensordot(Ut, t, axes=(list(range(2 * k - 1, k - 1, -1)), axes))
        t = np.moveaxis(t, list(range(k)), axes[::-1])
        return t.reshape(-1)

    def holds(bits, op, circ, cmap):
        cond = op.condition
        if type(cond[0]).__name__ == "Clbit":
            return bits[cmap[circ.find_bit(cond[0]).index]] == int(cond[1])
        reg = cond[0]
        val = sum(bits[cmap[circ.find_bit(b).index]] << j for j, b in enumerate(reg))
        return val == int(cond[1])

    def run(branches, circ, qmap, cmap):
        for ins in circ.data:
            op = ins.operation
            qs = [qmap[circ.find_bit(x).index] for x in ins.qubits]
            if op.name in ("barrier", "delay", "id"):
                continue
            if op.name == "measure":
                cb, q = cmap[circ.find_bit(ins.clbits[0]).index], qs[0]
                nb = []
                for w, ps, bits in branches:
                    t = ps.reshape([2] * n)
                    for v in (0, 1):
                        sl = [slice(None)] * n
                        sl[n - 1 - q] = 1 - v
                        ph = t.copy()
                        ph[tuple(sl)] = 0
                        pr = float(np.vdot(ph, ph).real)
                        if pr * w > tol:
                            b2 = list(bits)
                            b2[cb] = v
                            nb.append((w * pr, (ph / pr ** 0.5).reshape(-1), b2))
                branches = nb
                continue
            if op.name == "if_else":
                tb = op.blocks[0]
                fb = op.blocks[1] if len(op.blocks) > 1 else None
                yes = [b for b in branches if holds(b[2], op, circ, cmap)]
                no = [b for b in branches if not holds(b[2], op, circ, cmap)]
                sub = lambda blk: ({blk.find_bit(x).index: qs[j] for j, x in enumerate(blk.qubits)},
                                   {blk.find_bit(x).index: cmap[circ.find_bit(ins.clbits[j]).index]
                                    for j, x in enumerate(blk.clbits)})
                if yes:
                    yes = run(yes, tb, *sub(tb))
                if no and fb is not None:
                    no = run(no, fb, *sub(fb))
                branches = yes + no
                continue
            U = Operator(op).data
            branches = [(w, apply(ps, qs, U), b) for w, ps, b in branches]
        return branches

    out = {}
    for w, _, bits in run(branches, qc, {j: j for j in range(n)}, {j: j for j in range(nc)}):
        k = "".join(str(b) for b in bits[::-1])
        out[k] = out.get(k, 0.0) + w
    return out


FIXED_PACK_TAG = "fixedpack"


def build_pack_fixed(dev, items, out, shots=4000, max_seconds=60, sentinel_spec=None, park=None, declared=None,
                     park_likely=None, allow_long_delay=False):
    """v0.2 prototype (2026-11-18): the production form of 2026-11-18's P-A. Every item's ff arm is its own circuit,
    placed on the segment bundle_place gives it; the fixed arms (no conditional) of all items are packed: one off
    circuit (the items' off arms side by side, one register with a bit offset per item) and one floor circuit
    (floor_of the items' ff circuits side by side). Acceptance: checks() of every circuit, the conditionals kept in each
    ff arm, bundle_acceptance on the packed off and floor, and per item the packed off and floor restricted to the item
    equal (wire_signature) the item's off and floor compiled alone on the same qubits, with the same exact distribution
    of the off arm (exact_distribution). Records: per item its ff record as build writes it (with fixed_pack naming the
    packed records), and the packed off and floor records with the table fixed_pack_table = {tag: dict(segment, qubits,
    bit_offset, clbits)}. cards() gives every item the card build would give it, its off and floor read from its own
    bits of the packed circuits. Circuits: one per item + 2 + the sentinel's.
    park, declared (release-0.3.0): as build's: every item's ff and off arms parked as build parks them; the compiled
    CZ of each ff arm and of the packed off arm equal to the unparked ones (else refused); each ff record carries
    "park", the manifest the setting (psf_os_build). park=False: as before."""
    from psf_os import job as J
    from psf_os import user2d as U
    if os.path.exists(out):
        raise Refused(f"{out} exists (nothing is overwritten)")
    tags = [t for t, _, _ in items]
    if len(items) < 2 or len(set(tags)) != len(tags) or not all(U._tag_ok(t) for t in tags) or FIXED_PACK_TAG in tags:
        raise Refused(f"pack_fixed needs two or more distinct items with plain tags (not '{FIXED_PACK_TAG}'): {tags}")
    prep = [_prepare(t, qc, spec) for t, qc, spec in items]
    park, park_how = resolve_park(park, declared)
    park_likely, likely_how, likely_guard = resolve_park_likely(park_likely, declared)
    if not park and likely_how == "default":
        park_likely = False                    # release-0.3.1: by default, the likely value is parked only with park
    prep, park_recs, unparked_prep = _park_prep(tags, prep, park, park_likely, likely_guard, declared)
    sizes = [p[0]["n"] for p in prep]
    pl = bundle_place(dev, sizes)
    chain, segs = pl["chain"], pl["segments"]
    N = len(chain)
    offs = [0]
    for p in prep:
        offs.append(offs[-1] + p[0]["clbits"])
    comp = Compiler(dev)
    phys = [[chain[k] for k in s] for s in segs]
    # the ff arms, single, each on its segment (as build with chains = the segments)
    ff, chk_ff = {}, {}
    for i, (t, p) in enumerate(zip(tags, prep)):
        c = comp.run(on_device(p[1], phys[i], dev.num_qubits))
        chk = checks(dev, c, phys[i])
        if not all(chk.values()):
            raise Refused(f"{t}: acceptance failed: {chk}")
        if p[0]["conditionals"] and not any(n == "if_else" for n, _ in walk(c)):
            raise Refused(f"{t}: the compiler removed the conditionals")
        ff[t], chk_ff[t] = c, chk
    # the packed fixed arms
    packed = dict(off=comp.run(on_device(_packed_logical(prep, 2, segs, offs, N), chain, dev.num_qubits)),
                  floor=floor_of(comp.run(on_device(_packed_logical(prep, 1, segs, offs, N), chain, dev.num_qubits))))
    if unparked_prep is not None:
        for i, (t, p) in enumerate(zip(tags, unparked_prep)):
            c0 = comp.run(on_device(p[1], phys[i], dev.num_qubits))
            if cz_count(c0) != cz_count(ff[t]):
                raise Refused(f"{t}: parking changed the compiled CZ count of the ff arm: not built")
        c0 = comp.run(on_device(_packed_logical(unparked_prep, 2, segs, offs, N), chain, dev.num_qubits))
        if cz_count(c0) != cz_count(packed["off"]):
            raise Refused("pack_fixed: parking changed the compiled CZ count of the packed off arm: not built")
    chk_p = {arm: checks(dev, c, chain) for arm, c in packed.items()}
    if not all(all(v.values()) for v in chk_p.values()):
        raise Refused(f"pack_fixed: acceptance of the packed arms failed: {chk_p}")
    if any(n == "if_else" for n, _ in walk(packed["off"])):
        raise Refused("pack_fixed: the packed off arm holds a conditional")
    qsets = [set(q) for q in phys]
    bsets = [set(range(offs[i], offs[i + 1])) for i in range(len(prep))]
    bundle_acceptance(packed, qsets, bsets)
    table = {}
    for i, (t, p) in enumerate(zip(tags, prep)):
        a = p[0]
        bits = list(range(offs[i], offs[i + 1]))
        alone_off = comp.run(on_device(p[2], phys[i], dev.num_qubits))
        alone = dict(off=alone_off, floor=floor_of(ff[t]))
        for w in ("off", "floor"):
            mine = restrict(packed[w], phys[i], bits)
            ref = restrict(alone[w], phys[i], list(range(a["clbits"])))
            if wire_signature(mine) != wire_signature(ref):
                raise Refused(f"pack_fixed: {t} ({w}) differs from its {w} arm compiled alone on the same qubits")
            if w == "off":
                dm, dr = exact_distribution(mine), exact_distribution(ref)
                if max(abs(dm.get(k, 0.0) - dr.get(k, 0.0)) for k in set(dm) | set(dr)) > 1e-12:
                    raise Refused(f"pack_fixed: {t} (off) has another exact distribution packed than alone")
        table[t] = dict(segment=segs[i], qubits=phys[i], bit_offset=offs[i], clbits=a["clbits"])
    runs = os.path.join(out, "runs")
    os.makedirs(runs)
    rep_items = {}
    for i, (t, p) in enumerate(zip(tags, prep)):
        a, qc, off, succ, spec = p
        labels = list(range(a["n"]))
        extra = dict(tag=t, chain=phys[i], chain_rule=dict(rule="fixed_pack", start=pl["start"],
                                                           bundle_gap=BUNDLE_GAP),
                     **_item_extra(t, a, succ, spec), fixed_pack=FIXED_PACK_TAG)
        U._record(dev, runs, f"ff_{t}", "user", KIND, [ff[t]], [dict(input="f", setting="z")],
                  extra=dict(extra, ff_arm="ff", cz=cz_count(ff[t]), checks=chk_ff[t],
                             **({"park": _park_physical(park_recs[i], phys[i])} if park_recs[i] is not None
                                else {})), valid=True, labels=labels,
                  mapping={k: phys[i][k] for k in labels}, shots=shots)
        rep_items[t] = dict(n=a["n"], chain=phys[i], cz=cz_count(ff[t]), mid=a["mid"], conditionals=a["conditionals"],
                            bit_offset=offs[i], **({"rus": extra["rus"]} if "rus" in extra else {}))
    labels = list(range(N))
    pextra = dict(tag=FIXED_PACK_TAG, chain=chain, chain_rule=dict(rule=pl["rule"], start=pl["start"],
                                                                   bundle_gap=BUNDLE_GAP),
                  clbits=offs[-1], fixed_pack_table=table)
    for arm, kind in (("off", KIND + "_off"), ("floor", KIND + "_floor")):
        c = packed[arm]
        U._record(dev, runs, f"ff_{FIXED_PACK_TAG}_{arm}", arm, kind, [c], [dict(input="f", setting="z")],
                  extra=dict(pextra, ff_arm=arm, cz=cz_count(c), checks=chk_p[arm]), valid=True, labels=labels,
                  mapping={k: chain[k] for k in labels}, shots=shots)
    _sentinel(dev, runs, shots, sentinel_spec)
    man = J.assemble(dev, runs, os.path.join(out, "job"), max_seconds=max_seconds, allow_long_delay=allow_long_delay,
                     extra=dict(pack_fixed=dict(items=tags, packed=["off", "floor"]),
                                conditionals_per_item={t: p[0]["conditionals"] for t, p in zip(tags, prep)}),
                     build=dict(_build_rec(park, park_how, declared),
                                **({"park_likely": park_likely if park_likely is True else sorted(park_likely),
                                    "park_likely_set": likely_how} if park_likely else {})))
    rep = dict(device=dev.name, kind=KIND, pack_fixed=True, chain=chain, segments=segs, items=rep_items,
               circuits=len(man["pubs"]), qpu_seconds_model=man["qpu_seconds_model"], shots=shots,
               cz=dict(off=cz_count(packed["off"]), floor=cz_count(packed["floor"])))
    json.dump(rep, open(os.path.join(out, "build.json"), "w"), indent=1, default=str)
    return rep


def marginal_counts(counts, offset, nbits):
    """Counts of bits offset .. offset + nbits - 1 only (Qiskit order: bit 0 rightmost)."""
    out = {}
    for key, v in counts.items():
        k = key.replace(" ", "")
        sub = k[len(k) - offset - nbits:len(k) - offset]
        out[sub] = out.get(sub, 0) + v
    return out


# ------------------------------------------------------------------------------------------------ local run, card

LOCAL_STRIDE_MIN = 10_000     # run_local's seed slots (after-2026-11-20): at least this far apart, and more than the shots
LOCAL_PARTS_MAX = 64          # slots per circuit for the parts of a bundle


def local_seeds(seed, shots, n_circuits, stride=None):
    """The seed of circuit k (part j) of a local run: slot k * LOCAL_PARTS_MAX + j at seed + w * slot with
    w = max(LOCAL_STRIDE_MIN, shots + 1), so that no two runs share a per-shot seed (Aer seeds shot i with seed + i:
    seeds closer than the shots give the same shots shifted, 2026-11-18 / 2026-11-20). stride=1 is the formula before
    after-2026-11-20 (circuit k at seed + k, part j at seed + 1000 k + j), kept to reproduce old dry runs byte for byte.
    Returns a function (k, j=None) -> seed."""
    if stride == 1:
        return lambda k, j=None: seed + k if j is None else seed + 1000 * k + j
    w = max(LOCAL_STRIDE_MIN, shots + 1) if stride is None else int(stride)
    if w <= shots:
        raise Refused(f"seed stride {w} is not above the shots ({shots})")
    return lambda k, j=None: seed + w * (k * LOCAL_PARTS_MAX + (j or 0))


def run_local(out, seed=20261116, stride=None):
    """Noiseless Aer on the job's circuits (compacted to their touched qubits; Clifford circuits by the stabilizer
    method, others by statevector up to LOCAL_MAX_QUBITS touched qubits, else refused at once; control flow
    supported); raw_counts.json and
    submission.json as a device run writes them (local=True). For dry runs only: refused if a submission exists.
    Seeds (after-2026-11-20): local_seeds(seed, shots, circuits, stride); the default keeps every circuit's and every
    part's shot seeds apart; stride=1 gives the seeds of v0.2.0 (recorded runs reproduce with it)."""
    from qiskit import qpy
    from qiskit_aer import AerSimulator
    from psf_os.sim import to_clifford_gates
    job = os.path.join(out, "job")
    sub = os.path.join(job, "submission.json")
    if os.path.exists(sub):
        raise Refused(f"{sub} exists: this job was already run")
    man = json.load(open(os.path.join(job, "manifest.json")))
    circs = qpy.load(open(os.path.join(job, "circuits.qpy"), "rb"))
    rows = []
    sd = local_seeds(seed, man["shots"], len(circs), stride)
    if stride != 1 and len(circs) * LOCAL_PARTS_MAX * max(LOCAL_STRIDE_MIN, man["shots"] + 1) + seed >= 2 ** 62:
        raise Refused("too many circuits for the seed slots")
    for k, c in enumerate(circs):
        cc = compact(c)
        cl = None if cc.has_control_flow_op() else to_clifford_gates(cc)
        if cl is not None:
            cc, meth = cl, "stabilizer"
        else:
            parts = components(c) if cc.num_qubits > LOCAL_MAX_QUBITS else None
            if parts and len(parts) > 1 and all(len(q) <= LOCAL_MAX_QUBITS for q, _ in parts):
                # a bundle (2026-11-17): independent parts, each simulated alone, joined shot by shot (exact: the
                # parts share no qubit and no bit, so the joint distribution is the product)
                if len(parts) > LOCAL_PARTS_MAX and stride != 1:
                    raise Refused(f"circuit {k}: {len(parts)} parts, above {LOCAL_PARTS_MAX} seed slots")
                rows.append(dict(pub=k, counts=_run_parts(c, parts, man["shots"], seed,
                                                          seed_of=lambda j, k=k: sd(k, j))))
                continue
            if cc.num_qubits > LOCAL_MAX_QUBITS:
                raise Refused(f"circuit {k} ({c.name}) is not Clifford and touches {cc.num_qubits} qubits: too slow to "
                              "simulate here (the ring sentinel at h = 0.6 on 40 qubits did not finish in 18 min). "
                              "For a dry run build the job with sentinel_spec=dict(kind='echo_clifford', D=24)")
            meth = "statevector"
        res = AerSimulator(method=meth).run(cc, shots=man["shots"], seed_simulator=sd(k)).result()
        rows.append(dict(pub=k, counts={key.replace(" ", ""): v for key, v in res.get_counts(0).items()}))
    from psf_os.ibm import _stamp
    t = _stamp()
    info = dict(job_id=f"local-noiseless-{seed}", submitted=t, local=True, backend=man["fetch"]["backend"],
                simulator="aer noiseless (statevector / mps, compacted)")
    if stride != 1:
        info["seed_stride"] = "default" if stride is None else int(stride)
    json.dump(info, open(sub, "w"), indent=1)
    json.dump(dict(fetched=t, job_id=f"local-noiseless-{seed}", status="DONE (local noiseless)", rows=rows),
              open(os.path.join(job, "raw_counts.json"), "w"))
    return f"local-noiseless-{seed}"


def components(qc):
    """The independent parts of a circuit: [(sorted qubits, sorted clbits)] joined by any instruction (blocks and
    condition bits included); barriers join nothing. Qubits and bits no instruction touches are left out."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for name, qs, cs in walk_bits(qc):
        if name == "barrier":
            continue
        ws = [("q", q) for q in qs] + [("c", b) for b in cs if b >= 0]
        for w in ws:
            find(w)
        for w in ws[1:]:
            ra, rb = find(ws[0]), find(w)
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for w in list(parent):
        groups.setdefault(find(w), []).append(w)
    out = [(sorted(i for t, i in g if t == "q"), sorted(i for t, i in g if t == "c")) for g in groups.values()]
    return sorted(out)


def _run_parts(qc, parts, shots, seed, seed_of=None):
    """Noiseless shots of qc as the product of its independent parts (each by statevector, memory per shot); part j
    with seed_of(j), or seed + j (v0.2.0) when seed_of is None."""
    from qiskit_aer import AerSimulator
    nb = qc.num_clbits
    shots_bits = [["0"] * nb for _ in range(shots)]
    for j, (qs, cs) in enumerate(parts):
        sub = restrict(qc, qs, cs)
        if not cs:
            continue
        res = AerSimulator(method="statevector").run(sub, shots=shots, memory=True,
                                                     seed_simulator=seed + j if seed_of is None else seed_of(j)).result()
        mem = res.get_memory(0)
        for s, key in enumerate(mem):
            key = key.replace(" ", "")
            for k, b in enumerate(cs):
                shots_bits[s][nb - 1 - b] = key[len(key) - 1 - k]
    out = {}
    for row in shots_bits:
        key = "".join(row)
        out[key] = out.get(key, 0) + 1
    return out


def compact(qc):
    """The circuit on its touched qubits only (classical register objects kept, so conditions stay valid)."""
    from qiskit import QuantumCircuit, QuantumRegister
    act = sorted({q for _, qs in walk(qc) for q in qs})
    idx = {q: i for i, q in enumerate(act)}
    out = QuantumCircuit(QuantumRegister(len(act), "q"), *qc.cregs)
    for ins in qc.data:
        out.append(ins.operation, [out.qubits[idx[qc.find_bit(x).index]] for x in ins.qubits], ins.clbits)
    return out


def _bits(key, n):
    """Classical bits of a counts key (Qiskit order: clbit 0 rightmost)."""
    key = key.replace(" ", "")
    return [int(key[-1 - i]) for i in range(n)]


def success_share(counts, n, success):
    tot = sum(counts.values())
    ok = sum(c for key, c in counts.items() if all(_bits(key, n)[b] == v for b, v in success.items()))
    p = ok / tot
    return dict(shots=tot, counted=tot, discarded=0, success=ok, share=p,
                two_se=2 * math.sqrt(p * (1 - p) / tot))     # 0 when every shot (or none) succeeded


def best_fixed_rule(counts, n, final_bit, mid_bits):
    """Beside, after the fact: the best success any fixed rule f(mid bits) -> flip of the final bit reaches on these
    counts (chosen after seeing them: an optimistic estimate of what classical processing can do)."""
    by = {}
    for key, c in counts.items():
        b = _bits(key, n)
        by.setdefault(tuple(b[i] for i in mid_bits), [0, 0])[b[final_bit]] += c
    return sum(max(v) for v in by.values()) / sum(counts.values())


def _p1(counts, n, bit, keep=None):
    tot = sum(counts.values())
    kept = {k: v for k, v in counts.items() if keep is None or keep(_bits(k, n))}
    nk = sum(kept.values())
    ones = sum(v for k, v in kept.items() if _bits(k, n)[bit] == 1)
    p = ones / nk if nk else float("nan")
    return dict(shots=tot, counted=nk, discarded=tot - nk, kept_share=nk / tot, p1=p,
                two_se=2 * math.sqrt(p * (1 - p) / nk) if nk else float("nan"))


def rus_card(tag, rec, ff_counts, off_counts, floor_counts, sentinel_status):
    """The repeat-until-success card: P(last output 1) against the exact value for ff (every shot counted), off
    (no control, every shot counted) and post (the off arm's shots whose first try succeeded for every neuron:
    post-selected, shots discarded, never a verdict); ff's share still failing after m tries; the floor."""
    rus, n = rec["rus"], rec["clbits"]
    out, tries, ex = rus["out_bit"], rus["tries"], rus["exact"]
    ff = _p1(ff_counts, n, out)
    still = sum(v for k, v in ff_counts.items() if any(all(_bits(k, n)[b] == 1 for b in t) for t in tries))
    ff["still_failed_after_m"] = still / sum(ff_counts.values())
    off = _p1(off_counts, n, out)
    post = _p1(off_counts, n, out, keep=lambda b: all(b[t[0]] == 0 for t in tries))
    for d in (ff, off, post):
        d["minus_exact"] = d["p1"] - ex
        d["in_se"] = d["minus_exact"] / (d["two_se"] / 2) if d["two_se"] and d["two_se"] > 0 else None
    tot = sum(floor_counts.values())
    zero = sum(v for k, v in floor_counts.items() if set(_bits(k, n)) == {0}) / tot
    ones = [sum(v for k, v in floor_counts.items() if _bits(k, n)[i] == 1) / tot for i in range(n)]
    return dict(tag=tag, chain=rec["chain"], L=rus["L"], theta=rus["theta"], m=rus["m"], exact=ex, ff=ff, off=off,
                post=post, floor=dict(all_zero=zero, ones_per_bit=ones), sentinel=sentinel_status)


def cards(out):
    """{tag: card} of a finished feed-forward job (module docstring); the sentinel's status beside."""
    from psf_os import job as J
    from psf_os import sentinel
    job, runs = os.path.join(out, "job"), os.path.join(out, "runs")
    read = J.read(job, runs)
    got = {}
    for name, (rec, cc) in read.items():
        kind = rec.get("psfos_job", {}).get("kind", "")
        if kind.startswith(KIND):
            got.setdefault(rec["tag"], {})[rec["ff_arm"]] = (rec, cc[("f", "z")])
    # v0.2 prototype (2026-11-18): the packed fixed arms of a pack_fixed build give each item its off and floor counts
    # from its own bits
    for ptag in [t for t, arms in got.items() if any("fixed_pack_table" in r for r, _ in arms.values())]:
        parms = got.pop(ptag)
        for arm, (prec, pcnt) in parms.items():
            for t, info in prec["fixed_pack_table"].items():
                got.setdefault(t, {})[arm] = (prec, marginal_counts(pcnt, info["bit_offset"], info["clbits"]))
    try:
        sent = sentinel.verdict(job, runs)["status"]
    except Exception as e:  # noqa: BLE001
        sent = f"not scored ({type(e).__name__})"
    res = {}
    for tag, arms in got.items():
        rec, cnt = arms["ff"]
        if "bundle" in rec:
            # a bundle (2026-11-17): one card per sub-item, from the marginal counts of its own bits, computed exactly
            # as for a single item; the floor's all-zero share is over the sub-item's own bits
            for sub, info in rec["bundle"].items():
                o, nb = info["bit_offset"], info["clbits"]
                part = {a: marginal_counts(arms[a][1] if a != "ff" else cnt, o, nb) for a in ("ff", "off", "floor")}
                c = _single_card(sub, dict(info, clbits=nb), part["ff"], part["off"], part["floor"], sent)
                c["bundle"] = dict(tag=tag, bit_offset=o, segment=info["segment"])
                res[sub] = c
            continue
        res[tag] = _single_card(tag, rec, cnt, arms["off"][1], arms["floor"][1], sent)
    return res


def _single_card(tag, rec, cnt, off_counts, floor_counts, sent):
    """The card of one item from its three arms' counts (the body of cards() before bundles, unchanged)."""
    n = rec["clbits"]
    if "rus" in rec:
        return rus_card(tag, rec, cnt, off_counts, floor_counts, sent)
    succ = {int(b): v for b, v in rec["success"].items()}
    ff = success_share(cnt, n, succ)
    off = success_share(off_counts, n, succ)
    fl = floor_counts
    tot = sum(fl.values())
    ones = [sum(c for key, c in fl.items() if _bits(key, n)[i] == 1) / tot for i in range(n)]
    zero = sum(c for key, c in fl.items() if set(_bits(key, n)) == {0}) / tot
    bound = rec.get("bound")
    card = dict(tag=tag, chain=rec["chain"], ff=ff, off=off,
                floor=dict(all_zero=zero, ones_per_bit=ones), bound=bound, sentinel=sent)
    if bound is not None:
        se = ff["two_se"] / 2
        card["above_bound_in_se"] = (ff["share"] - bound) / se if se > 0 else None
    if len(succ) == 1:
        fb = next(iter(succ))
        mids = [i for i in range(n) if i != fb]
        card["off_best_fixed_rule_after_the_fact"] = best_fixed_rule(off_counts, n, fb, mids)
    return card
