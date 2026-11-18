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

def _sentinel(dev, runs, shots, spec, ring=40):
    import numpy as np
    from psf_os import circuits as C
    from psf_os import sentinel
    from psf_os.compile import RuleCompiler, measured_qubits
    rc = RuleCompiler(dev)
    mq = measured_qubits(rc.compile(C.model_circuit(list(np.zeros(C.n_model_angles(ring, 1))), ring, 1)))
    sentinel.build(dev, runs, [int(mq[j]) for j in range(ring)], shots, rc, spec=spec)


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


def build(dev, items, out, shots=4000, max_seconds=60, sentinel_spec=None, chains=None):
    """items: [(tag, QuantumCircuit, spec)], spec either dict(success={clbit: value}, bound=float | None) (a success
    test such as teleport_test) or the dict rus_chain returns (kind "rus": out_bit, tries per neuron, exact, form
    "blocks"). One job: per item its ff, off and floor arms (for "rus" the off arm is also read post-selected:
    the "post" view, no extra circuit), one ring sentinel. Nothing is written if any item is refused. Returns the build report (also
    out/build.json).
    chains (2026-11-17): {tag: [physical qubits]} places those items on the given chains (given_chain checks them);
    items not named are placed by rule C as before."""
    from psf_os import job as J
    from psf_os import user2d as U
    if os.path.exists(out):
        raise Refused(f"{out} exists (nothing is overwritten)")
    tags = [t for t, _, _ in items]
    if not items or len(set(tags)) != len(tags) or not all(U._tag_ok(t) for t in tags):
        raise Refused(f"tags must be distinct and use letters, digits, '_', '.', '-' only: {tags}")
    chains = dict(chains or {})
    if set(chains) - set(tags):
        raise Refused(f"chains given for unknown items {sorted(set(chains) - set(tags))}")
    comp = Compiler(dev)
    prepared = []
    for tag, qc, spec in items:
        a, qc, off, succ, spec = _prepare(tag, qc, spec)
        try:
            pl = given_chain(dev, chains[tag], a["n"]) if tag in chains else place(dev, a["n"])
        except Refused as e:
            raise Refused(f"{tag}: {e}") from None
        chain = pl["chain"]
        arms = {}
        for arm, logical in (("ff", qc), ("off", off)):
            arms[arm] = comp.run(on_device(logical, chain, dev.num_qubits))
        arms["floor"] = floor_of(arms["ff"])
        chk = {arm: checks(dev, c, chain) for arm, c in arms.items()}
        if not all(all(v.values()) for v in chk.values()):
            raise Refused(f"{tag}: acceptance failed: {chk}")
        if not any(n == "if_else" for n, _ in walk(arms["ff"])) and a["conditionals"]:
            raise Refused(f"{tag}: the compiler removed the conditionals")
        prepared.append((tag, a, succ, spec, pl, arms, chk))
    runs = os.path.join(out, "runs")
    os.makedirs(runs)
    rep_items = {}
    for tag, a, succ, spec, pl, arms, chk in prepared:
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
            c = arms[arm]
            U._record(dev, runs, f"ff_{tag}" + ("" if arm == "ff" else f"_{arm}"), "user" if arm == "ff" else arm,
                      kind, [c], [dict(input="f", setting="z")],
                      extra=dict(extra, ff_arm=arm, cz=cz_count(c), checks=chk[arm]), valid=True, **common)
        c = arms["ff"]
        rep_items[tag] = dict(n=a["n"], chain=chain, cz=cz_count(c), off_cz=cz_count(arms["off"]),
                              floor_cz=cz_count(arms["floor"]), depth=c.depth(),
                              mid=a["mid"], conditionals=a["conditionals"], success=succ, bound=bound,
                              **({"rus": extra["rus"], "cz_in_blocks": cz_count(c) - cz_count(arms["off"])}
                                 if "rus" in extra else {}))
    _sentinel(dev, runs, shots, sentinel_spec)
    man = J.assemble(dev, runs, os.path.join(out, "job"), max_seconds=max_seconds)
    rep = dict(device=dev.name, kind=KIND, items=rep_items, circuits=len(man["pubs"]),
               qpu_seconds_model=man["qpu_seconds_model"], shots=shots,
               note="the time model does not include mid-circuit measurement and feed-forward latency (microseconds "
                    "per shot), small against its per-shot term")
    json.dump(rep, open(os.path.join(out, "build.json"), "w"), indent=1, default=str)
    return rep


# ------------------------------------------------------------------------------------------------ bundled items
# 2026-11-17 (the packing test, notebook notes/next_2026-11-17/README.md "Bundled items"): independent items side by
# side in one circuit per arm. Each sub-item is read as a single item is (its own spec, its own off arm); the bundle is
# one ff, one off and one floor circuit plus the job's sentinel, whatever the number of sub-items.

BUNDLE_GAP = 1


def bundle_place(dev, sizes, gap=BUNDLE_GAP):
    """One rule-C chain of N sites and a segment per item: items laid in order on consecutive sites; after an item
    `gap` sites are skipped, and further sites while the next item's first site would be joined by any coupler of the
    device (working or failed) to a qubit of an earlier item. N is the smallest length that holds every item.
    Returns dict(chain, segments (chain positions), rule, start); Refused if no rule-C chain holds them."""
    from psf_os import chain as CH
    adj = {}
    for a, b in dev.all_couplers():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    N = sum(sizes) + gap * (len(sizes) - 1)
    while N <= dev.num_qubits:
        try:
            pl = CH.place(dev, N)
        except CH.Refused as e:
            raise Refused(f"bundle of sizes {sizes}: no rule-C chain of {N} sites ({e})") from None
        ch = list(pl["chain"])
        segs, used, pos = [], set(), 0
        for n in sizes:
            while pos + n <= N and any(q in used or adj.get(q, set()) & used for q in ch[pos:pos + n]):
                pos += 1
            if pos + n > N:
                break
            segs.append(list(range(pos, pos + n)))
            used |= set(ch[pos:pos + n])
            pos += n + gap
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


def build_bundle(dev, items, out, tag="pack", shots=4000, max_seconds=60, sentinel_spec=None):
    """items: [(sub_tag, QuantumCircuit, spec)] as for build. One job of one bundle: the sub-items placed by
    bundle_place, every sub-item on its segment and its bits at an offset of one classical register; ff = the
    sub-items' circuits, off = their off arms (explicit or the conditionals removed), floor = floor_of(ff), one ring
    sentinel. Acceptance: checks() of each arm on the bundle's chain, the conditionals kept, bundle_acceptance
    (independence), and per sub-item and arm the same wire_signature as the sub-item compiled alone on the same
    physical qubits. Records: one per arm (tag `tag`), each with the table bundle = {sub_tag: dict(segment, qubits,
    bit_offset, clbits, the sub-item's spec fields)}; cards() reads one card per sub-item from its own bits."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from psf_os import job as J
    from psf_os import user2d as U
    if os.path.exists(out):
        raise Refused(f"{out} exists (nothing is overwritten)")
    tags = [t for t, _, _ in items]
    if len(items) < 2 or len(set(tags)) != len(tags) or not all(U._tag_ok(t) for t in tags + [tag]):
        raise Refused(f"a bundle holds two or more distinct sub-items with plain tags: {tags}")
    prep = [_prepare(t, qc, spec) for t, qc, spec in items]
    sizes = [p[0]["n"] for p in prep]
    pl = bundle_place(dev, sizes)
    chain, segs = pl["chain"], pl["segments"]
    N = len(chain)
    offs = [0]
    for p in prep:
        offs.append(offs[-1] + p[0]["clbits"])
    logical = {}
    multi = any(len(p[1].cregs) > 1 for p in prep)
    for which, k in (("ff", 1), ("off", 2)):
        if multi:
            # 2026-11-18: a sub-item with several registers (a register condition) keeps them, renamed per sub-item and
            # laid one after another in bit order, so compose maps every register condition onto a register of the
            # bundle (one register "c" would make compose add an aliasing register: bits reported twice)
            regs = [ClassicalRegister(len(r), f"s{i}_{r.name}") for i, p in enumerate(prep) for r in p[1].cregs]
            c1 = QuantumCircuit(QuantumRegister(N, "q"), *regs)
        else:
            c1 = QuantumCircuit(QuantumRegister(N, "q"), ClassicalRegister(offs[-1], "c"))
        for i, p in enumerate(prep):
            c1.compose(p[k], qubits=segs[i], clbits=list(range(offs[i], offs[i + 1])), inplace=True)
        logical[which] = c1
    comp = Compiler(dev)
    arms = {w: comp.run(on_device(logical[w], chain, dev.num_qubits)) for w in ("ff", "off")}
    arms["floor"] = floor_of(arms["ff"])
    chk = {arm: checks(dev, c, chain) for arm, c in arms.items()}
    if not all(all(v.values()) for v in chk.values()):
        raise Refused(f"{tag}: acceptance failed: {chk}")
    if any(p[0]["conditionals"] for p in prep) and not any(n == "if_else" for n, _ in walk(arms["ff"])):
        raise Refused(f"{tag}: the compiler removed the conditionals")
    qsets = [set(chain[k] for k in s) for s in segs]
    bsets = [set(range(offs[i], offs[i + 1])) for i in range(len(prep))]
    bundle_acceptance(arms, qsets, bsets)
    table = {}
    for i, (t, p) in enumerate(zip(tags, prep)):
        a, qc, off, succ, spec = p
        phys = [chain[k] for k in segs[i]]
        bits = list(range(offs[i], offs[i + 1]))
        alone = {w: comp.run(on_device(lg, phys, dev.num_qubits)) for w, lg in (("ff", qc), ("off", off))}
        alone["floor"] = floor_of(alone["ff"])
        for w in ("ff", "off", "floor"):
            mine = restrict(arms[w], phys, bits)
            ref = restrict(alone[w], phys, list(range(a["clbits"])))
            if wire_signature(mine) != wire_signature(ref):
                raise Refused(f"{tag}: sub-item {t} ({w}) differs from the sub-item compiled alone on the same qubits")
        table[t] = dict(segment=segs[i], qubits=phys, bit_offset=offs[i], tag=t, chain=phys,
                        **_item_extra(t, a, succ, spec), cz=cz_count(restrict(arms["ff"], phys, bits)))
    runs = os.path.join(out, "runs")
    os.makedirs(runs)
    labels = list(range(N))
    common = dict(labels=labels, mapping={k: chain[k] for k in labels}, shots=shots)
    extra = dict(tag=tag, chain=chain, chain_rule=dict(rule=pl["rule"], start=pl["start"], bundle_gap=BUNDLE_GAP),
                 clbits=offs[-1], bundle=table)
    for arm, kind in (("ff", KIND), ("off", KIND + "_off"), ("floor", KIND + "_floor")):
        c = arms[arm]
        U._record(dev, runs, f"ff_{tag}" + ("" if arm == "ff" else f"_{arm}"), "user" if arm == "ff" else arm,
                  kind, [c], [dict(input="f", setting="z")],
                  extra=dict(extra, ff_arm=arm, cz=cz_count(c), checks=chk[arm]), valid=True, **common)
    _sentinel(dev, runs, shots, sentinel_spec)
    man = J.assemble(dev, runs, os.path.join(out, "job"), max_seconds=max_seconds)
    rep = dict(device=dev.name, kind=KIND, bundle=tag, chain=chain, segments=segs,
               items={t: dict(n=sizes[i], chain=table[t]["qubits"], bit_offset=offs[i], clbits=prep[i][0]["clbits"],
                              cz=table[t]["cz"]) for i, t in enumerate(tags)},
               circuits=len(man["pubs"]), qpu_seconds_model=man["qpu_seconds_model"], shots=shots,
               cz=dict(ff=cz_count(arms["ff"]), off=cz_count(arms["off"]), floor=cz_count(arms["floor"])))
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

def run_local(out, seed=20261116):
    """Noiseless Aer on the job's circuits (compacted to their touched qubits; Clifford circuits by the stabilizer
    method, others by statevector up to LOCAL_MAX_QUBITS touched qubits, else refused at once; control flow
    supported); raw_counts.json and
    submission.json as a device run writes them (local=True). For dry runs only: refused if a submission exists."""
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
                rows.append(dict(pub=k, counts=_run_parts(c, parts, man["shots"], seed + 1000 * k)))
                continue
            if cc.num_qubits > LOCAL_MAX_QUBITS:
                raise Refused(f"circuit {k} ({c.name}) is not Clifford and touches {cc.num_qubits} qubits: too slow to "
                              "simulate here (the ring sentinel at h = 0.6 on 40 qubits did not finish in 18 min). "
                              "For a dry run build the job with sentinel_spec=dict(kind='echo_clifford', D=24)")
            meth = "statevector"
        res = AerSimulator(method=meth).run(cc, shots=man["shots"], seed_simulator=seed + k).result()
        rows.append(dict(pub=k, counts={key.replace(" ", ""): v for key, v in res.get_counts(0).items()}))
    from psf_os.ibm import _stamp
    t = _stamp()
    json.dump(dict(job_id=f"local-noiseless-{seed}", submitted=t, local=True, backend=man["fetch"]["backend"],
                   simulator="aer noiseless (statevector / mps, compacted)"), open(sub, "w"), indent=1)
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


def _run_parts(qc, parts, shots, seed):
    """Noiseless shots of qc as the product of its independent parts (each by statevector, memory per shot)."""
    from qiskit_aer import AerSimulator
    nb = qc.num_clbits
    shots_bits = [["0"] * nb for _ in range(shots)]
    for j, (qs, cs) in enumerate(parts):
        sub = restrict(qc, qs, cs)
        if not cs:
            continue
        res = AerSimulator(method="statevector").run(sub, shots=shots, memory=True, seed_simulator=seed + j).result()
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
