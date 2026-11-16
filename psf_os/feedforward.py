"""Feed-forward jobs (2026-11-16; owner: the test that no after-the-fact processing can reach). A new job kind beside
user2d: circuits with mid-circuit measurements and conditional Pauli gates applied in real time. user2d is not changed
and still refuses any gate after a measurement.

Accepted (narrow on purpose; anything else is refused with the reason):
  * qubits 0 .. n-1 on a line: two-qubit gates are CZ between neighbours (k, k+1) only;
  * one-qubit gates (h, x, y, z, s, sdg, t, tdg, sx, sxdg, rx, ry, rz, p, id);
  * measurements anywhere, one classical register; each classical bit written once (a definite record per shot);
  * `if_test` blocks on one classical bit already written, value 0 or 1, holding only X or Z on one qubit, no else.

Placement: the n-site chain by rule C (psf_os.chain.place: the coupling graph and the failed marks only, no error
number), logical qubit k on chain[k]. Compilation: the map-free Target with the control-flow op if_else added (no
property, no duration), Qiskit's level-1 pass manager, layout fixed, no routing (as compile.RuleCompiler).

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


def analyse(qc):
    """dict(n, clbits, mid, conditionals, cz) or Refused. Checks the accepted form (module docstring)."""
    if len(qc.cregs) != 1:
        raise Refused(f"one classical register holding every measured bit (got {len(qc.cregs)})")
    if qc.num_clbits != len(qc.cregs[0]):
        raise Refused("every classical bit must belong to the one register")
    written, measured_last, cz, conds, mids = set(), {}, 0, 0, 0
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
            b, _ = _cond_bit(op, qc)
            if b not in written:
                raise Refused(f"a conditional tests bit {b} before it is measured")
            if len(op.blocks) != 1 or op.blocks[0] is None:
                raise Refused("a conditional has no else branch here")
            body = op.blocks[0]
            for bi in body.data:
                if bi.operation.name not in CONDITIONAL or len(bi.qubits) != 1:
                    raise Refused(f"a conditional block holds only X or Z on one qubit (got '{bi.operation.name}')")
            conds += 1
            continue
        if getattr(op, "condition", None) is not None:
            raise Refused("conditions are written as if_test blocks")
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
    last_op = max((i for i, nm in enumerate(ops) if nm not in ("measure", "barrier")), default=-1)
    mids = sum(1 for i, nm in enumerate(ops) if nm == "measure" and i < last_op)   # an operation follows it
    return dict(n=qc.num_qubits, clbits=qc.num_clbits, mid=mids, conditionals=conds, cz=cz)


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


def build(dev, items, out, shots=4000, max_seconds=60, sentinel_spec=None):
    """items: [(tag, QuantumCircuit, dict(success={clbit: value}, bound=float | None))]. One job: per item its ff, off
    and floor arms, one ring sentinel. Nothing is written if any item is refused. Returns the build report (also
    out/build.json)."""
    from psf_os import job as J
    from psf_os import user2d as U
    if os.path.exists(out):
        raise Refused(f"{out} exists (nothing is overwritten)")
    tags = [t for t, _, _ in items]
    if not items or len(set(tags)) != len(tags) or not all(U._tag_ok(t) for t in tags):
        raise Refused(f"tags must be distinct and use letters, digits, '_', '.', '-' only: {tags}")
    comp = Compiler(dev)
    prepared = []
    for tag, qc, spec in items:
        try:
            a = analyse(qc)
        except Refused as e:
            raise Refused(f"{tag}: {e}") from None
        succ = {int(b): int(v) for b, v in (spec or {}).get("success", {}).items()}
        if not succ or any(b >= a["clbits"] or v not in (0, 1) for b, v in succ.items()):
            raise Refused(f"{tag}: success must name classical bits of the circuit and their values 0 / 1")
        pl = place(dev, a["n"])
        chain = pl["chain"]
        arms = {}
        for arm, logical in (("ff", qc), ("off", strip_conditionals(qc))):
            arms[arm] = comp.run(on_device(logical, chain, dev.num_qubits))
        arms["floor"] = floor_of(arms["ff"])
        chk = {arm: checks(dev, c, chain) for arm, c in arms.items()}
        if not all(all(v.values()) for v in chk.values()):
            raise Refused(f"{tag}: acceptance failed: {chk}")
        if not any(n == "if_else" for n, _ in walk(arms["ff"])) and a["conditionals"]:
            raise Refused(f"{tag}: the compiler removed the conditionals")
        prepared.append((tag, a, succ, (spec or {}).get("bound"), pl, arms, chk))
    runs = os.path.join(out, "runs")
    os.makedirs(runs)
    rep_items = {}
    for tag, a, succ, bound, pl, arms, chk in prepared:
        chain = pl["chain"]
        labels = list(range(a["n"]))
        mapping = {k: chain[k] for k in labels}
        common = dict(labels=labels, mapping=mapping, shots=shots)
        extra = dict(tag=tag, chain=chain, chain_rule=dict(rule=pl["rule"], start=pl["start"]), success=
                     {str(b): v for b, v in succ.items()}, bound=bound, clbits=a["clbits"], mid=a["mid"],
                     conditionals=a["conditionals"], logical_cz=a["cz"])
        for arm, kind in (("ff", KIND), ("off", KIND + "_off"), ("floor", KIND + "_floor")):
            c = arms[arm]
            U._record(dev, runs, f"ff_{tag}" + ("" if arm == "ff" else f"_{arm}"), "user" if arm == "ff" else arm,
                      kind, [c], [dict(input="f", setting="z")],
                      extra=dict(extra, ff_arm=arm, cz=cz_count(c), checks=chk[arm]), valid=True, **common)
        c = arms["ff"]
        rep_items[tag] = dict(n=a["n"], chain=chain, cz=cz_count(c), off_cz=cz_count(arms["off"]),
                              floor_cz=cz_count(arms["floor"]), depth=c.depth(),
                              mid=a["mid"], conditionals=a["conditionals"], success=succ, bound=bound)
    _sentinel(dev, runs, shots, sentinel_spec)
    man = J.assemble(dev, runs, os.path.join(out, "job"), max_seconds=max_seconds)
    rep = dict(device=dev.name, kind=KIND, items=rep_items, circuits=len(man["pubs"]),
               qpu_seconds_model=man["qpu_seconds_model"], shots=shots,
               note="the time model does not include mid-circuit measurement and feed-forward latency (microseconds "
                    "per shot), small against its per-shot term")
    json.dump(rep, open(os.path.join(out, "build.json"), "w"), indent=1, default=str)
    return rep


# ------------------------------------------------------------------------------------------------ local run, card

def run_local(out, seed=20261116):
    """Noiseless Aer on the job's circuits (compacted to their touched qubits; statevector up to 24, else matrix product
    state; control flow supported); raw_counts.json and
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
            meth = "statevector" if cc.num_qubits <= 24 else "matrix_product_state"
        res = AerSimulator(method=meth).run(cc, shots=man["shots"], seed_simulator=seed + k).result()
        rows.append(dict(pub=k, counts={key.replace(" ", ""): v for key, v in res.get_counts(0).items()}))
    from psf_os.ibm import _stamp
    t = _stamp()
    json.dump(dict(job_id=f"local-noiseless-{seed}", submitted=t, local=True, backend=man["fetch"]["backend"],
                   simulator="aer noiseless (statevector / mps, compacted)"), open(sub, "w"), indent=1)
    json.dump(dict(fetched=t, job_id=f"local-noiseless-{seed}", status="DONE (local noiseless)", rows=rows),
              open(os.path.join(job, "raw_counts.json"), "w"))
    return f"local-noiseless-{seed}"


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
        n = rec["clbits"]
        succ = {int(b): v for b, v in rec["success"].items()}
        ff = success_share(cnt, n, succ)
        off = success_share(arms["off"][1], n, succ)
        fl = arms["floor"][1]
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
            card["off_best_fixed_rule_after_the_fact"] = best_fixed_rule(arms["off"][1], n, fb, mids)
        res[tag] = card
    return res
