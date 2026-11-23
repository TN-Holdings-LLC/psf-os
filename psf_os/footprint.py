"""The wait footprint of a circuit with mid-circuit measurements (2026-11-20, after-2026-11-20; a measure, not a rule).

While a mid-circuit measurement is read and a conditional waits on it, every other qubit waits too. A qubit in a
computational basis state (|0> or |1>, in every branch) loses only by decay, which the device's one-sided errors can
even favour (the 2026-11-16 diagnosis: hold the likely value as |0>); a qubit in a superposition or entangled with
others also loses its phase. The footprint counts, at each mid-circuit measurement group, the waiting qubits that are
NOT in a computational basis state ("carriers"). It reads the circuit only: no noise model, no calibration number.

Why (the workplace, 2026-11-20): the same distance-3 repetition-code test written three ways kept the logical state
on 3 qubits (16 and 10 CZ) or on 1 qubit (6 CZ, the encoding undone before the syndrome is read) during the
conditional wait; under an illustrative dephasing per wait the 1-qubit form lost about a fifth as much. Whether the
device's loss follows the footprint or the CZ count is a question for a declared test; this module only counts.

  wait_footprint(qc) -> dict(waits=[dict(at, measured, carriers, idle_basis)], max=..., total=..., carrier_waits=...)
  python -m psf_os.footprint CIRCUITS.qpy        one line per circuit

Exact: branching over every mid-circuit measurement (statevector per branch), so up to MAX_QUBITS active qubits.
A carrier in ANY branch counts (the footprint of the worst branch, per wait). Measurements at the end (nothing but
measurements and barriers after them on any qubit) are not waits.

Beside the carriers, each wait also gives `excited`: the expected number of waiting qubits in |1> (the population of
|1> summed over the waiting qubits, averaged over the branches with their probabilities), the part that decays
(2026-11-16: holding the likely value as |0> recovered 0.11-0.22); `excited_waits` sums it over the waits.
"""
from __future__ import annotations

import numpy as np

MAX_QUBITS = 14
MAX_WORK = 2 ** 20      # branches x 2^qubits held at once (after-2026-11-20: a bound, so a build never stalls on it;
                        # about 0.5 s per circuit at worst on the workplace container)
EPS = 1e-9


def _active(qc):
    from psf_os.feedforward import walk
    return sorted({q for n, qs in walk(qc) if n not in ("barrier", "delay") for q in qs})


def _is_basis(rho):
    """True when a one-qubit density matrix is |0><0| or |1><1| (to EPS)."""
    return abs(rho[0, 1]) < EPS and min(rho[0, 0].real, rho[1, 1].real) < EPS


def _reduced(sv, k, n):
    """One-qubit reduced density matrix of qubit k from an n-qubit statevector (little-endian, as qiskit)."""
    psi = np.asarray(sv).reshape([2] * n)               # axis i is qubit n-1-i
    ax = n - 1 - k
    psi = np.moveaxis(psi, ax, 0).reshape(2, -1)
    return psi @ psi.conj().T


def fence_waits(qc, conditionals=False):
    """A copy of qc with a barrier on every qubit before each mid-circuit measurement group and after each run of
    conditionals (after-2026-11-20, the workplace VS's finding of 2026-11-20): the compiler then cannot merge a gate
    before a wait with one after it (on the compiled IPE it merged the target's return to |1> with the next round's
    preparation, so the target waited in a superposition), and nothing on a waiting qubit is moved into the wait.
    The circuit's action is unchanged (barriers only).
    conditionals (after-2026-11-20, the workplace VS's その6: default False, as checked): also a barrier on every
    qubit before each run of conditionals that does not follow a read directly (IPE's phase feedback, decided on
    bits read rounds before). Without it the compiler moves the next round's gates into such a run and the target
    waits there in a superposition."""
    out = qc.copy_empty_like()
    data = list(qc.data)
    allq = list(qc.qubits)

    def mid_after(i):
        return any(ins.operation.name not in ("measure", "barrier") for ins in data[i + 1:])

    prev = None
    for i, ins in enumerate(data):
        name = ins.operation.name
        if name == "measure" and prev not in ("measure", "barrier_fence") and mid_after(i):
            out.barrier(allq)
        if prev == "if_else" and name != "if_else":
            out.barrier(allq)
        if conditionals and name == "if_else" and prev not in ("if_else", "measure", "barrier"):
            out.barrier(allq)
        out.append(ins.operation, ins.qubits, ins.clbits)
        prev = name
    return out


def wait_footprint(qc, span=False, conditionals=False):
    """The footprint at each mid-circuit measurement group of qc (top level; conditional blocks are applied per
    branch). Returns dict(waits, max, total, carrier_waits, qubits) with qubits the active qubits (compacted).
    span (after-2026-11-20; default False: as before): a waiting qubit counts when it is not in a basis state at the
    measurement OR after any instruction up to the end of the conditionals that follow (in program order): the whole
    wait, not only its start (the workplace VS counted at every measurement and conditional test). For a circuit as
    sent (compiled), span=True is the honest count; without fence_waits, gates after the conditionals on other qubits
    may still be scheduled into the wait (not counted).
    conditionals (after-2026-11-20, as the workplace VS counts; default False): a run of conditionals that does not
    follow a read directly (e.g. IPE's phase feedback on bits read rounds before) is a wait too: the qubits outside
    its blocks wait while it is decided; recorded with measured=[] and kind "conditional"."""
    from qiskit.quantum_info import Statevector
    act = _active(qc)
    if len(act) > MAX_QUBITS:
        raise ValueError(f"{len(act)} active qubits: above {MAX_QUBITS} for the exact footprint")
    pos = {q: i for i, q in enumerate(act)}
    n = len(act)
    if n == 0:
        return dict(waits=[], max=0, total=0, carrier_waits=0, qubits=[])
    ncl = qc.num_clbits
    data = list(qc.data)

    def later_ops(i):
        return any(ins.operation.name not in ("measure", "barrier") for ins in data[i + 1:])

    branches = [(1.0, Statevector.from_label("0" * n), [0] * ncl)]
    waits = []
    pending = []          # span: waits still open (until the run of conditionals after them ends)
    seen_cond = [False]

    def recheck():
        for wi in pending:
            w = waits[wi]
            meas = {pos[q] for q in w["measured"]}
            for _p, sv, _b in branches:
                for k in range(n):
                    if k in meas or act[k] in w["carriers"]:
                        continue
                    if not _is_basis(_reduced(sv.data, k, n)):
                        w["carriers"] = sorted(w["carriers"] + [act[k]])
                        w["idle_basis"] -= 1
    i = 0
    while i < len(data):
        ins = data[i]
        op = ins.operation
        name = op.name
        if span and pending and name not in ("if_else", "barrier", "delay") and seen_cond[0]:
            pending.clear()               # the conditionals after the wait are over
            seen_cond[0] = False
        if name in ("barrier", "delay"):
            i += 1
            continue
        if name == "measure":
            group = []
            j = i
            while j < len(data) and data[j].operation.name in ("measure", "barrier"):
                if data[j].operation.name == "measure":
                    group.append(data[j])
                j += 1
            mid = later_ops(j - 1)
            measured = [pos[qc.find_bit(g.qubits[0]).index] for g in group]
            if span:
                pending.clear()           # a new read closes the waits before it
                seen_cond[0] = False
            if mid and span:
                pending.append(len(waits))
            if mid:
                carriers = set()
                excited = 0.0
                for p, sv, _ in branches:
                    for k in range(n):
                        if k in measured:
                            continue
                        rho = _reduced(sv.data, k, n)
                        excited += p * float(rho[1, 1].real)
                        if not _is_basis(rho):
                            carriers.add(k)
                waits.append(dict(at=i, measured=[act[k] for k in measured], carriers=sorted(act[k] for k in carriers),
                                  idle_basis=n - len(measured) - len(carriers), excited=excited))
            for g in group:
                k = pos[qc.find_bit(g.qubits[0]).index]
                cb = qc.find_bit(g.clbits[0]).index
                new = []
                for p, sv, bits in branches:
                    probs = sv.probabilities([k])
                    for v in (0, 1):
                        if probs[v] < 1e-15:
                            continue
                        proj = np.zeros((2, 2))
                        proj[v, v] = 1.0
                        s2 = sv.evolve(proj, [k])
                        s2 = Statevector(s2.data / np.sqrt(probs[v]))
                        b2 = list(bits)
                        b2[cb] = v
                        new.append((p * probs[v], s2, b2))
                branches = new
                if len(branches) * 2 ** n > MAX_WORK:
                    raise ValueError(f"{len(branches)} branches on {n} qubits: above the bound {MAX_WORK} for the "
                                     "exact footprint")
            i = j
            if span:
                recheck()
            continue
        if name == "if_else" and conditionals and i > 0 and data[i - 1].operation.name not in ("if_else", "measure") \
                and not (data[i - 1].operation.name == "barrier" and i > 1 and
                         data[i - 2].operation.name in ("measure", "if_else")):
            inside = set()
            j2 = i
            while j2 < len(data) and data[j2].operation.name == "if_else":
                inside |= {pos[qc.find_bit(q).index] for q in data[j2].qubits}
                j2 += 1
            carriers = set()
            excited = 0.0
            for p_, sv, _ in branches:
                for k in range(n):
                    if k in inside:
                        continue
                    rho = _reduced(sv.data, k, n)
                    excited += p_ * float(rho[1, 1].real)
                    if not _is_basis(rho):
                        carriers.add(k)
            waits.append(dict(at=i, measured=[], kind="conditional", carriers=sorted(act[k] for k in carriers),
                              idle_basis=n - len(inside) - len(carriers), excited=excited))
        if name == "if_else":
            seen_cond[0] = True
            tgt, val = op.condition
            if type(tgt).__name__ == "ClassicalRegister":
                idx = [qc.find_bit(x).index for x in tgt]
            else:
                idx = [qc.find_bit(tgt).index]
            body = op.blocks[0]
            qmap = [pos[qc.find_bit(q).index] for q in ins.qubits]
            new = []
            for p, sv, bits in branches:
                v = sum(bits[b] << t for t, b in enumerate(idx))
                if v == int(val):
                    for bi in body.data:
                        if bi.operation.name in ("barrier", "delay"):
                            continue
                        sv = sv.evolve(bi.operation, [qmap[body.find_bit(q).index] for q in bi.qubits])
                new.append((p, sv, bits))
            branches = new
            i += 1
            if span:
                recheck()
            continue
        qs = [pos[qc.find_bit(q).index] for q in ins.qubits]
        branches = [(p, sv.evolve(op, qs), bits) for p, sv, bits in branches]
        i += 1
        if span and pending:
            recheck()
    counts = [len(w["carriers"]) for w in waits]
    return dict(waits=waits, max=max(counts, default=0), total=sum(counts), carrier_waits=sum(counts),
                excited_waits=sum(w["excited"] for w in waits), qubits=act)


def footprint_or_none(qc, span=True, conditionals=True):
    """wait_footprint(qc, span) summarised for a record: dict(counted=True, max, total, carriers per wait), or
    dict(counted=False, why) when the circuit is above the bounds (MAX_QUBITS active qubits, MAX_WORK). Never raises
    for size: a build that records the footprint goes on without it."""
    try:
        f = wait_footprint(qc, span=span, conditionals=conditionals)
    except ValueError as e:
        return dict(counted=False, why=str(e))
    return dict(counted=True, max=f["max"], total=f["total"], excited=round(f["excited_waits"], 6),
                carriers=[w["carriers"] for w in f["waits"]])


# ------------------------------------------------------------------------------------------------ parking (park-default)
# park-default-2026-11-21 (the owner's decision of 2026-11-21 9:59 JST, after the weekend's test C: waiting in |0>
# kept +0.0387 (2 SE 0.0045) more on IPE n = 4, waiting in |1> the same as the superposition; one night, one device).
# A qubit that waits in |1> in every branch during a mid-circuit measurement or a run of conditionals is put in |0>
# for the wait: X before the wait's fence, X after it. Qubits in |0>, carriers (superposition, entangled) and qubits
# whose basis value depends on the branch are left as they are.

def exact_distribution(qc):
    """The exact distribution of qc's classical bits (branching over every measurement and reset; conditional blocks
    applied per branch), {bit string (clbit 0 rightmost): probability}. Up to MAX_QUBITS active qubits and MAX_WORK."""
    branches, _ = _simulate(qc, ())
    out = {}
    ncl = qc.num_clbits
    for p, _sv, bits in branches:
        key = "".join(str(bits[ncl - 1 - i]) for i in range(ncl))
        out[key] = out.get(key, 0.0) + p
    return out


def _simulate(qc, stops):
    """Branches (p, statevector over the active qubits, clbits) after the whole circuit, and at each instruction
    index in `stops` (before it) the branch list then."""
    from qiskit.quantum_info import Statevector
    act = _active(qc)
    if len(act) > MAX_QUBITS:
        raise ValueError(f"{len(act)} active qubits: above {MAX_QUBITS}")
    pos = {q: i for i, q in enumerate(act)}
    n = max(len(act), 1)
    ncl = qc.num_clbits
    branches = [(1.0, Statevector.from_label("0" * n), [0] * ncl)]
    at = {}
    stops = set(stops)

    def collapse(brs, k, cb):
        new = []
        for p, sv, bits in brs:
            probs = sv.probabilities([k])
            for v in (0, 1):
                if probs[v] < 1e-15:
                    continue
                proj = np.zeros((2, 2))
                proj[v, v] = 1.0
                s2 = Statevector(sv.evolve(proj, [k]).data / np.sqrt(probs[v]))
                b2 = list(bits)
                if cb is not None:
                    b2[cb] = v
                elif v == 1:                                   # reset: flip the |1> branch back to |0>
                    s2 = s2.evolve(np.array([[0, 1], [1, 0]]), [k])
                new.append((p * probs[v], s2, b2))
        if len(new) * 2 ** n > MAX_WORK:
            raise ValueError(f"{len(new)} branches on {n} qubits: above the bound {MAX_WORK}")
        return new

    def apply(brs, circ, qmap, ins_list):
        for bi in ins_list:
            name = bi.operation.name
            if name in ("barrier", "delay"):
                continue
            qs = [qmap[circ.find_bit(q).index] for q in bi.qubits]
            if name == "measure":
                raise ValueError("a measurement inside a conditional block is not supported here")
            brs = [(p, sv.evolve(bi.operation, qs), b) for p, sv, b in brs]
        return brs

    top = {qc.find_bit(q).index: pos.get(qc.find_bit(q).index) for q in qc.qubits}
    for i, ins in enumerate(qc.data):
        if i in stops:
            at[i] = list(branches)
        op = ins.operation
        name = op.name
        if name in ("barrier", "delay"):
            continue
        if name == "measure":
            k = pos[qc.find_bit(ins.qubits[0]).index]
            branches = collapse(branches, k, qc.find_bit(ins.clbits[0]).index)
            continue
        if name == "reset":
            branches = collapse(branches, pos[qc.find_bit(ins.qubits[0]).index], None)
            continue
        if name == "if_else":
            tgt, val = op.condition
            idx = ([qc.find_bit(x).index for x in tgt] if type(tgt).__name__ == "ClassicalRegister"
                   else [qc.find_bit(tgt).index])
            new = []
            for p, sv, bits in branches:
                v = sum(bits[b] << t for t, b in enumerate(idx))
                blk = op.blocks[0] if v == int(val) else (op.blocks[1] if len(op.blocks) > 1 else None)
                if blk is not None:
                    qmap = {blk.find_bit(bq).index: top[qc.find_bit(q).index] for bq, q in zip(blk.qubits, ins.qubits)}
                    sv = apply([(p, sv, bits)], blk, qmap, blk.data)[0][1]
                new.append((p, sv, bits))
            branches = new
            continue
        qs = [pos[qc.find_bit(q).index] for q in ins.qubits]
        branches = [(p, sv.evolve(op, qs), bits) for p, sv, bits in branches]
    return branches, dict(at=at, act=act, pos=pos)


def _windows(f):
    """The waits of a fenced circuit f (fence_waits(qc, conditionals=True)): a full fence followed by a measurement
    group (and the run of conditionals after it, up to the fence that closes it) or by a run of conditionals.
    Returns [dict(start, end, close, kind, involved)]: start the index of the opening fence; end the index after
    which the wait is over (the closing fence, or the last measurement when no conditional follows: close=False,
    a fence is then added after it); involved the qubit indices measured or inside the conditionals."""
    data = list(f.data)
    nq = f.num_qubits
    full = lambda ins: ins.operation.name == "barrier" and len(ins.qubits) == nq
    out = []
    i = 0
    while i < len(data):
        if full(data[i]) and i + 1 < len(data) and data[i + 1].operation.name in ("measure", "if_else"):
            kind = "read" if data[i + 1].operation.name == "measure" else "conditional"
            involved = set()
            j = i + 1
            while j < len(data) and data[j].operation.name == "measure":
                involved.add(f.find_bit(data[j].qubits[0]).index)
                j += 1
            k = j
            while k < len(data) and data[k].operation.name == "if_else":
                involved |= {f.find_bit(q).index for q in data[k].qubits}
                k += 1
            if k > j and k < len(data) and full(data[k]):
                out.append(dict(start=i, end=k, close=True, kind=kind, involved=sorted(involved)))
                i = k
                continue
            if k == j and kind == "read":
                if not any(x.operation.name not in ("measure", "barrier") for x in data[j:]):
                    i = j                                      # final measurements: not a wait
                    continue
                out.append(dict(start=i, end=j - 1, close=False, kind=kind, involved=sorted(involved)))
                i = j
                continue
        i += 1
    return out


def _decide_always(qc):
    """(f, windows, plan) of park_waits: f = fence_waits(qc, conditionals=True), its windows, and per window the
    waiting qubits in |1> in every branch under "qubits"."""
    f = fence_waits(qc, conditionals=True)
    wins = _windows(f)
    if not wins:
        return f, wins, []
    _, sim = _simulate(f, [w["start"] for w in wins])
    act, pos = sim["act"], sim["pos"]
    plan = []
    for w in wins:
        brs = sim["at"][w["start"]]
        put = []
        for q in act:
            if q in w["involved"]:
                continue
            k = pos[q]
            if all(p < EPS or (abs(_reduced(sv.data, k, len(act))[1, 1].real - 1) < EPS) for p, sv, _ in brs):
                put.append(q)
        plan.append(dict(w, qubits=put))
    return f, wins, plan


def park_plan(qc):
    """0.3.3rc1: park_waits' plan as anchored waits for psf_os.compiled.check_parked ([] when nothing is parked)."""
    from psf_os.compiled import plan_anchors
    f, wins, plan = _decide_always(qc)
    return plan_anchors(f, plan, "always") if wins else []


def park_waits(qc):
    """(circuit, info): qc with every waiting qubit that is in |1> in every branch put in |0> for the wait (X before
    the wait's fence, X after its closing fence; the waits fenced as fence_waits(conditionals=True), so that the
    compiler can neither cancel the pair nor move gates into the wait). When no qubit is parked, qc itself is
    returned unchanged (not fenced) and info["parked"] is empty. Checked here: the exact distribution of the classical
    bits is the same as qc's (else ValueError). Above the footprint's bounds: ValueError (the caller keeps qc)."""
    f, wins, plan = _decide_always(qc)
    if not wins:
        return qc, dict(parked=[], waits=0)
    if not any(w["qubits"] for w in plan):
        return qc, dict(parked=[], waits=len(wins))
    before = {w["start"]: w["qubits"] for w in plan if w["qubits"]}
    after = {w["end"]: w for w in plan if w["qubits"]}
    out = f.copy_empty_like()
    for i, ins in enumerate(f.data):
        for q in before.get(i, ()):
            out.x(q)
        out.append(ins.operation, ins.qubits, ins.clbits)
        if i in after:
            w = after[i]
            if not w["close"]:
                out.barrier(list(out.qubits))
            for q in w["qubits"]:
                out.x(q)
    d0, d1 = exact_distribution(qc), exact_distribution(out)
    keys = set(d0) | set(d1)
    if any(abs(d0.get(k, 0.0) - d1.get(k, 0.0)) > 1e-9 for k in keys):
        raise ValueError("parking changed the noiseless distribution (not built)")
    e0 = wait_footprint(f, conditionals=True)["excited_waits"]
    e1 = wait_footprint(out, conditionals=True)["excited_waits"]
    info = dict(parked=[dict(kind=w["kind"], qubits=w["qubits"]) for w in plan if w["qubits"]],
                waits=len(wins), parked_waits=sum(1 for w in plan if w["qubits"]),
                excited_before=round(float(e0), 6), excited_after=round(float(e1), 6))
    return out, info


# ------------------------------------------------------------------------------------------------ the likely value
# likely-park-2026-11-22 (a 0.3.1 candidate; off by default). On ibm_fez (results/rus-fence-night-2026-11-21) parking
# the likely value, an X pair around each wait on every waiting qubit with P(|1>) > 0.5, gained on top of the full
# fence (NF +0.088), while the full fence itself cost on P0 and P3 (-0.045, -0.024). Here the pair is kept by a partial
# fence instead: at the waits where a qubit is parked only, a barrier on the parked qubits and the wait's own qubits
# (measured there or inside its conditionals). Every other qubit is left to the compiler as without fences.

def _fence_inserted(qc, f):
    """Indices of f's instructions that fence_waits added to qc (f is qc with barriers inserted)."""
    j, ins = 0, set()
    qd = list(qc.data)
    for i, x in enumerate(f.data):
        if j < len(qd) and x.operation == qd[j].operation and list(x.qubits) == list(qd[j].qubits) and \
                list(x.clbits) == list(qd[j].clbits):
            j += 1
        else:
            ins.add(i)
    if j != len(qd):
        raise ValueError("fence_waits did more than add barriers")
    return ins


def _decide_likely(qc, threshold=0.5):
    """(f, windows, plan) of park_likely: per window the waiting qubits with P(|1>) above threshold under "flip"
    as (qubit, p1)."""
    f = fence_waits(qc, conditionals=True)
    wins = _windows(f)
    if not wins:
        return f, wins, []
    _, sim = _simulate(f, [w["start"] for w in wins])
    act, pos = sim["act"], sim["pos"]
    n = len(act)
    plan = []
    for w in wins:
        flip = []
        for q in act:
            if q in w["involved"]:
                continue
            p1 = sum(p * float(_reduced(sv.data, pos[q], n)[1, 1].real) for p, sv, _ in sim["at"][w["start"]])
            if p1 > threshold + 1e-9:
                flip.append((q, p1))
        plan.append(dict(w, flip=flip))
    return f, wins, plan


def park_likely_plan(qc, threshold=0.5):
    """0.3.3rc1: park_likely's plan as anchored waits for psf_os.compiled.check_parked ([] when nothing is parked)."""
    from psf_os.compiled import plan_anchors
    f, wins, plan = _decide_likely(qc, threshold)
    return plan_anchors(f, [dict(w, qubits=[q for q, _ in w["flip"]]) for w in plan], "likely") if wins else []


def park_likely(qc, threshold=0.5):
    """(circuit, info): an X before and after each wait on every waiting qubit (neither measured there nor inside its
    conditionals) whose noiseless probability of |1>, over the branches with their weights, is above `threshold`
    (entangled qubits included: X X = I across a wait the qubit takes no part in). The pair is kept by a partial fence
    at those waits only (the parked qubits and the wait's own qubits); no other barrier is added. Nothing to park: qc
    itself. Checked here: the exact distribution of the classical bits unchanged (else ValueError)."""
    f, wins, plan = _decide_likely(qc, threshold)
    if not wins:
        return qc, dict(parked=[], waits=0)
    parked = [w for w in plan if w["flip"]]
    if not parked:
        return qc, dict(parked=[], waits=len(wins))
    ins = _fence_inserted(qc, f)
    starts = {w["start"]: w for w in parked}
    ends = {w["end"]: w for w in parked}
    allq = list(f.qubits)
    out = f.copy_empty_like()
    for i, x in enumerate(f.data):
        w = starts.get(i)
        if w is not None:
            qs = [q for q, _ in w["flip"]]
            for q in qs:
                out.x(allq[q])
            out.barrier([allq[q] for q in sorted(set(qs) | set(w["involved"]))])
        if i not in ins:
            out.append(x.operation, x.qubits, x.clbits)
        w = ends.get(i)
        if w is not None:
            qs = [q for q, _ in w["flip"]]
            out.barrier([allq[q] for q in sorted(set(qs) | set(w["involved"]))])
            for q in qs:
                out.x(allq[q])
    d0, d1 = exact_distribution(qc), exact_distribution(out)
    if any(abs(d0.get(k, 0.0) - d1.get(k, 0.0)) > 1e-9 for k in set(d0) | set(d1)):
        raise ValueError("parking the likely value changed the noiseless distribution (not built)")
    info = dict(parked=[dict(kind=w["kind"], qubits=[q for q, _ in w["flip"]],
                             p1=[round(p, 6) for _, p in w["flip"]]) for w in parked],
                waits=len(wins), parked_waits=len(parked), threshold=threshold, fence="partial")
    return out, info


def main(argv=None):
    import sys
    from qiskit import qpy
    args = sys.argv[1:] if argv is None else argv
    for path in args:
        with open(path, "rb") as f:
            for k, c in enumerate(qpy.load(f)):
                try:
                    r = wait_footprint(c)
                    print(f"{path} [{k}] {c.name}: waits {len(r['waits'])}, carriers per wait "
                          f"{[len(w['carriers']) for w in r['waits']]}, max {r['max']}, total {r['total']}")
                except ValueError as e:
                    print(f"{path} [{k}] {c.name}: {e}")


if __name__ == "__main__":
    main()
