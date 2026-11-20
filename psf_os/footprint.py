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


def fence_waits(qc):
    """A copy of qc with a barrier on every qubit before each mid-circuit measurement group and after each run of
    conditionals (after-2026-11-20, the workplace VS's finding of 2026-11-20): the compiler then cannot merge a gate
    before a wait with one after it (on the compiled IPE it merged the target's return to |1> with the next round's
    preparation, so the target waited in a superposition), and nothing on a waiting qubit is moved into the wait.
    The circuit's action is unchanged (barriers only)."""
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
        out.append(ins.operation, ins.qubits, ins.clbits)
        prev = name
    return out


def wait_footprint(qc, span=False):
    """The footprint at each mid-circuit measurement group of qc (top level; conditional blocks are applied per
    branch). Returns dict(waits, max, total, carrier_waits, qubits) with qubits the active qubits (compacted).
    span (after-2026-11-20; default False: as before): a waiting qubit counts when it is not in a basis state at the
    measurement OR after any instruction up to the end of the conditionals that follow (in program order): the whole
    wait, not only its start (the workplace VS counted at every measurement and conditional test). For a circuit as
    sent (compiled), span=True is the honest count; without fence_waits, gates after the conditionals on other qubits
    may still be scheduled into the wait (not counted)."""
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


def footprint_or_none(qc, span=True):
    """wait_footprint(qc, span) summarised for a record: dict(counted=True, max, total, carriers per wait), or
    dict(counted=False, why) when the circuit is above the bounds (MAX_QUBITS active qubits, MAX_WORK). Never raises
    for size: a build that records the footprint goes on without it."""
    try:
        f = wait_footprint(qc, span=span)
    except ValueError as e:
        return dict(counted=False, why=str(e))
    return dict(counted=True, max=f["max"], total=f["total"], excited=round(f["excited_waits"], 6),
                carriers=[w["carriers"] for w in f["waits"]])


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
