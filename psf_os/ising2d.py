"""Practice problem P6: the 2D kicked Ising problem of IBM's public tutorial "Ising model with Qedma QESEM" (35 qubits of
a Heron r2 heavy-hex lattice, theta_x = 0.53, theta_z = 0.1, theta_zz = 1.0, steps 5, 7, 9; average magnetisation),
run by psf-os without error mitigation and scored against exact answers.

The problem is fixed on the tutorial's qubit labels (`QUBITS`, `LAYERS`: the three edge layers as the tutorial lists
them). Each Trotter step, as in the tutorial's code: RX(theta_x) then RZ(theta_z) on every qubit (sorted), then
RZZ(theta_zz) on every edge, layer by layer. |0...0> in; Z readout of every qubit.

On the device RZZ(theta) is written with two CZ: CX RZ_b(theta) CX, CX = H_b CZ H_b (`ops`), so every circuit has
exactly 2 x 38 x t CZs.

Placement (map-free): a Heron r2 lattice repeats every two rows (+40 in the index) and every four columns (+4 for a row
qubit, +1 for a bridge qubit), so the tutorial's patch has six exact copies (row offset 0, 2, 4 x column offset 0, 4).
`place` takes the first copy, in that fixed order, whose qubits are working, whose edges are working couplers, and that
has no qubit at an end of a failed or masked coupler (the suspect-qubit rule of P5). No search.

Answers: <Z_q> depends only on the ball of radius t - 1 around q (the last RZZ layer and RZ commute with Z_q; every
earlier RZZ layer widens the support by one edge), checked against the whole patch in tests/test_ising2d.py. They are
the same for every copy, so they are computed once, on the tutorial's labels.
"""
from __future__ import annotations

import json
import os

import numpy as np

THETA_X, THETA_Z, THETA_ZZ = 0.53, 0.1, 1.0
LAYERS = [
    [(6, 7), (10, 11), (16, 23), (24, 25), (17, 27), (28, 29), (18, 31), (36, 41), (42, 43), (37, 45), (46, 47), (38, 49)],
    [(3, 4), (5, 6), (7, 8), (9, 10), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (43, 44), (45, 46), (47, 48)],
    [(3, 16), (7, 17), (11, 18), (22, 23), (26, 27), (30, 31), (21, 36), (25, 37), (29, 38), (41, 42), (44, 45),
     (48, 49), (4, 5), (8, 9)],
]
EDGES = sorted(tuple(sorted(e)) for L in LAYERS for e in L)
QUBITS = sorted({q for e in EDGES for q in e})
BRIDGES = {16, 17, 18, 19}  # within one row pair; the patch's bridges are 16-18 and 36-38


def shift(q, drow, dcol):
    """The tutorial label q moved by drow rows (even) and dcol columns (a multiple of 4) on a Heron r2 lattice."""
    r, c = divmod(q, 20)
    if c >= 16:  # a bridge qubit: index 20 r + 16 + k joins the rows at column 3 + 4k (even r) or 1 + 4k (odd r)
        return 20 * (r + drow) + c + dcol // 4
    return 20 * (r + drow) + c + dcol


def copies():
    """The six copies, in the fixed order: [(drow, dcol, {label: physical})]."""
    return [(dr, dc, {q: shift(q, dr, dc) for q in QUBITS}) for dr in (0, 2, 4) for dc in (0, 4)]


def place(dev):
    """(drow, dcol, mapping, reasons): the first copy with no failed, masked or suspect element; reasons lists why each
    earlier copy was passed over. Raises if none qualifies."""
    working = {tuple(sorted(e)) for e in dev.working_couplers()}
    suspect = {q for c in dev.failed_couplers for q in c}
    reasons = []
    for dr, dc, m in copies():
        why = []
        for q in QUBITS:
            p = m[q]
            if p >= dev.num_qubits or not dev.qubit_ok(p):
                why.append(f"qubit {p} failed")
            elif p in suspect:
                why.append(f"qubit {p} suspect")
        for a, b in EDGES:
            if tuple(sorted((m[a], m[b]))) not in working:
                why.append(f"coupler ({m[a]}, {m[b]}) not working")
        if not why:
            return dr, dc, m, reasons
        reasons.append(dict(drow=dr, dcol=dc, why=why))
    raise RuntimeError(f"no copy of the patch qualifies: {reasons}")


def ops(qubits, layers, t, tx=THETA_X, tz=THETA_Z, tzz=THETA_ZZ):
    """The gate list in time order: ("h"|"rx"|"rz", angle or None, q) and ("cz", a, b)."""
    qs = sorted(qubits)
    out = []
    for _ in range(t):
        for q in qs:
            out.append(("rx", tx, q))
            out.append(("rz", tz, q))
        for L in layers:
            for a, b in L:
                out += [("h", None, b), ("cz", a, b), ("h", None, b), ("rz", tzz, b),
                        ("h", None, b), ("cz", a, b), ("h", None, b)]
    return out


def circuit(t, mapping=None, width=None, measure=True, qubits=None, layers=None, tx=THETA_X):
    """The circuit on labels -> physical qubits `mapping` (default: the tutorial's labels); clbit k = sorted labels[k].
    `tx` is theta_x (the tutorial's 0.53 by default; the problem bank also uses other values)."""
    from qiskit import QuantumCircuit
    qubits = qubits or QUBITS
    layers = layers or LAYERS
    m = mapping or {q: q for q in qubits}
    width = width or (max(m.values()) + 1)
    qc = QuantumCircuit(width, len(qubits) if measure else 0)
    for op in ops(qubits, layers, t, tx=tx):
        if op[0] == "cz":
            qc.cz(m[op[1]], m[op[2]])
        elif op[0] == "h":
            qc.h(m[op[2]])
        else:
            getattr(qc, op[0])(op[1], m[op[2]])
    if measure:
        for k, q in enumerate(sorted(qubits)):
            qc.measure(m[q], k)
    return qc


def ball(q, r, edges=None):
    edges = edges or EDGES
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, fr = {q}, {q}
    for _ in range(r):
        fr = {y for x in fr for y in adj.get(x, ())} - seen
        seen |= fr
    return sorted(seen)


def cone(q, t, edges=None, layers=None):
    """(cone qubits, cone layers): the ball of radius t - 1 around q and the edges of each layer inside it."""
    layers = layers or LAYERS
    b = set(ball(q, max(t - 1, 0), edges))
    return sorted(b), [[e for e in L if e[0] in b and e[1] in b] for L in layers]


def answer_cone(q, t, max_qubits=26, edges=None, layers=None, tx=THETA_X):
    """<Z_q> from q's cone by Aer statevector (CPU)."""
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator
    cq, cl = cone(q, t, edges, layers)
    if len(cq) > max_qubits:
        return None, len(cq)
    idx = {p: k for k, p in enumerate(cq)}
    qc = circuit(t, mapping=idx, width=len(cq), measure=False, qubits=cq, layers=cl, tx=tx)
    qc.save_expectation_value(Pauli("Z"), [idx[q]], label="z")
    return float(AerSimulator(method="statevector").run(qc).result().data(0)["z"]), len(cq)


def build(dev, job, runs, compiler=None, name=None):
    """One record: t in job["ts"] on the copy chosen by `place`. Acceptance: 2 x 38 x t CZs, every qubit measured."""
    from qiskit import qpy
    from psf_os import circuits as C
    from psf_os.compile import VERSION, RuleCompiler, measured_qubits
    from psf_os.registry import sha256_file
    rc = compiler or RuleCompiler(dev)
    dr, dc, m, reasons = place(dev)
    ts = [int(t) for t in job["ts"]]
    tx = float(job.get("theta_x", THETA_X))
    cs, points, q2 = [], [], []
    for t in ts:
        c = rc.pm.run(circuit(t, mapping=m, width=dev.num_qubits, tx=tx))
        cs.append(c)
        points.append(dict(input=f"t{t}", setting="z", t=t))
        q2.append(C.q2_count(c))
    want = {k: m[q] for k, q in enumerate(QUBITS)}
    accepted = all(measured_qubits(c) == want for c in cs) and all(n == 2 * len(EDGES) * t for t, n in zip(ts, q2))
    name = name or f"pis2d_tx{tx:g}_qedma-patch"
    for k, c in enumerate(cs):
        c.name = f"{name}-{k}"
    os.makedirs(runs, exist_ok=True)
    jf, qf = os.path.join(runs, name + ".json"), os.path.join(runs, name + ".qpy")
    for f in (jf, qf):
        if os.path.exists(f):
            raise FileExistsError(f"{f} exists: a run never overwrites another")
    with open(qf, "wb") as f:
        qpy.dump(cs, f)
    rec = dict(kind="psfos", psfos_role="user", psfos_job=dict(job, kind="ising2d"), arm="patch:copy", version=VERSION,
               device=dev.name, masked=dev.masked, task=dict(n=len(QUBITS), layers=max(ts)),
               copy=dict(drow=dr, dcol=dc, passed_over=reasons), mapping={str(q): m[q] for q in QUBITS},
               echo=dict(kind="ising2d", q2_circuits=q2, q2_expected=[2 * len(EDGES) * t for t in ts],
                         accepted=bool(accepted)),
               points=points, valid=bool(accepted), shots=job.get("shots", 4000),
               export=dict(qpy=os.path.basename(qf), circuits=len(cs), sha256=sha256_file(qf)))
    json.dump(rec, open(jf, "w"), indent=1)
    return rec


def score(job_dir, runs):
    """{theta_x: {t: [<Z> per label in sorted order]}} of the job's ising2d records."""
    from psf_os import job as J
    from psf_os.kim import z_values
    out = {}
    for name, (rec, cc) in J.read(job_dir, runs).items():
        if rec.get("psfos_job", {}).get("kind") == "ising2d":
            tx = float(rec["psfos_job"].get("theta_x", THETA_X))
            out[f"{tx:g}"] = {p["t"]: z_values(cc[(p["input"], "z")], len(QUBITS))[0].tolist() for p in rec["points"]}
    if not out:
        raise RuntimeError("no ising2d record")
    return out
