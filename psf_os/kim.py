"""Practice problem P5: the kicked Ising model on the device's own lattice (the form of IBM's 2023 "utility"
experiment, Kim et al., Nature 618, 500), without error mitigation.

The lattice: every working coupler of the device record between working qubits (failed elements and the registry's
masked couplers removed), its largest connected part (`region`). Nothing but the coupling graph and the failed marks
is read.

The circuit (|0...0> in, `t` steps): each step is RZZ(-pi/2) on every lattice edge, then RX(theta) on every lattice
qubit; Z readout of every lattice qubit. On the device RZZ(-pi/2) is one CZ: e^{+i pi/4 ZZ} = CZ (RZ(-pi/2) x
RZ(-pi/2)) up to a global phase. The edges are scheduled in colour classes (greedy edge colouring; the RZZs commute, so
the schedule changes the depth, not the answer).

Answers, per qubit q: theta = 0: <Z_q> = 1. theta = pi/2: the circuit is Clifford (stabilizer simulation of the whole
lattice). Any theta: the causal cone. The first step's RZZs act on |0...0> and give a phase only; after them each RZZ
layer widens the support of the backward-evolved Z_q by one edge. So <Z_q> is the same on the lattice and on the ball
of radius t - 1 around q (its qubits, the edges inside it, the same gate list), checked against the whole lattice in
tests/test_kim.py.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

from psf_os.compile import VERSION, RuleCompiler, measured_qubits
from psf_os.registry import sha256_file

THETA_J = -np.pi / 2


def region(dev, suspect=False):
    """(sorted qubits, sorted edges) of the largest connected part of the working lattice.
    suspect=True also leaves out both ends of every failed or masked coupler (a qubit next to a failed coupler is not
    trusted: on 3 November the two worst qubits of the lattice were the ends of IBM's failed coupler (130, 131))."""
    drop = {q for c in dev.failed_couplers for q in c} if suspect else set()
    edges = sorted(tuple(sorted(e)) for e in dev.working_couplers()
                   if dev.qubit_ok(e[0]) and dev.qubit_ok(e[1]) and e[0] not in drop and e[1] not in drop)
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, best = set(), []
    for s in sorted(adj):
        if s in seen:
            continue
        comp, stack = {s}, [s]
        while stack:
            x = stack.pop()
            for y in adj[x] - comp:
                comp.add(y)
                stack.append(y)
        seen |= comp
        if len(comp) > len(best):
            best = sorted(comp)
    keep = set(best)
    return best, [e for e in edges if e[0] in keep and e[1] in keep]


def colour(edges):
    """Greedy edge colouring: a list of edge lists, no qubit twice in one list."""
    layers = []
    for e in sorted(edges):
        for L in layers:
            if all(e[0] not in f and e[1] not in f for f in L):
                L.append(e)
                break
        else:
            layers.append([e])
    return layers


def ops(qubits, edges, theta, t):
    """The gate list: ("cz", a, b), ("rz", angle, q), ("rx", angle, q). Edges in colour order every step."""
    out = []
    layers = colour(edges)
    for _ in range(t):
        for L in layers:
            for a, b in L:
                out.append(("cz", a, b))
                out.append(("rz", THETA_J, a))
                out.append(("rz", THETA_J, b))
        for q in qubits:
            out.append(("rx", float(theta), q))
    return out


def circuit(qubits, edges, theta, t, width=None, measure=True):
    """A circuit on `width` qubits (default: max qubit + 1) with qubit q = physical q; clbit k = qubits[k]."""
    from qiskit import QuantumCircuit
    width = width or (max(qubits) + 1)
    qc = QuantumCircuit(width, len(qubits) if measure else 0)
    for op in ops(qubits, edges, theta, t):
        if op[0] == "cz":
            qc.cz(op[1], op[2])
        else:
            getattr(qc, op[0])(op[1], op[2])
    if measure:
        for k, q in enumerate(qubits):
            qc.measure(q, k)
    return qc


def ball(edges, q, r):
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, fr = {q}, {q}
    for _ in range(r):
        fr = {y for x in fr for y in adj[x]} - seen
        seen |= fr
    return sorted(seen)


def cone(qubits, edges, q, t):
    """(cone qubits, cone edges): the ball of radius t - 1 around q and the lattice edges inside it."""
    b = ball(edges, q, max(t - 1, 0))
    s = set(b)
    return b, [e for e in edges if e[0] in s and e[1] in s]


def answer_cone(qubits, edges, q, theta, t, max_qubits=26, beyond=None):
    """<Z_q> from q's causal cone by statevector (relabelled to 0..m-1). beyond="tn" (2026-11-11): a cone over
    max_qubits goes to psf_os.exact_tn (light-cone tensor network, memory guard) instead of being refused."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator
    cq, ce = cone(qubits, edges, q, t)
    if len(cq) > max_qubits and beyond != "tn":
        raise ValueError(f"the cone of {q} at t = {t} has {len(cq)} qubits, over {max_qubits}")
    idx = {p: k for k, p in enumerate(cq)}
    qc = QuantumCircuit(len(cq))
    for op in ops(cq, ce, theta, t):
        if op[0] == "cz":
            qc.cz(idx[op[1]], idx[op[2]])
        else:
            getattr(qc, op[0])(op[1], idx[op[2]])
    if len(cq) > max_qubits:
        from psf_os import exact_tn
        return exact_tn.expectation(qc, {idx[q]: "Z"})[0], len(cq)
    qc.save_expectation_value(Pauli("Z"), [idx[q]], label="z")
    sim = AerSimulator(method="statevector", precision="double")
    return float(sim.run(qc).result().data(0)["z"]), len(cq)


def answers(qubits, edges, theta, t, max_qubits=26, beyond=None):
    """{"z": [<Z_q> for q in qubits], "method": ..., "cone_max": ...}. beyond="tn": cones over max_qubits by
    psf_os.exact_tn (method then reads "cone statevector / tn")."""
    if abs(theta) < 1e-12 or t == 0:
        return dict(z=[1.0] * len(qubits), method="trivial", cone_max=0)
    if abs(abs(theta) - np.pi / 2) < 1e-12:
        from qiskit.quantum_info import Pauli, StabilizerState
        st = StabilizerState(circuit(qubits, edges, theta, t, width=max(qubits) + 1, measure=False))
        n = max(qubits) + 1
        return dict(z=[float(st.expectation_value(Pauli("I" * (n - 1 - q) + "Z" + "I" * q))) for q in qubits],
                    method="stabilizer", cone_max=0)
    z, m = [], 0
    for q in qubits:
        v, k = answer_cone(qubits, edges, q, theta, t, max_qubits, beyond)
        z.append(v)
        m = max(m, k)
    return dict(z=z, method="cone statevector" if m <= max_qubits else "cone statevector / tn", cone_max=m)


def cell_name(theta, tag=""):
    return f"pkim{tag}_th{theta:.6f}-lattice"


def build(dev, job, runs, compiler=None, name=None):
    """One record per theta, points t = 0..t_max (job = {"theta": .., "t_max": .., "shots": ..}). The circuit is placed
    as written (qubit q = physical q) and run through the rule compiler's map-free pass manager (no layout, no routing).
    Acceptance: each compiled circuit has exactly t x (lattice edges) CZs and measures qubits[k] into clbit k."""
    from qiskit import qpy
    from psf_os import circuits as C
    rc = compiler or RuleCompiler(dev)
    qubits, edges = region(dev, suspect=bool(job.get("suspect")))
    theta = float(job["theta"])
    ts = [int(t) for t in job["ts"]] if "ts" in job else list(range(int(job["t_max"]) + 1))
    tmax = max(ts)
    cs, points, q2 = [], [], []
    for t in ts:
        c = rc.pm.run(circuit(qubits, edges, theta, t, width=dev.num_qubits))
        cs.append(c)
        points.append(dict(input=f"t{t}", setting="z", t=t))
        q2.append(C.q2_count(c))
    ok_meas = all(measured_qubits(c) == {k: q for k, q in enumerate(qubits)} for c in cs)
    accepted = ok_meas and all(q == t * len(edges) for t, q in zip(ts, q2))
    name = name or cell_name(theta, job.get("tag", ""))
    for k, c in enumerate(cs):
        c.name = f"{name}-{k}"
    os.makedirs(runs, exist_ok=True)
    jf, qf = os.path.join(runs, name + ".json"), os.path.join(runs, name + ".qpy")
    for f in (jf, qf):
        if os.path.exists(f):
            raise FileExistsError(f"{f} exists: a run never overwrites another")
    with open(qf, "wb") as f:
        qpy.dump(cs, f)
    rec = dict(kind="psfos", psfos_role="user", psfos_job=dict(job, kind="kim_lattice"), arm="lattice:region",
               version=VERSION, device=dev.name, masked=dev.masked, task=dict(n=len(qubits), layers=tmax),
               qubits=qubits, edges=[list(e) for e in edges], colours=len(colour(edges)),
               echo=dict(kind="kim_lattice", q2_circuits=q2, q2_expected=[t * len(edges) for t in ts],
                         accepted=bool(accepted)),
               points=points, valid=bool(accepted), shots=job.get("shots", 4000),
               export=dict(qpy=os.path.basename(qf), circuits=len(cs), sha256=sha256_file(qf)))
    json.dump(rec, open(jf, "w"), indent=1)
    return rec


def z_values(counts, n):
    """<Z_k> for clbit k (k = position in the record's qubit list), and the shot-noise SE."""
    tot = float(sum(counts.values()))
    m = np.zeros(n)
    for key, c in counts.items():
        b = key.replace(" ", "")
        m += c * (1 - 2 * np.array([int(b[-1 - k]) for k in range(n)]))
    m /= tot
    return m, np.sqrt(np.maximum(1 - m ** 2, 0) / tot)


def score(job_dir, runs):
    """Per theta: measured <Z_q> per t (sentinel cells skipped)."""
    from psf_os import job as J
    out = {}
    for name, (rec, cc) in J.read(job_dir, runs).items():
        if rec.get("psfos_job", {}).get("kind") != "kim_lattice":
            continue
        n = len(rec["qubits"])
        rows = []
        for p in rec["points"]:
            m, se = z_values(cc[(p["input"], "z")], n)
            rows.append(dict(t=p["t"], z=[float(x) for x in m], se=[float(x) for x in se]))
        out[name] = dict(theta=rec["psfos_job"]["theta"], tag=rec["psfos_job"].get("tag", ""),
                         suspect=bool(rec["psfos_job"].get("suspect")), qubits=rec["qubits"], rows=rows)
    return out
