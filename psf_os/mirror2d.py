"""A sentinel made of the user's own 2D circuit: U, a barrier, then U^dagger (gate list reversed, angles negated), on
the same qubits and the same edge order, in the same job. Ideal output |0...0>, so <Z_q> = 1 for every qubit and no
classical answer is needed. Per qubit, the drop of this mirror signal below the readout floor (the theta_x = 0 control
of the same job) marks where the circuit loses its signal.

Pitfall: the level-1 pass manager cancels mutually inverse gates that meet at the join of U and U^dagger. A barrier
sits at the join, and acceptance requires exactly twice U's CZ count.

Limit, stated first: a mirror cancels some errors, coherent ones in particular (a static ZZ, as the frame flip showed
on mirror circuits). A healthy mirror does not show that the problem's answer is right; a failing mirror shows that
something on those qubits loses the signal. The flags are a check, not a correction.

Flag rule (the same form as rule R): for every qubit, d_q = z0_q - zm_q (floor minus mirror signal, each the mean of
its cells); flag q if d_q > median(d) + max(5 x 1.4826 x MAD(d), 0.05). An edge is flagged when both its ends are.
"""
from __future__ import annotations

import json
import os

import numpy as np

from psf_os import ising2d as I


def mirror(logical, measure_qubits=None):
    """U, barrier, U^dagger, then Z measurement of `measure_qubits` (physical; clbit k = k-th of them)."""
    from qiskit import QuantumCircuit
    u = logical.remove_final_measurements(inplace=False) if logical.num_clbits else logical
    qs = list(measure_qubits) if measure_qubits is not None else sorted(
        {u.find_bit(q).index for ins in u.data for q in ins.qubits})
    out = QuantumCircuit(u.num_qubits, len(qs))
    out.compose(u, inplace=True)
    out.barrier()
    out.compose(u.inverse(), inplace=True)
    for k, q in enumerate(qs):
        out.measure(q, k)
    return out


def cz_count(c):
    return sum(1 for i in c.data if i.operation.name == "cz")


def build(dev, job, runs, compiler=None, name=None):
    """One record of mirror cells of the P6 circuit: job = {"ts": [...], "theta_x": 0.53, "shots": 4000}, placed by
    ising2d.place. Acceptance: every compiled mirror has exactly 2 x 2 x 38 x t CZs and measures the patch."""
    from qiskit import qpy
    from psf_os.compile import VERSION, RuleCompiler, measured_qubits
    from psf_os.registry import sha256_file
    rc = compiler or RuleCompiler(dev)
    dr, dc, m, reasons = I.place(dev)
    ts = [int(t) for t in job["ts"]]
    tx = float(job.get("theta_x", I.THETA_X))
    phys = [m[q] for q in I.QUBITS]
    cs, points, q2 = [], [], []
    for t in ts:
        u = I.circuit(t, mapping=m, width=dev.num_qubits, measure=False, tx=tx)
        c = rc.pm.run(mirror(u, phys))
        cs.append(c)
        points.append(dict(input=f"m{t}", setting="z", t=t))
        q2.append(cz_count(c))
    want = {k: p for k, p in enumerate(phys)}
    expected = [2 * 2 * len(I.EDGES) * t for t in ts]
    accepted = all(measured_qubits(c) == want for c in cs) and q2 == expected
    name = name or f"pmir2d_tx{tx:g}_qedma-patch"
    for k, c in enumerate(cs):
        c.name = f"{name}-{k}"
    os.makedirs(runs, exist_ok=True)
    jf, qf = os.path.join(runs, name + ".json"), os.path.join(runs, name + ".qpy")
    for f in (jf, qf):
        if os.path.exists(f):
            raise FileExistsError(f"{f} exists: a run never overwrites another")
    with open(qf, "wb") as f:
        qpy.dump(cs, f)
    rec = dict(kind="psfos", psfos_role="mirror", psfos_job=dict(job, kind="mirror2d"), arm="patch:copy",
               version=VERSION, device=dev.name, masked=dev.masked, task=dict(n=len(I.QUBITS), layers=2 * max(ts)),
               copy=dict(drow=dr, dcol=dc, passed_over=reasons), mapping={str(q): m[q] for q in I.QUBITS},
               echo=dict(kind="mirror2d", q2_circuits=q2, q2_expected=expected, accepted=bool(accepted)),
               points=points, valid=bool(accepted), shots=job.get("shots", 4000),
               export=dict(qpy=os.path.basename(qf), circuits=len(cs), sha256=sha256_file(qf)))
    json.dump(rec, open(jf, "w"), indent=1)
    return rec


def score(job_dir, runs):
    """{t: [<Z> per label in sorted order]} of the job's mirror cells."""
    from psf_os import job as J
    from psf_os.kim import z_values
    out = {}
    for name, (rec, cc) in J.read(job_dir, runs).items():
        if rec.get("psfos_job", {}).get("kind") == "mirror2d":
            for p in rec["points"]:
                out[p["t"]] = z_values(cc[(p["input"], "z")], len(I.QUBITS))[0].tolist()
    if not out:
        raise RuntimeError("no mirror2d record")
    return out


def flags(z_floor, z_mirror, labels=None, edges=None, k=5.0, margin=0.05):
    """The flag rule. z_floor, z_mirror: per qubit (same order as `labels`). Returns a dict with d, the line and the
    flagged qubits and edges."""
    labels = list(labels if labels is not None else I.QUBITS)
    edges = edges if edges is not None else I.EDGES
    d = np.asarray(z_floor, dtype=float) - np.asarray(z_mirror, dtype=float)
    med = float(np.median(d))
    sig = 1.4826 * float(np.median(np.abs(d - med)))
    line = med + max(k * sig, margin)
    bad = [labels[i] for i in range(len(labels)) if d[i] > line]
    bset = set(bad)
    return dict(d=[float(x) for x in d], median=med, sigma=sig, line=line, qubits=bad,
                edges=[list(e) for e in edges if e[0] in bset and e[1] in bset])
