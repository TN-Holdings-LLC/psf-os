"""Practice problem P4: the dual-unitary kicked Ising chain with an exact answer at every depth (reproduction of
IBM's experiment, arXiv 2411.00765, SI Eq. S5 and Fig. 2).

The circuit (open chain of N sites, N odd):
  - initial state |+> on site 0 and Bell pairs (|00> + |11>)/sqrt(2) on (1, 2), (3, 4), ..., (N - 2, N - 1);
  - t brickwork layers, the first on the even pairs (0, 1), (2, 3), ..., then odd pairs (1, 2), ..., alternating;
    each gate is U = e^{-ihZ_a} e^{-iJ Z_a Z_b} e^{-ib(X_a + X_b)} e^{-iJ Z_a Z_b} e^{-ihZ_a} (a the left site);
  - every site measured in the X basis.
At the dual-unitary point J = b = pi/4 and for t <= (N - 1)/2: <X_n> = cos(2h)^t if n = t, else 0. The answer needs
no simulation; `check_statevector` confirms it on small chains (and the order of the layers: odd pairs first gives a
different answer).

On the device the gate is written with CZ: e^{-i pi/4 ZZ} = CZ (RZ(pi/2) x RZ(pi/2)) up to a global phase, so every
gate is exactly 2 CZ and every Bell pair 1 CZ. J must be pi/4; b may differ from pi/4 (then the answer is not known in
closed form: the second stage, not built here).

Placement: the chain is ring positions 0..N-1 of the user's n-site ring as psf-os places it (the rule, with the
registry's mask or without); the ring is a cycle of the device, so consecutive positions are coupled (checked).
The compiled circuit uses the same map-free pass manager as the rule compiler (layout fixed, no routing).
"""
from __future__ import annotations

import json
import os

import numpy as np

from psf_os.compile import VERSION, RuleCompiler, measured_qubits
from psf_os.registry import sha256_file

PI4 = np.pi / 4


def exact(N, t, h):
    """{site: <X_site>} at the dual-unitary point (closed form)."""
    if N % 2 == 0 or t > (N - 1) // 2:
        raise ValueError("N odd and t <= (N - 1)/2")
    return [float(np.cos(2 * h) ** t) if n == t else 0.0 for n in range(N)]


def _zz(qc, a, b):
    """e^{-i pi/4 Z_a Z_b} with one CZ (global phase dropped)."""
    qc.cz(a, b)
    qc.rz(np.pi / 2, a)
    qc.rz(np.pi / 2, b)


def gate(qc, a, b, h, kick=PI4):
    """SI Eq. S5 at J = pi/4, right to left: RZ(2h) on a, ZZ, RX(2 kick) on both, ZZ, RZ(2h) on a."""
    qc.rz(2 * h, a)
    _zz(qc, a, b)
    qc.rx(2 * kick, a)
    qc.rx(2 * kick, b)
    _zz(qc, a, b)
    qc.rz(2 * h, a)


def chain_circuit(N, t, h, kick=PI4, measure=True):
    """The logical chain circuit (N qubits, qubit k = site k), X-basis readout of every site."""
    from qiskit import QuantumCircuit
    if N % 2 == 0:
        raise ValueError("N must be odd")
    qc = QuantumCircuit(N, N if measure else 0)
    qc.h(0)
    for i in range(1, N - 1, 2):
        qc.h(i)
        qc.h(i + 1)
        qc.cz(i, i + 1)
        qc.h(i + 1)
    for k in range(t):
        for i in range(k % 2, N - 1, 2):
            gate(qc, i, i + 1, h, kick)
    if measure:
        for q in range(N):
            qc.h(q)
        for q in range(N):
            qc.measure(q, q)
    return qc


def cz_count(N, t):
    """CZs of one chain circuit: (N - 1)/2 Bell pairs + 2 per gate."""
    return (N - 1) // 2 + 2 * sum(len(range(k % 2, N - 1, 2)) for k in range(t))


def check_statevector(N, t, h, kick=PI4):
    """<X_n> for every site from the statevector (small N)."""
    from qiskit.quantum_info import Pauli, Statevector
    sv = Statevector(chain_circuit(N, t, h, kick, measure=False))
    return [float(sv.expectation_value(Pauli("I" * (N - 1 - n) + "X" + "I" * n)).real) for n in range(N)]


def chain_from_ring(dev, ring, N):
    """Ring positions 0..N-1 as a chain; refuses if two consecutive positions are not a working coupler."""
    edges = {tuple(sorted(e)) for e in dev.working_couplers()}
    chain = [int(q) for q in ring[:N]]
    for a, b in zip(chain, chain[1:]):
        if tuple(sorted((a, b))) not in edges:
            raise RuntimeError(f"({a}, {b}) is not a working coupler: the ring positions are not a chain")
    return chain


def compile_on(rc, qc, chain):
    """Place logical qubit k on physical chain[k] and run the rule compiler's map-free pass manager."""
    from qiskit import QuantumCircuit
    nq = rc.dev.num_qubits
    full = QuantumCircuit(nq, qc.num_clbits)
    full.compose(qc, qubits=list(chain), clbits=list(range(qc.num_clbits)), inplace=True)
    return rc.pm.run(full)


def cell_name(N, h, kick=PI4):
    return f"pdu_N{N}_h{h:g}_b{kick:.6f}-chain"


def build(dev, job, runs, ring, compiler=None, name=None):
    """One record per h: the points are t = 0..t_max. job = {"N": 21, "h": 0.1, "t_max": 10, "shots": 4000}.
    Acceptance: each compiled circuit has exactly cz_count(N, t) CZs and measures site k into clbit k."""
    from qiskit import qpy
    from psf_os import circuits as C
    rc = compiler or RuleCompiler(dev)
    N, h, tmax = job["N"], job["h"], job["t_max"]
    kick = job.get("kick", PI4)
    if tmax > (N - 1) // 2:
        raise ValueError("t_max <= (N - 1)/2")
    chain = chain_from_ring(dev, ring, N)
    cs, points, q2 = [], [], []
    for t in range(tmax + 1):
        c = compile_on(rc, chain_circuit(N, t, h, kick), chain)
        cs.append(c)
        points.append(dict(input=f"t{t}", setting="x", t=t))
        q2.append(C.q2_count(c))
    mq = [measured_qubits(c) for c in cs]
    ok_meas = all(m == {k: chain[k] for k in range(N)} for m in mq)
    accepted = ok_meas and all(q == cz_count(N, t) for t, q in enumerate(q2))
    name = name or cell_name(N, h, kick)
    for k, c in enumerate(cs):
        c.name = f"{name}-{k}"
    os.makedirs(runs, exist_ok=True)
    jf, qf = os.path.join(runs, name + ".json"), os.path.join(runs, name + ".qpy")
    for f in (jf, qf):
        if os.path.exists(f):
            raise FileExistsError(f"{f} exists: a run never overwrites another")
    with open(qf, "wb") as f:
        qpy.dump(cs, f)
    answers = {p["input"]: exact(N, p["t"], h) for p in points} if abs(kick - PI4) < 1e-12 else None
    rec = dict(kind="psfos", psfos_role="user", psfos_job=dict(job, kind="du_kising"), arm="chain:rule-ring",
               version=VERSION, device=dev.name, masked=dev.masked, task=dict(n=N, layers=tmax),
               chain=chain, ring=[int(q) for q in ring],
               echo=dict(kind="du_kising", measured_qubits={str(k): chain[k] for k in range(N)}, q2_circuits=q2,
                         q2_expected=[cz_count(N, t) for t in range(tmax + 1)], accepted=bool(accepted)),
               points=points, answers=answers, valid=bool(accepted), shots=job.get("shots", 4000),
               export=dict(qpy=os.path.basename(qf), circuits=len(cs), sha256=sha256_file(qf)))
    json.dump(rec, open(jf, "w"), indent=1)
    return rec


# ------------------------------------------------------------------------------------------------ after the run

def x_values(counts, N):
    """<X_n> per site from X-basis counts (clbit n = site n), and the shot-noise SE."""
    tot = float(sum(counts.values()))
    m = np.zeros(N)
    for key, k in counts.items():
        b = key.replace(" ", "")
        m += k * np.array([1 - 2 * int(b[-1 - n]) for n in range(N)])
    m /= tot
    return m, np.sqrt(np.maximum(1 - m ** 2, 0) / tot)


def score(job_dir, runs):
    """Per h: measured <X_t> on the light cone (site t) and off it, with the exact values. Sentinel cells skipped."""
    from psf_os import job as J
    out = {}
    for name, (rec, cc) in J.read(job_dir, runs).items():
        if rec.get("psfos_job", {}).get("kind") != "du_kising":
            continue
        N, h = rec["task"]["n"], rec["psfos_job"]["h"]
        rows = []
        for p in rec["points"]:
            m, se = x_values(cc[(p["input"], "x")], N)
            t = p["t"]
            off = [abs(m[n]) for n in range(N) if n != t]
            row = dict(t=t, on=float(m[t]), on_se=float(se[t]), off_mean_abs=float(np.mean(off)),
                       off_max_abs=float(np.max(off)), all=[float(x) for x in m])
            if rec.get("answers"):
                row["exact"] = rec["answers"][p["input"]][t]
            rows.append(row)
        out[name] = dict(N=N, h=h, chain=rec["chain"], rows=rows)
    return out
