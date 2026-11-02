"""Local stand-ins for the device (L0): Aer runs of an assembled job, written in the device's result format.

- `noiseless`: the ideal circuit. Clifford circuits (sh6, qca, references) run on Aer's stabilizer method, so the
  full 156-qubit layout is no problem; others run on the statevector of the active qubits only.
- `pauli`: Clifford circuits on the stabilizer method with a ZZ Pauli error of probability p after every CZ on chosen
  physical couplers (stochastic, per shot). Used with the Clifford-point echo, so the full 72-qubit rings run.
- `planted`: the ideal circuit plus a coherent ZZ rotation RZZ(theta) after every CZ on chosen physical couplers.
  It stays unitary, so one statevector per circuit is enough. The planted couplers are given by the caller (a test);
  nothing in psf-os's decision path sees them.

These are simulations, not device results; submission.json says which one ran.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np


def _stamp():
    from zoneinfo import ZoneInfo
    now = dt.datetime.now(dt.timezone.utc)
    loc = now.astimezone(ZoneInfo("Europe/Zurich"))
    return f"{now.isoformat(timespec='seconds')} ({loc.strftime('%Y-%m-%d %H:%M:%S')} CET)"


MAX_STATEVECTOR = 28


def compact(circ):
    """The circuit on its active qubits only (same clbits), and the active physical qubits in order."""
    from qiskit import QuantumCircuit
    act = sorted({circ.find_bit(q).index for i in circ.data for q in i.qubits
                  if i.operation.name not in ("barrier", "delay")})
    pos = {q: k for k, q in enumerate(act)}
    out = QuantumCircuit(len(act), circ.num_clbits)
    for i in circ.data:
        if i.operation.name in ("barrier", "delay"):
            continue
        out.append(i.operation, [pos[circ.find_bit(q).index] for q in i.qubits],
                   [circ.find_bit(c).index for c in i.clbits])
    return out, act


def to_clifford_gates(circ):
    """rz(k pi/2) -> z / s / sdg (or nothing); None if any rz angle is not a multiple of pi/2."""
    from qiskit import QuantumCircuit
    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    for i in circ.data:
        nm = i.operation.name
        qs = [circ.find_bit(q).index for q in i.qubits]
        cs = [circ.find_bit(c).index for c in i.clbits]
        if nm == "rz":
            k = float(i.operation.params[0]) / (np.pi / 2)
            if abs(k - round(k)) > 1e-9:
                return None
            k = int(round(k)) % 4
            if k == 1:
                out.s(qs[0])
            elif k == 2:
                out.z(qs[0])
            elif k == 3:
                out.sdg(qs[0])
        else:
            out.append(i.operation, qs, cs)
    return out


def with_planted_zz(circ, couplers, theta):
    """The circuit with RZZ(theta) after every CZ on one of the given physical couplers."""
    from qiskit import QuantumCircuit
    bad = {tuple(sorted(e)) for e in couplers}
    out = QuantumCircuit(*circ.qregs, *circ.cregs)
    for i in circ.data:
        out.append(i.operation, [out.qubits[circ.find_bit(q).index] for q in i.qubits],
                   [out.clbits[circ.find_bit(c).index] for c in i.clbits])
        if i.operation.name == "cz":
            a, b = (circ.find_bit(q).index for q in i.qubits)
            if tuple(sorted((a, b))) in bad:
                out.rzz(theta, out.qubits[a], out.qubits[b])
    return out


def pauli_noise(couplers, p, kind="zz"):
    """kind "zz": a ZZ error of probability p after each CZ on the couplers; "depol2": each of the 15 non-identity
    two-qubit Paulis with probability p / 15 (a ZZ error keeps every even-weight Z parity, so it is invisible to a
    parity read in X; the two-qubit depolarizing error is not)."""
    from qiskit_aer.noise import NoiseModel, pauli_error
    nm = NoiseModel(basis_gates=["cz", "sx", "x", "rz", "s", "sdg", "z", "h", "measure"])
    if kind == "zz":
        err = pauli_error([("ZZ", p), ("II", 1 - p)])
    elif kind == "depol2":
        labels = [a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"]
        err = pauli_error([(lab, p / 15) for lab in labels] + [("II", 1 - p)])
    else:
        raise ValueError(kind)
    for a, b in couplers:
        nm.add_quantum_error(err, ["cz"], [a, b])
        nm.add_quantum_error(err, ["cz"], [b, a])
    return nm


def run_circuits(circs, shots, seed=0, planted=None, theta=0.0, pauli=None):
    from qiskit_aer import AerSimulator
    rows = []
    if pauli is not None:
        nm = pauli_noise(*pauli)
        for k, c in enumerate(circs):
            cc = to_clifford_gates(c)
            if cc is None:
                raise ValueError("the pauli mode needs Clifford circuits")
            res = AerSimulator(method="stabilizer", noise_model=nm).run(cc, shots=shots, seed_simulator=seed + k).result()
            rows.append(dict(pub=k, counts=dict(res.get_counts(0))))
        return rows
    for k, c in enumerate(circs):
        if planted:
            c = with_planted_zz(c, planted, theta)
        cc = to_clifford_gates(c)
        if cc is not None:
            sim, run_c = AerSimulator(method="stabilizer"), cc
        else:
            run_c, act = compact(c)
            if len(act) > MAX_STATEVECTOR:
                raise ValueError(f"circuit {k} is not Clifford and has {len(act)} active qubits (over "
                                 f"{MAX_STATEVECTOR}): it cannot be run locally. Scan n = 20 only (--scan-sizes 20), "
                                 "or use the Clifford-point scan for dry runs (--scan-clifford)")
            sim = AerSimulator(method="statevector")
        res = sim.run(run_c, shots=shots, seed_simulator=seed + k).result()
        rows.append(dict(pub=k, counts=dict(res.get_counts(0))))
    return rows


def submit(job_dir, mode="noiseless", seed=20261102, planted=None, theta=0.0, p=0.0, kind="zz"):
    """Run the job's circuits locally and write raw_counts.json and submission.json (the device's format)."""
    from qiskit import qpy
    sub = os.path.join(job_dir, "submission.json")
    if os.path.exists(sub):
        raise FileExistsError(f"{sub} exists: this job was already run")
    man = json.load(open(os.path.join(job_dir, "manifest.json")))
    circs = qpy.load(open(os.path.join(job_dir, "circuits.qpy"), "rb"))
    rows = run_circuits(circs, man["shots"], seed, planted if mode == "planted" else None, theta,
                        pauli=(planted, p, kind) if mode == "pauli" else None)
    t = _stamp()
    json.dump(dict(job_id=f"local-{mode}-{seed}", submitted=t, local=True, backend=man["fetch"]["backend"],
                   simulator=f"aer {mode}"), open(sub, "w"), indent=1)
    json.dump(dict(fetched=t, job_id=f"local-{mode}-{seed}", status=f"DONE (local {mode})", rows=rows),
              open(os.path.join(job_dir, "raw_counts.json"), "w"))
    return f"local-{mode}-{seed}"
