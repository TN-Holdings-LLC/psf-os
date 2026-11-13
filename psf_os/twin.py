"""The Clifford twin of a compiled circuit: a ruler that needs no classical answer of the user's problem.

Twin = the compiled circuit with every RZ angle rounded to the nearest multiple of pi/2; CZ, SX, X, measurements,
barriers and their order are kept as they are. On IBM Heron, RZ is a virtual frame change, so the twin runs the same
physical pulses as the user's circuit, gate for gate (the user circuit's own non-virtual gates in the same places).
Every gate of the twin is Clifford, so its per-qubit <Z> is exact at any size by stabilizer simulation.

Use: run the twin beside the user circuit in the same job; the share of the twin's exact pattern kept by the device,
r_twin = sum z e / sum e^2 over the measured qubits, estimates the share r kept on the user circuit. If the device
error is mainly a uniform shrink, READS (MSE <= mean square / 2) is r >= 1 - 1/sqrt(2) ~ 0.293 whatever the signal's
size (TN Holdings, psf-os-lab-notebook, notes/paper9/r_thread_2026-11-13).

Limits, stated first: the twin sees the shrink, not the shape errors of the user circuit (records where the residual is
as large as the shrink, e.g. the 1D Hubbard model at U 4 with 6 steps, fail to read with r about 0.4). Whether r_twin
follows r on a device is a question for a declared test, not something this module shows. Coherent errors that depend
on the RZ angles (e.g. a static ZZ interacting with the frame) can differ between the twin and the user circuit.

The twin must be submitted as built: a later optimisation pass would merge the rounded RZ with neighbouring SX gates
and change the pulses (`check_same_schedule` guards this).
"""
from __future__ import annotations

import numpy as np

HALF_PI = np.pi / 2


def twin(compiled):
    """The Clifford twin of `compiled` (a circuit in the CZ / RZ / SX / X basis)."""
    from qiskit import QuantumCircuit
    out = QuantumCircuit(*compiled.qregs, *compiled.cregs)
    for ins in compiled.data:
        op = ins.operation
        if op.name == "rz":
            k = int(np.round(float(op.params[0]) / HALF_PI))
            q = QuantumCircuit(1)
            q.rz(k * HALF_PI, 0)
            out.append(q.data[0].operation, ins.qubits, ins.clbits)
        elif op.name in ("cz", "sx", "x", "measure", "barrier", "delay", "id"):
            out.append(op, ins.qubits, ins.clbits)
        else:
            raise ValueError(f"twin: gate {op.name!r} is not in the CZ / RZ / SX / X basis")
    return out


def schedule(c):
    """The circuit's non-RZ instructions in order: (name, qubit indices)."""
    return [(i.operation.name, tuple(c.find_bit(q).index for q in i.qubits)) for i in c.data
            if i.operation.name != "rz"]


def check_same_schedule(user, tw):
    """True when the twin has exactly the user circuit's non-RZ instructions in the same order."""
    return schedule(user) == schedule(tw)


def z_exact(tw, qubits):
    """Exact <Z_q> of the twin for the physical qubits `qubits` (order kept), by stabilizer simulation on the qubits
    the twin touches; gates on other qubits raise."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, StabilizerState
    touched = sorted({tw.find_bit(q).index for i in tw.data if i.operation.name not in ("barrier", "measure", "delay")
                      for q in i.qubits} | set(qubits))
    pos = {p: k for k, p in enumerate(touched)}
    r = QuantumCircuit(len(touched))
    for ins in tw.data:
        if ins.operation.name in ("measure", "barrier", "delay"):
            continue
        r.append(ins.operation, [pos[tw.find_bit(q).index] for q in ins.qubits])
    st = StabilizerState(r)
    n = len(touched)
    out = []
    for p in qubits:
        lab = ["I"] * n
        lab[n - 1 - pos[p]] = "Z"
        out.append(float(np.real(st.expectation_value(Pauli("".join(lab))))))
    return np.array(out)


def kept_share(z, exact):
    """r = sum z e / sum e^2 (the share of the exact pattern the device kept); None if the pattern is all zero."""
    z, e = np.asarray(z, float), np.asarray(exact, float)
    d = float(e @ e)
    return float(z @ e) / d if d > 0 else None
