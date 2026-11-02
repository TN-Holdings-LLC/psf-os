"""Logical ring circuits used by psf-os: the scan's mirror echo, the forward identity computations, the readout
reference and the frame flip.

Ported from TOQB at 8d44f05: toqb/task_loop.py (model_circuit), toqb/kising.py (step_euler), toqb/kising_echo.py
(step_angles, blocks, angles, circuit, frame_flip, q2_count), toqb/qca_test.py (circuit, circuit_sh) and
toqb/kising_charge.py (reference_circuit). Unchanged except for imports.
"""
from __future__ import annotations

import numpy as np


def model_circuit(angles, n, L):
    """TOQB's ring model: per layer RY on all, then RY RZ on all, then the CZ ring; final RY; measure all."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    it = iter(angles)
    for _ in range(L):
        for q in range(n):
            qc.ry(next(it), q)
        for q in range(n):
            qc.ry(next(it), q)
            qc.rz(next(it), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(next(it), q)
    qc.measure_all()
    return qc


def n_model_angles(n, L):
    return L * 3 * n + n


def step_euler(h):
    """(theta, phi, lam) with RX(2h) Z = e^{i a} RZ(phi) RY(theta) RZ(lam)."""
    from qiskit.synthesis import OneQubitEulerDecomposer
    rx = np.array([[np.cos(h), -1j * np.sin(h)], [-1j * np.sin(h), np.cos(h)]])
    u = rx @ np.diag([1.0, -1.0])
    theta, phi, lam = OneQubitEulerDecomposer("ZYZ").angles(u)
    return float(theta), float(phi), float(lam)


def _rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def _ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)


_X = np.array([[0, 1], [1, 0]], dtype=complex)


def step_angles(h, hz):
    theta, phi, lam = step_euler(h)
    return theta, phi, lam + 2 * hz


def blocks(L, h, hz):
    """The 2L + 1 single-qubit blocks V_0 .. V_2L around the 2L CZ rings of the echo (flip W = X^n at the turn)."""
    theta, phi, lamp = step_angles(h, hz)
    A = _rz(phi) @ _ry(theta)
    Api = _ry(theta) @ _rz(phi)
    R = _rz(lamp)
    return [R] + [R @ A] * (L - 1) + [R @ Api @ _X @ A] + [R @ Api] * (L - 1) + [np.eye(2, dtype=complex)]


def _zyz(V):
    from qiskit.synthesis import OneQubitEulerDecomposer
    t, p, l = OneQubitEulerDecomposer("ZYZ").angles(V)
    return float(p), float(t), float(l)


def echo_angles(bits, n, L, h, hz):
    if L < 1:
        raise ValueError("L >= 1")
    V = [_zyz(B) for B in blocks(L, h, hz)]
    D = 2 * L
    rz_slot = [0.0] * (D + 1)
    for k in range(1, D + 1):
        rz_slot[k] += V[k - 1][0] + V[k][2]
    a = []
    for k in range(1, D + 1):
        b_prev = V[k - 1][1]
        if k == 1:
            a += [np.pi * bits[q] for q in range(n)]
            for _ in range(n):
                a += [b_prev, rz_slot[1]]
        else:
            a += [b_prev] * n
            for _ in range(n):
                a += [0.0, rz_slot[k]]
    a += [V[D][1]] * n
    return a


def echo_circuit(bits, n, L, h=0.6, hz=0.0):
    """The mirror echo of the kicked-Ising ring (2L layers, identity up to the flip W): output = input flipped."""
    return model_circuit(echo_angles(bits, n, L, h, hz), n, 2 * L)


def _clifford_gates(V):
    """A 2x2 Clifford block as a list of parameterless gate names (h, s, sdg, x, y, z, ...), in time order."""
    from qiskit.quantum_info import Clifford, Operator
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.unitary(Operator(V), [0])
    return [i.operation.name for i in Clifford(qc).to_circuit().data]


def echo_clifford_circuit(bits, n, L):
    """The same mirror echo at h = pi/4, hz = 0 (the kicked Ising ring at its Clifford point), written with
    parameterless Clifford gates so that a stabilizer simulator can run it at any size. For simulation tests only:
    the scan itself uses h = 0.6."""
    from qiskit import QuantumCircuit
    V = [_clifford_gates(B) for B in blocks(L, np.pi / 4, 0.0)]
    qc = QuantumCircuit(n)
    for q in range(n):
        if bits[q]:
            qc.x(q)
    for k in range(1, 2 * L + 1):
        for q in range(n):
            for g in V[k - 1]:
                getattr(qc, g)(q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        for g in V[2 * L]:
            getattr(qc, g)(q)
    qc.measure_all()
    return qc


def frame_flip(qc, n, sites=None):
    """X on chosen ring positions just before and after the CZ ring of every even layer, plus the virtual Z that keeps
    the ideal circuit unchanged (see TOQB toqb/kising_echo.py frame_flip). Positions must not be ring neighbours."""
    from qiskit import QuantumCircuit
    if sites is None:
        if n % 2:
            raise ValueError("the full frame flip needs an even ring")
        sites = list(range(0, n, 2))
    F = sorted({int(q) % n for q in sites})
    if any((q + 1) % n in F for q in F):
        raise ValueError("flipped positions must not be neighbours on the ring")
    odd = [a for a in range(n) if a not in F and (((a - 1) % n in F) + ((a + 1) % n in F)) % 2]
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    incz, layer = False, 0

    def flips(after):
        if layer % 2 == 0:
            for q in F:
                out.x(out.qubits[q])
            if after:
                for a in odd:
                    out.z(out.qubits[a])
    for i in qc.data:
        nm = i.operation.name
        if nm == "cz" and not incz:
            incz, layer = True, layer + 1
            flips(False)
        elif nm != "cz" and incz:
            incz = False
            flips(True)
        out.append(i.operation, [out.qubits[qc.find_bit(q).index] for q in i.qubits],
                   [out.clbits[qc.find_bit(c).index] for c in i.clbits])
    if incz:
        flips(True)
    return out


def qca_circuit(bits, n):
    """n layers of (H on every qubit, CZ ring): the identity on an n-ring (forward, not a mirror)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for q, b in enumerate(bits):
        if b:
            qc.x(q)
    for _ in range(n):
        for q in range(n):
            qc.h(q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    qc.measure_all()
    return qc


def sh6_circuit(bits, n, D):
    """D layers alternating (S on all, CZ ring) and (H on all, CZ ring): the identity whenever D is a multiple of 6,
    at every ring length (forward, not a mirror; OS plan M3, second form)."""
    from qiskit import QuantumCircuit
    if D % 6:
        raise ValueError("D must be a multiple of 6")
    qc = QuantumCircuit(n)
    for q, b in enumerate(bits):
        if b:
            qc.x(q)
    for t in range(1, D + 1):
        for q in range(n):
            (qc.s if t % 2 else qc.h)(q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    qc.measure_all()
    return qc


def reference_circuit(compiled, bits):
    """The readout reference on the physical qubits `compiled` measures: X where the input bit is 1, then the same
    measurements (clbit q from the same physical qubit)."""
    from qiskit import ClassicalRegister, QuantumCircuit
    meas = {compiled.find_bit(i.clbits[0]).index: compiled.find_bit(i.qubits[0]).index
            for i in compiled.data if i.operation.name == "measure"}
    n = len(meas)
    qc = QuantumCircuit(compiled.num_qubits)
    qc.add_register(ClassicalRegister(compiled.num_clbits, "meas"))
    for q in range(n):
        if bits[q]:
            qc.x(meas[q])
    for q in range(n):
        qc.measure(meas[q], q)
    return qc, meas


def q2_count(c):
    return sum(1 for i in c.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "measure"))
