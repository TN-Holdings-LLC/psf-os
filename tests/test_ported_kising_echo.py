"""Ported from TOQB tests/test_kising_echo.py at 8d44f05 (imports changed to psf-os modules)."""
import numpy as np
import pytest

from psf_os import circuits as C
from psf_os import scan as KC
from psf_os import scan as S

INPUTS = KC.INPUTS


def _flipped_probability(c, bits):
    from qiskit.quantum_info import Statevector
    n = len(bits)
    p = Statevector(c.remove_final_measurements(inplace=False)).probabilities()
    return p[sum((1 - bits[q]) << q for q in range(n))]


@pytest.mark.parametrize("n", [6, 8])
def test_noiseless_echo_returns_the_flipped_input(n):
    for hz in (0.0, 0.1, 0.3):
        for L in range(1, 6):
            for i in KC.INPUTS:
                b = KC.input_bits(i, n)
                assert _flipped_probability(C.echo_circuit(b, n, L, 0.6, hz), b) > 1 - 1e-12


def test_forward_step_is_the_kicked_ising_step_with_hz():
    """V_0, the CZ ring and A make one step S = (RX(2h) Z RZ(2 hz))^n CZ_ring (up to a global phase)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    n, h = 4, 0.6
    for hz in (0.0, 0.1, 0.3):
        theta, phi, lamp = C.step_angles(h, hz)
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.rz(lamp, q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
        for q in range(n):
            qc.ry(theta, q)
            qc.rz(phi, q)
        rx = np.array([[np.cos(h), -1j * np.sin(h)], [-1j * np.sin(h), np.cos(h)]])
        u1 = rx @ np.diag([1.0, -1.0]) @ np.diag([np.exp(-1j * hz), np.exp(1j * hz)])
        U = np.array([[1.0]])
        for _ in range(n):
            U = np.kron(u1, U)
        cz = np.ones(2 ** n)
        for q in range(n):
            b = np.arange(2 ** n)
            cz *= np.where(((b >> q) & 1) & ((b >> ((q + 1) % n)) & 1), -1, 1)
        ref = U @ np.diag(cz)
        got = Operator(qc).data
        k = np.argmax(np.abs(ref))
        ph = got.flat[k] / ref.flat[k]
        assert np.max(np.abs(got - ph * ref)) < 1e-10 and abs(abs(ph) - 1) < 1e-10
        if hz == 0.0:  # toqb.kising's step: the same Euler angles
            assert np.allclose(C.step_angles(h, 0.0), C.step_euler(h))


def test_estimate_from_synthetic_counts():
    n = 4
    cc = {}
    for i in KC.INPUTS:
        b = KC.input_bits(i, n)
        flipped = "".join(str(1 - b[q]) for q in reversed(range(n)))
        same = "".join(str(b[q]) for q in reversed(range(n)))
        cc[(i, "echo")] = {flipped: 900, same: 100}  # each Z_j right with 0.9: <Z_j> s_j = 0.8
        cc[(i, "ref")] = {same: 1000}
    est = S.estimate(cc, n)
    assert abs(est["rho4"] - 0.8) < 1e-12 and est["se4"] > 0
    assert abs(est["rho_w2"] - 1.0) < 1e-12  # both flips leave Z_j Z_j+1 unchanged


def test_compiled_rule_keeps_every_two_qubit_gate_and_returns():
    """As TOQB's test for the fold:rule arm, on psf-os's compiler and FakeMarrakesh."""
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit.quantum_info import Statevector
    from psf_os.compile import RuleCompiler, measured_qubits
    from psf_os.device import fetch_fake
    from psf_os.sim import compact
    A = RuleCompiler(fetch_fake("FakeMarrakesh"))
    n = 12
    per = A.per_layer_q2(n)
    for L in (1, 2):
        for hz in (0.0, 0.3):
            b = KC.input_bits("neel", n)
            c = A.compile(C.echo_circuit(b, n, L, 0.6, hz))
            assert C.q2_count(c) == 2 * L * per
            mm = measured_qubits(c)
            small, act = compact(c.remove_final_measurements(inplace=False))
            pos = {q: k for k, q in enumerate(act)}
            sv = Statevector(small)
            for j in range(n):
                z = 1 - 2 * sv.probabilities([pos[mm[j]]])[1]
                assert abs(z + (1 - 2 * b[j])) < 1e-9
