"""Tests of psf_os.twin: same schedule, Clifford, exact Z against a statevector, and refusal of non-basis gates."""
import numpy as np
import pytest

from psf_os import twin as T


def _compiled(n=6, layers=4, seed=7):
    from qiskit import QuantumCircuit, transpile
    rng = np.random.default_rng(seed)
    c = QuantumCircuit(n)
    for k in range(1, n, 2):
        c.x(k)
    for layer in range(layers):
        for a in range(layer % 2, n - 1, 2):
            c.rxx(rng.uniform(0, 1), a, a + 1)
            c.ryy(rng.uniform(0, 1), a, a + 1)
            c.rz(rng.uniform(0, 3), a)
    return transpile(c, basis_gates=["cz", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)


def test_same_schedule_and_clifford():
    from qiskit.quantum_info import Clifford
    c = _compiled()
    tw = T.twin(c)
    assert T.check_same_schedule(c, tw)
    assert tw.count_ops().get("cz") == c.count_ops().get("cz")
    Clifford(tw)  # raises if any gate is not Clifford
    for ins in tw.data:
        if ins.operation.name == "rz":
            assert abs(float(ins.operation.params[0]) / (np.pi / 2) - round(float(ins.operation.params[0]) / (np.pi / 2))) < 1e-12


def test_z_exact_against_statevector():
    from qiskit.quantum_info import Statevector
    c = _compiled()
    tw = T.twin(c)
    qs = list(range(c.num_qubits))
    z = T.z_exact(tw, qs)
    sv = Statevector(tw)
    p = sv.probabilities_dict()
    ref = [sum(v * (1 if k[::-1][q] == "0" else -1) for k, v in p.items()) for q in qs]
    assert np.allclose(z, ref, atol=1e-12)


def test_refuses_non_basis():
    from qiskit import QuantumCircuit
    c = QuantumCircuit(2)
    c.h(0)
    with pytest.raises(ValueError):
        T.twin(c)


def test_kept_share():
    assert T.kept_share([0.5, -0.5], [1, -1]) == pytest.approx(0.5)
    assert T.kept_share([0.1], [0.0]) is None
