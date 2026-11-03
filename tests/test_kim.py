"""P5 (kicked Ising on the device lattice): the causal-cone answer equals the whole lattice's; the Clifford point by
stabilizer equals the statevector; CZ count and colouring."""
import numpy as np
import pytest

from psf_os import kim as K

# a 14-qubit piece of a heavy-hex lattice (two hexagons sharing an edge, with bridge qubits and two pendants)
EDGES = sorted(tuple(sorted(e)) for e in [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),       # ring A
    (4, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 3),                 # ring B shares (3, 4)
    (0, 13)])                                                             # a pendant
QUBITS = sorted({q for e in EDGES for q in e})


def full(theta, t):
    from qiskit.quantum_info import Pauli, Statevector
    sv = Statevector(K.circuit(QUBITS, EDGES, theta, t, measure=False))
    n = max(QUBITS) + 1
    return [float(sv.expectation_value(Pauli("I" * (n - 1 - q) + "Z" + "I" * q)).real) for q in QUBITS]


@pytest.mark.parametrize("theta", [0.3, 0.8])
def test_cone_equals_whole_lattice(theta):
    for t in range(1, 5):
        a = K.answers(QUBITS, EDGES, theta, t)
        assert np.allclose(a["z"], full(theta, t), atol=1e-10), (theta, t)


def test_cone_one_step_smaller_is_wrong():
    # the radius t - 1 is needed: radius t - 2 gives a different answer somewhere (the argument is tight)
    theta, t = 0.8, 3
    ref = full(theta, t)
    bad = []
    for k, q in enumerate(QUBITS):
        b = K.ball(EDGES, q, t - 2)
        s = set(b)
        ce = [e for e in EDGES if e[0] in s and e[1] in s]
        from qiskit.quantum_info import Pauli, Statevector
        idx = {p: i for i, p in enumerate(b)}
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(len(b))
        for op in K.ops(b, ce, theta, t):
            if op[0] == "cz":
                qc.cz(idx[op[1]], idx[op[2]])
            else:
                getattr(qc, op[0])(op[1], idx[op[2]])
        z = Statevector(qc).expectation_value(Pauli("I" * (len(b) - 1 - idx[q]) + "Z" + "I" * idx[q])).real
        bad.append(abs(z - ref[k]))
    assert max(bad) > 1e-3


def test_clifford_point_stabilizer_equals_statevector():
    for t in range(1, 4):
        a = K.answers(QUBITS, EDGES, np.pi / 2, t)
        assert a["method"] == "stabilizer"
        assert np.allclose(a["z"], full(np.pi / 2, t), atol=1e-10)


def test_trivial_point_and_counts():
    assert K.answers(QUBITS, EDGES, 0.0, 3)["z"] == [1.0] * len(QUBITS)
    assert np.allclose(full(0.0, 3), 1)
    from psf_os.circuits import q2_count
    assert q2_count(K.circuit(QUBITS, EDGES, 0.4, 3)) == 3 * len(EDGES)
    for L in K.colour(EDGES):
        qs = [q for e in L for q in e]
        assert len(qs) == len(set(qs))


def test_suspect_region_drops_ends_of_failed_couplers():
    from psf_os.device import Device
    couplers = [list(e) for e in EDGES]
    d = Device(name="toy", num_qubits=14, couplers=couplers, failed_couplers=[[3, 4]], failed_qubits=[])
    q0, e0 = K.region(d)
    q1, e1 = K.region(d, suspect=True)
    assert 3 in q0 and 4 in q0 and (3, 4) not in e0
    assert 3 not in q1 and 4 not in q1
    assert all(3 not in e and 4 not in e for e in e1)
