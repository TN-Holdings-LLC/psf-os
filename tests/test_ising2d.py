"""P6 (the Qedma tutorial's 2D Ising patch): the CZ form equals the tutorial's gates, the cone answer equals the whole
patch, the six copies are exact copies on a Heron r2 lattice, and placement passes over a copy with a suspect qubit."""
import numpy as np

from psf_os import ising2d as I


def tutorial_circuit(qubits, layers, t):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(max(qubits) + 1)
    for _ in range(t):
        for q in sorted(qubits):
            qc.rx(I.THETA_X, q)
            qc.rz(I.THETA_Z, q)
        for L in layers:
            for a, b in L:
                qc.rzz(I.THETA_ZZ, a, b)
    return qc


def small():
    # the part of the patch on rows 0 and 1 near the left end: a 9-qubit piece with all three layers represented
    keep = {3, 4, 5, 6, 7, 16, 17, 23, 24}
    layers = [[e for e in L if e[0] in keep and e[1] in keep] for L in I.LAYERS]
    return sorted(keep), layers


def test_cz_form_equals_tutorial_gates():
    from qiskit.quantum_info import Operator
    q, L = small()
    idx = {x: k for k, x in enumerate(q)}
    cq, cl = list(range(len(q))), [[(idx[a], idx[b]) for a, b in l] for l in L]
    for t in (1, 2):
        a = Operator(I.circuit(t, measure=False, qubits=cq, layers=cl, width=len(q)))
        b = Operator(tutorial_circuit(cq, cl, t))
        assert a.equiv(b)


def test_cone_equals_whole_piece():
    from qiskit.quantum_info import Pauli, Statevector
    q0, L0 = small()
    idx = {x: k for k, x in enumerate(q0)}
    q, L = list(range(len(q0))), [[(idx[a], idx[b]) for a, b in l] for l in L0]
    edges = sorted(e for l in L for e in l)
    for t in (1, 2, 3, 4):
        sv = Statevector(I.circuit(t, measure=False, qubits=q, layers=L, width=len(q)))
        n = len(q)
        for x in q:
            ref = sv.expectation_value(Pauli("I" * (n - 1 - x) + "Z" + "I" * x)).real
            v, _ = I.answer_cone(x, t, edges=edges, layers=L)
            assert abs(v - ref) < 1e-10, (t, x)


def test_copies_are_exact_on_heron_r2():
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    cm = {tuple(sorted(e)) for e in FakeMarrakesh().coupling_map.get_edges()}
    for dr, dc, m in I.copies():
        assert len(set(m.values())) == len(I.QUBITS)
        for a, b in I.EDGES:
            assert tuple(sorted((m[a], m[b]))) in cm, (dr, dc, a, b)


def test_place_skips_suspect_copy():
    from psf_os.device import Device
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    cm = sorted({tuple(sorted(e)) for e in FakeMarrakesh().coupling_map.get_edges()})
    d = Device(name="toy", num_qubits=156, couplers=[list(e) for e in cm], failed_couplers=[[5, 6]], failed_qubits=[])
    dr, dc, m, reasons = I.place(d)
    assert (dr, dc) != (0, 0) and reasons[0]["drow"] == 0 and reasons[0]["dcol"] == 0


def test_theta_x_parameter_reaches_the_answer():
    import numpy as np
    for tx in (0.3, 0.8):
        assert abs(I.answer_cone(I.QUBITS[0], 1, tx=tx)[0] - np.cos(tx)) < 1e-12
