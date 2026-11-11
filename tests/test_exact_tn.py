"""psf_os.exact_tn (2026-11-11): light-cone tensor network against the statevector; Pauli products as per-site
tensors; the memory guard refuses before computing; slicing gives the same value; the "tn" routes of card.exact,
kim.answer_cone, ising2d.answer_cone and grid3d.exact. Heavy (72 sites): set PSF_OS_HEAVY=1."""
import math
import os

import numpy as np
import pytest

pytest.importorskip("quimb")
pytest.importorskip("cotengra")

from psf_os import card as C, exact_tn as E, grid3d, ising2d, kim

HEAVY = os.environ.get("PSF_OS_HEAVY")


def _sv_value(qc, p, s):
    from qiskit.quantum_info import SparsePauliOp, Statevector
    op = SparsePauliOp.from_sparse_list([(p, list(s), 1.0)], num_qubits=qc.num_qubits)
    return float(np.real(Statevector(qc).expectation_value(op)))


def test_random_circuits_equal_statevector():
    from qiskit.circuit.random import random_circuit
    rng = np.random.default_rng(7)
    worst = 0.0
    for seed in range(6):
        n = 3 + seed % 4
        qc = random_circuit(n, 5, max_operands=3, seed=seed)
        qc.remove_final_measurements()
        for _ in range(3):
            k = int(rng.integers(1, n + 1))
            s = tuple(sorted(rng.choice(n, k, replace=False).tolist()))
            p = "".join(rng.choice(list("XYZ"), k))
            worst = max(worst, abs(E.expectation(qc, (p, s))[0] - _sv_value(qc, p, s)))
    assert worst < 1e-10


def test_observable_forms_and_identity():
    qc = grid3d.logical(2, 2, 2, 2, 0.7)
    a = E.expectations(qc, [("ZX", (1, 4)), {1: "Z", 4: "X"}, {0: "I"}])
    assert abs(a[0] - a[1]) < 1e-14 and a[2] == 1.0
    assert abs(a[0] - _sv_value(qc, "ZX", (1, 4))) < 1e-10


def test_guard_refuses_before_computing_and_slicing_agrees():
    qc = grid3d.logical(4, 2, 2, 3, 0.8)
    ref = grid3d.exact(4, 2, 2, 3, 0.8)["5"]
    v, r = E.expectation(qc, {5: "Z"}, max_size=2 ** 4, max_peak=2 ** 10)
    assert r["slices"] > 1 and r["max_intermediate"] <= 2 ** 4 and abs(v - ref) < 1e-10
    with pytest.raises(E.TooLarge) as err:
        E.expectation(qc, {5: "Z"}, max_size=2 ** 3, slicing=False)
    assert "guard" in err.value.reason and "contract_s" not in err.value.report


def test_card_tn_equals_statevector_and_cli(capsys):
    cd = C.load(os.path.join(os.path.dirname(__file__), "..", "examples", "cards", "kicked_ising_4x2x2.json"))
    cd["time"]["steps"] = 2
    cd["measure"] = cd["measure"] + [{"op": "XY", "sites": [0, 1]}]
    a = C.exact(cd, method="statevector")
    info = []
    b = C.exact(cd, method="tn", info=info)
    assert max(abs(a[k] - b[k]) for k in a) < 1e-10 and len(info) == len(a)
    with pytest.raises(C.CardRefused) as err:
        C.exact(cd, method="tn", max_size=2, slicing=False, simplify=False)
    assert "memory guard" in err.value.reasons[0]


def test_card_over_28_sites_by_tn_only():
    cd = C.load(os.path.join(os.path.dirname(__file__), "..", "examples", "cards", "kicked_ising_4x2x2.json"))
    cd["lattice"]["size"] = [4, 4, 2]
    cd["initial"]["bits"] = "0" * 32
    cd["measure"] = [{"op": "Z", "sites": [0]}, {"op": "Z", "sites": [13]}]
    with pytest.raises(C.CardRefused):
        C.exact(cd, method="statevector")
    with pytest.raises(C.CardRefused):
        C.check(cd)                                   # the entrance's default is unchanged
    assert C.check(cd, exact=False)["sites"] == 32
    ans = C.exact(cd)                                 # auto -> tn above 20 sites
    ref = C.exact(dict(cd, measure=[{"op": "Z", "sites": [0]}]), method="tn")
    assert abs(ans["Z_0"] - ref["Z_0"]) < 1e-14
    th = 0.6                                          # kicked Ising at one step: <Z> = cos(theta) on every site
    assert abs(ans["Z_13"] - math.cos(th)) < 1e-10


def test_kim_and_ising2d_cones_beyond_by_tn():
    from test_kim import EDGES, QUBITS
    for q in (0, 4, 9):
        a, n = kim.answer_cone(QUBITS, EDGES, q, 0.8, 3)
        b, m = kim.answer_cone(QUBITS, EDGES, q, 0.8, 3, max_qubits=3, beyond="tn")
        assert n == m and n > 3 and abs(a - b) < 1e-10
    with pytest.raises(ValueError):
        kim.answer_cone(QUBITS, EDGES, 4, 0.8, 3, max_qubits=3)
    a, n = ising2d.answer_cone(ising2d.QUBITS[10], 2)
    assert ising2d.answer_cone(ising2d.QUBITS[10], 2, max_qubits=2) == (None, n)
    b, m = ising2d.answer_cone(ising2d.QUBITS[10], 2, max_qubits=2, beyond="tn")
    assert m == n and abs(a - b) < 1e-10


@pytest.mark.skipif(not HEAVY, reason="heavy: set PSF_OS_HEAVY=1 (72 sites, minutes)")
def test_grid3d_72_sites_clifford_point_against_stabilizer():
    from qiskit.quantum_info import Clifford, Pauli
    from qiskit_aer import AerSimulator
    L, M, N, t = 4, 6, 3, 4
    n = L * M * N
    qc = grid3d.logical(L, M, N, t, math.pi / 2)
    cl = Clifford(qc).to_circuit()
    for q in range(0, n, 7):
        cl.save_expectation_value(Pauli("I" * (n - 1 - q) + "Z" + "I" * q), list(range(n)), label=f"z{q}")
    data = AerSimulator(method="stabilizer").run(cl).result().data()
    got = grid3d.exact(L, M, N, t, math.pi / 2, beyond="tn")
    for q in range(0, n, 7):
        assert abs(got[str(q)] - float(np.real(data[f"z{q}"]))) < 1e-10
    with pytest.raises(ValueError):
        grid3d.exact(L, M, N, t, 0.8)
