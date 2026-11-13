"""psf_os.observables on small circuits (made-up circuits; shots by Qiskit's seeded statevector sampler)."""
import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import StatevectorSampler

from psf_os import observables as O
from psf_os.kim import z_values


def run(plan, shots=4000, seed=7):
    res = StatevectorSampler(seed=seed).run([c for _, c in plan["circuits"]], shots=shots).result()
    return [r.data.obs.get_counts() for r in res]


def small_circuit(n=5, seed=3):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(3):
        for q in range(n):
            qc.rx(rng.uniform(0, np.pi), q)
            qc.ry(rng.uniform(0, np.pi), q)
        for q in range(0, n - 1, 2):
            qc.cz(q, q + 1)
        for q in range(1, n - 1, 2):
            qc.rzz(rng.uniform(0, np.pi), q, q + 1)
    return qc


MIXED = [{"name": "Z0Z1", "pauli": "ZZIII"}, {"name": "X0X1", "pauli": "XXIII"},
         {"name": "Y2Z3X4", "pauli": {2: "Y", 3: "Z", 4: "X"}}, {"name": "Z0Z2Z4", "pauli": "ZIZIZ"},
         {"name": "Y1", "pauli": {1: "Y"}}, {"name": "X3", "pauli": "IIIXI"},
         {"name": "Z1Z3", "pauli": {1: "Z", 3: "Z"}}]


def test_basis_rotation_on_eigenstates():
    qc = QuantumCircuit(6)
    qc.x(1)                       # |1>: Z = -1
    qc.h(2)                       # |+>: X = +1
    qc.x(3)
    qc.h(3)                       # |->: X = -1
    qc.h(4)
    qc.s(4)                       # |+i>: Y = +1
    qc.h(5)
    qc.sdg(5)                     # |-i>: Y = -1
    obs = [{"name": f"{p}{q}", "pauli": {q: p}} for q, p in enumerate("ZZXXYY")]
    plan = O.build(qc, obs)
    vals = [v["value"] for v in O.values(plan, run(plan, shots=500))]
    assert vals == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    ex, how = O.exact(qc, obs)
    assert how == "statevector" and np.allclose(ex, [1, -1, 1, -1, 1, -1], atol=1e-12)
    # a product over the six: (+1)(-1)(+1)(-1)(+1)(-1) = -1
    allp = [{"name": "all", "pauli": "ZZXXYY"}]
    plan = O.build(qc, allp)
    assert O.values(plan, run(plan, shots=200))[0]["value"] == -1.0


def test_shots_agree_with_statevector_within_shot_error():
    qc = small_circuit()
    plan = O.build(qc, MIXED)
    vals = O.values(plan, run(plan, shots=20000, seed=11))
    ex, how = O.exact(qc, MIXED)
    assert how == "statevector"
    for v, e in zip(vals, ex):
        assert abs(v["value"] - e) <= 4 * v["se"] + 1e-9, (v, e)
    assert max(abs(e) for e in ex) > 0.1        # not a trivial all-zero case


def test_settings_group_qubitwise():
    qc = small_circuit()
    sets, where = O.settings(MIXED, 5)
    # Z0Z1 opens "ZZ---"; X0X1 needs a new one; Y2Z3X4 joins the first; Z0Z2Z4 cannot (qubit 2 is Y, 4 is X) ...
    for o, k in zip(MIXED, where):
        s = O.normalize(o, 5)
        assert all(sets[k][q] == ch for q, ch in s.items())
    zonly = [{"name": f"Z{q}", "pauli": {q: "Z"}} for q in range(5)] + [{"name": "ZZ", "pauli": "ZZZZZ"}]
    assert O.settings(zonly, 5)[0] == ["ZZZZZ"]
    plan = O.build(qc, MIXED)
    assert len(plan["circuits"]) == len(sets)


def test_z_only_equals_the_cards_reading():
    qc = small_circuit()
    obs = [{"name": f"Z{q}", "pauli": {q: "Z"}} for q in range(5)]
    plan = O.build(qc, obs)
    (counts,) = run(plan, shots=3000, seed=5)
    vals = O.values(plan, [counts])
    z, se = z_values(counts, 5)
    assert [v["value"] for v in vals] == pytest.approx(list(z), abs=1e-15)
    assert [v["se"] for v in vals] == pytest.approx(list(se), abs=1e-15)


def test_no_two_qubit_gate_added_and_rotations_only_at_the_end():
    qc = small_circuit()
    n2 = sum(1 for i in qc.data if len(i.qubits) == 2)
    for basis, c in O.build(qc, MIXED)["circuits"]:
        assert sum(1 for i in c.data if len(i.qubits) == 2) == n2
        tail = c.data[len(qc.data):]
        assert all(len(i.qubits) == 1 and i.operation.name in ("h", "sdg", "measure") for i in tail)
        assert sum(1 for i in tail if i.operation.name == "measure") == sum(ch != "-" for ch in basis)


def test_other_registers_in_the_keys():
    qc = small_circuit()
    qc.add_register(ClassicalRegister(2, "other"))       # present but unused: keys gain a second chunk
    obs = MIXED[:3]
    plan = O.build(qc, obs)
    counts = run(plan, shots=3000, seed=2)
    joined = [{f"{k} 01": v for k, v in c.items()} for c in counts]   # build()'s register is added last: leftmost
    a = O.values(plan, counts)
    b = O.values(plan, joined)
    assert [x["value"] for x in a] == [x["value"] for x in b]


def test_labels_for_physical_qubits():
    qc = small_circuit()
    labels = {80: 0, 81: 1, 82: 2, 83: 3, 84: 4}
    a = O.exact(qc, [{"name": "p", "pauli": {80: "X", 84: "Y"}}], labels=labels)[0]
    b = O.exact(qc, [{"name": "p", "pauli": "XIIIY"}])[0]
    assert a == pytest.approx(b, abs=1e-14)
    with pytest.raises(O.Refused):
        O.exact(qc, [{"name": "p", "pauli": {79: "Z"}}], labels=labels)


def test_stabilizer_for_large_clifford_and_equal_to_statevector_small():
    n = 26                                   # GHZ: above the statevector limit, Clifford
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    obs = [{"name": "Z0Z25", "pauli": {0: "Z", 25: "Z"}}, {"name": "Xall", "pauli": "X" * n},
           {"name": "Z3", "pauli": {3: "Z"}}, {"name": "YYX..", "pauli": "YY" + "X" * (n - 2)}]
    ex, how = O.exact(qc, obs)
    assert how == "stabilizer" and np.allclose(ex, [1, 1, 0, -1], atol=1e-12)
    small = QuantumCircuit(4)
    small.h(0)
    small.s(1)
    small.h(1)
    small.cz(0, 1)
    small.cx(1, 2)
    small.sdg(3)
    small.h(3)
    small.cz(2, 3)
    obs4 = [{"name": a, "pauli": a} for a in ("XZII", "YXZI", "IZZX", "ZIIZ", "XXYY", "IIIY")]
    sv = O.exact(small, obs4, method="statevector")[0]
    st = O.exact(small, obs4, method="stabilizer")[0]
    assert np.allclose(sv, st, atol=1e-12)
    with pytest.raises(O.Refused):
        O.exact(small_circuit(), MIXED, method="stabilizer")


def test_tn_equals_statevector():
    pytest.importorskip("quimb")
    pytest.importorskip("cotengra")
    qc = small_circuit()
    a = O.exact(qc, MIXED, method="tn")[0]
    b = O.exact(qc, MIXED, method="statevector")[0]
    assert np.allclose(a, b, atol=1e-9)


def test_verdict_is_the_cards():
    from psf_os.entrance import verdict as V
    e = [0.8, -0.5, 0.3, 0.1]
    v = [0.4, -0.25, 0.15, 0.05]                     # a pure shrink to one half
    got = O.verdict(v, e)
    assert got == V(v, e)
    assert got["r"] == pytest.approx(0.5) and got["reads"] == "READS" and got["of"] == 4
    assert O.verdict(e, e)["agrees"] == 4


def test_refuses():
    qc = small_circuit()
    for bad in ({"name": "a", "pauli": "ZZ"}, {"name": "b", "pauli": "IIIII"}, {"name": "c", "pauli": "ZZQII"},
                {"name": "d", "pauli": {7: "Z"}}, {"name": "e", "pauli": {1: "W"}}, {"name": "f", "pauli": 3}):
        with pytest.raises(O.Refused):
            O.build(qc, [bad])
    m = qc.copy()
    m.measure_all()
    with pytest.raises(O.Refused):
        O.build(m, MIXED)
