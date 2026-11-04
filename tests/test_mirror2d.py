"""psf_os.mirror2d: the user's circuit and its inverse as a sentinel."""
import numpy as np

from psf_os import device as Dv
from psf_os import ising2d as I
from psf_os import lattice2d as L
from psf_os import mirror2d as M
from psf_os import sim
from psf_os.compile import RuleCompiler


def small_piece():
    """A 6-qubit piece of the P6 patch (three edges of layer 1 and two of layer 2, connected) on FakeKingston."""
    layers = [[(6, 7), (24, 25)], [(5, 6), (7, 8), (23, 24)], [(22, 23), (7, 17)]]
    qubits = sorted({q for Lr in layers for e in Lr for q in e})
    return layers, qubits


def test_mirror_returns_to_zero_and_doubles_cz():
    dev = Dv.fetch_fake("FakeKingston")
    rc = RuleCompiler(dev)
    layers, qubits = small_piece()
    edges = [e for Lr in layers for e in Lr]
    dr, dc, m, _ = L.place(dev, edges, qubits)
    for t in (1, 3):
        u = I.circuit(t, mapping=m, width=dev.num_qubits, measure=False, qubits=qubits, layers=layers)
        c = rc.pm.run(M.mirror(u, [m[q] for q in qubits]))
        assert M.cz_count(c) == 2 * 2 * len(edges) * t
        (row,) = sim.run_circuits([c], 2000, seed=1)
        assert set(row["counts"]) == {"0" * len(qubits)}  # ideal: every shot returns to |0...0>


def test_barrier_is_what_keeps_the_join():
    """Without the barrier the pass manager may shorten the join; with it the CZ count is exactly doubled."""
    from qiskit import QuantumCircuit
    dev = Dv.fetch_fake("FakeKingston")
    rc = RuleCompiler(dev)
    layers, qubits = small_piece()
    edges = [e for Lr in layers for e in Lr]
    _, _, m, _ = L.place(dev, edges, qubits)
    u = I.circuit(2, mapping=m, width=dev.num_qubits, measure=False, qubits=qubits, layers=layers)
    plain = QuantumCircuit(u.num_qubits)
    plain.compose(u, inplace=True)
    plain.compose(u.inverse(), inplace=True)
    n_plain = M.cz_count(rc.pm.run(plain))
    n_mirror = M.cz_count(rc.pm.run(M.mirror(u, [m[q] for q in qubits])))
    assert n_mirror == 2 * M.cz_count(rc.pm.run(u))
    assert n_plain <= n_mirror
    print(f"CZ without barrier {n_plain}, with barrier {n_mirror}")


def test_p6_mirror_build_is_accepted(tmp_path):
    dev = Dv.fetch_fake("FakeKingston")
    rec = M.build(dev, dict(ts=[3, 5], theta_x=0.53, shots=4000), str(tmp_path / "runs"))
    assert rec["valid"] and rec["echo"]["q2_circuits"] == [2 * 2 * 38 * 3, 2 * 2 * 38 * 5]
    assert rec["copy"]["drow"] == I.place(dev)[0]


def test_flag_rule():
    rng = np.random.default_rng(1)
    z0 = 0.98 + 0.005 * rng.standard_normal(35)
    zm = 0.60 + 0.02 * rng.standard_normal(35)
    zm[[4, 17]] = [0.15, 0.25]          # two qubits lose the mirror signal
    zm[9] = 0.10
    z0[9] = 0.12                        # qubit 9 has a broken readout: its floor is low too, so its drop d is small
    f = M.flags(z0, zm)
    lab = I.QUBITS
    assert set(f["qubits"]) == {lab[4], lab[17]}
    assert M.flags(z0, np.full(35, 0.6))["qubits"] == []
