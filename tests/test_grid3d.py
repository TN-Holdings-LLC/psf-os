"""Prototype 3D layer: correct without noise, refuses what it cannot place, reads back with the 2D card."""
import json

import numpy as np
import pytest

from psf_os import cli
from psf_os import device as Dv
from psf_os import grid3d as G


def _fez_like():
    import dataclasses
    return dataclasses.replace(Dv.fetch_fake("FakeFez"), failed_qubits=[], failed_couplers=[])


def test_placed_circuit_equals_the_lattice_without_noise():
    dev = _fez_like()
    for layout, sizes in (("fold", ((4, 2, 1), (4, 1, 2), (4, 2, 2), (4, 1, 3))), ("rows", ((4, 2, 1), (4, 2, 2)))):
        for (L, M, N) in sizes:
            qc, placed, used, passes = G.physical(dev, L, M, N, 2, 0.8, layout=layout)
            worst = G.exact_placed_check(qc, placed, used, L, M, N, 2, 0.8)
            assert worst < 1e-9, (layout, L, M, N, worst)


def _clifford(qc):
    from psf_os.compile import rzz_exact
    q = rzz_exact(qc)
    out = q.copy_empty_like()
    for ins in q.data:
        if ins.operation.name == "rz":
            k = int(round(float(ins.operation.params[0]) / (np.pi / 2))) % 4
            for _ in range(k):
                out.s(ins.qubits[0])
        elif ins.operation.name == "rx":
            out.sx(ins.qubits[0])
        else:
            out.append(ins.operation, ins.qubits)
    return out


def test_placed_state_equals_the_lattice_state_clifford():
    # theta = pi/2: every stabilizer of the lattice state holds on the placed sites and every other touched qubit is
    # back in |0> (bridges), at sizes beyond a statevector
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, StabilizerState
    dev = _fez_like()
    for layout, (L, M, N) in (("fold", (4, 3, 2)), ("fold", (4, 2, 3)), ("rows", (8, 2, 2))):
        qc, placed, used, _ = G.physical(dev, L, M, N, 2, np.pi / 2, layout=layout)
        n, m = L * M * N, {u: i for i, u in enumerate(used)}
        small = QuantumCircuit(len(used))
        for ins in qc.data:
            small.append(ins.operation, [m[qc.find_bit(q).index] for q in ins.qubits])
        ps = StabilizerState(_clifford(small))
        for lab in StabilizerState(_clifford(G.logical(L, M, N, 2, np.pi / 2))).clifford.to_labels(mode="S"):
            sign, body = lab[0], lab[1:]
            chars = ["I"] * len(used)
            for s in range(n):
                chars[m[placed[s]]] = body[n - 1 - s]
            assert abs(ps.expectation_value(Pauli("".join(reversed(chars)))) - (1 if sign == "+" else -1)) < 1e-9
        data = {m[placed[s]] for s in range(n)}
        for q in set(range(len(used))) - data:
            z = ["I"] * len(used)
            z[q] = "Z"
            assert abs(ps.expectation_value(Pauli("".join(reversed(z)))) - 1) < 1e-9


def test_relocates_around_a_failed_element_and_refuses_bad_sizes():
    import dataclasses
    good = _fez_like()
    with pytest.raises(G.Refused, match="L must be"):
        G.physical(good, 5, 2, 1, 1, 0.8)
    with pytest.raises(G.Refused, match="needs L = 4"):
        G.physical(good, 8, 2, 1, 1, 0.8, layout="fold")
    first = G.physical(good, 4, 2, 2, 1, 0.8, info=True)
    hit = first[2][0]
    moved = G.physical(dataclasses.replace(good, failed_qubits=[hit]), 4, 2, 2, 1, 0.8, info=True)
    assert hit not in moved[2] and (moved[4]["r0"], moved[4]["c0"]) != (first[4]["r0"], first[4]["c0"])
    every_row = [20 * r + c for r in range(8) for c in range(0, 16, 4)]  # column 0 of every block of every row
    with pytest.raises(G.Refused, match="no translation"):
        G.physical(dataclasses.replace(good, failed_qubits=every_row), 4, 2, 2, 1, 0.8)


def test_run3d_and_the_2d_card_on_a_local_run(tmp_path):
    # theta = pi/2 keeps the job Clifford, so the local stabilizer simulator can run all 156 qubits
    from psf_os import sim
    from psf_os import user2d as U
    dev = _fez_like()
    dev.save(str(tmp_path / "dev"))
    out = tmp_path / "run"
    assert cli.main(["run3d", "--device", str(tmp_path / "dev"), "--L", "4", "--M", "2", "--N", "1", "--t", "1",
                     "--theta", str(np.pi / 2), "--out", str(out), "--answers-out", str(tmp_path / "a.json")]) == 0
    rep = json.load(open(out / "report.json"))
    assert rep["circuits"] == 3 and rep["sites"] == 8
    sim.submit(rep["job"], "noiseless", seed=3)
    c = U.card(rep["job"], rep["runs"], json.load(open(tmp_path / "a.json"))["g3d"])
    assert abs(c["floor_mean"] - 1) < 1e-9 and abs(c["mirror_mean"] - 1) < 1e-9
    assert c["answered"]["qubits"] == 8


def test_refuses_pairs_that_are_not_couplers():
    import dataclasses
    good = _fez_like()
    qc, placed, used, _ = G.physical(good, 4, 2, 2, 1, 0.8, layout="fold")
    pairs = {frozenset(qc.find_bit(q).index for q in i.qubits) for i in qc.data if len(i.qubits) == 2}
    cut = next(iter(pairs))
    fewer = dataclasses.replace(good, couplers=[c for c in good.couplers if frozenset(c) != cut])
    moved = G.physical(fewer, 4, 2, 2, 1, 0.8, layout="fold", info=True)
    assert cut not in {frozenset(moved[0].find_bit(q).index for q in i.qubits) for i in moved[0].data
                       if len(i.qubits) == 2}
