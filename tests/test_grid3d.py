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
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    dev = _fez_like()
    for (L, M, N) in ((4, 2, 1), (4, 1, 2), (4, 2, 2)):
        qc, placed, used, passes = G.physical(dev, L, M, N, 2, 0.8)
        m = {u: i for i, u in enumerate(used)}
        small = QuantumCircuit(len(used))
        for ins in qc.data:
            small.append(ins.operation, [m[qc.find_bit(q).index] for q in ins.qubits])
        p = Statevector(small).probabilities()
        idx = np.arange(len(p))
        ex = G.exact(L, M, N, 2, 0.8)
        worst = max(abs(float(np.sum(p * (1 - 2 * ((idx >> m[placed[s]]) & 1)))) - ex[str(s)]) for s in placed)
        assert worst < 1e-9, (L, M, N, worst)


def test_refuses_a_failed_element_and_bad_sizes():
    import dataclasses
    good = _fez_like()
    with pytest.raises(G.Refused, match="L must be"):
        G.physical(good, 5, 2, 1, 1, 0.8)
    bad = dataclasses.replace(good, failed_qubits=[21])
    with pytest.raises(G.Refused, match="failed element"):
        G.physical(bad, 4, 2, 1, 1, 0.8)


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
