"""T5: the frame flip leaves the ideal circuit unchanged (operator equality, as TOQB's tests/test_kising_echo.py),
and the repair rules (mirror circuits only, two days, on the ring, no neighbouring flips)."""
from qiskit.quantum_info import Operator

from psf_os import circuits as C
from psf_os import repair


def strip(c):
    return c.remove_final_measurements(inplace=False)


def test_frame_flip_keeps_the_ideal_circuit():
    qc = C.echo_circuit([0, 1, 0, 1, 1, 0], 6, 2, 0.6, 0.0)
    ff = C.frame_flip(qc, 6)
    assert sum(1 for i in ff.data if i.operation.name == "x") == 2 * 3 * 2
    assert Operator(strip(ff)).equiv(Operator(strip(qc)))


def test_local_frame_flip_keeps_the_ideal_circuit():
    qc = C.echo_circuit([0, 1, 0, 1, 1, 0, 0, 1], 8, 2, 0.6, 0.0)
    for sites in ([2], [2, 4], [1, 3, 5]):
        assert Operator(strip(C.frame_flip(qc, 8, sites))).equiv(Operator(strip(qc)))


def test_repair_rules():
    ring = [10, 11, 12, 13, 14, 15, 16, 17]
    ready = [dict(device="d", path=[12, 13, 14], flip=[13], helped_on=["2026-10-31", "2026-11-01"])]
    assert repair.plan("echo", ring, ready)[0] == [3]
    assert repair.plan("sh6", ring, ready)[0] is None
    assert repair.plan("echo", ring, [])[0] is None
    assert repair.plan("echo", [1, 2, 3], ready)[0] is None
    two = ready + [dict(device="d", path=[13, 14], flip=[14], helped_on=["a", "b"])]
    assert repair.plan("echo", ring, two)[0] is None  # positions 3 and 4 are neighbours
