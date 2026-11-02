"""Ported from TOQB tests/test_echo_tomo.py at 8d44f05 (imports changed to psf-os modules)."""
"""toqb.echo_tomo (2026-10-31 draft): the basis changes read X and Y correctly, the sign-aligned components of a
perfect echo are (0, 0, 1), and the verdict rules."""
import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from psf_os import scan as KC  # noqa: E402
from psf_os import tomo as ET  # noqa: E402


@pytest.mark.parametrize("prep,basis,bit", [("+", "X", 0), ("-", "X", 1), ("+i", "Y", 0), ("-i", "Y", 1),
                                            ("0", "Z", 0), ("1", "Z", 1)])
def test_basis_change_reads_the_right_axis(prep, basis, bit):
    qc = QuantumCircuit(1, 1)
    if prep in ("1", "-"):
        qc.x(0)
    if prep in ("+", "-", "+i", "-i"):
        qc.h(0)
    if prep == "+i":
        qc.s(0)
    if prep == "-i":
        qc.sdg(0)
    qc.measure(0, 0)
    c = ET.with_basis(qc, basis)
    probs = Statevector(c.remove_final_measurements(inplace=False)).probabilities()
    assert abs(probs[bit] - 1) < 1e-9


def _perfect(n, N=1000):
    out = {}
    for i in KC.INPUTS:
        bits = [1 - b for b in KC.input_bits(i, n)]
        out[i] = {"".join(str(b) for b in reversed(bits)): N}
    return out


def test_aligned_and_verdicts():
    a, se = ET.aligned(_perfect(6), 6)
    assert np.allclose(a, 1) and np.allclose(se, 0)
    assert ET.verdict_length(0.38, 0.42) == "KEPT"
    assert ET.verdict_length(0.10, 0.42) == "SHRUNK"
    assert ET.verdict_length(0.30, 0.42) == "MIXED"
    s = [0.02] * 5
    assert ET.verdict_depth([0.8, 0.3, -0.2, -0.1, 0.3], s) == "OSCILLATES"
    assert ET.verdict_depth([0.8, 0.3, -0.2, -0.1, 0.01], s) == "CROSSES ONCE"
    assert ET.verdict_depth([0.8, 0.5, 0.3, 0.1, 0.01], s) == "NO CROSSING"
    assert ET.verdict_depth([0.01, -0.3, -0.2, 0.3, 0.01], s) == "STARTS INVERTED"
    assert ET.verdict_corr(0.70, [0.20, 0.22, 0.21, 0.19]) == "ENTANGLED"
    assert ET.verdict_corr(0.15, [0.30, 0.32, 0.31, 0.33]) == "NOT SEEN"
    assert ET.sign_change([4, 8, 24], [0.5, 0.2, -0.2], [0.02, 0.02, 0.02]) == "OSCILLATES"
    assert ET.sign_change([4, 8, 24], [0.5, 0.2, 0.01], [0.02, 0.02, 0.02]) == "MONOTONE"


def test_connected_is_zero_for_a_product_and_one_for_a_bell_pair():
    n = 4
    prod = {i: {"0000": 500, "0001": 500} for i in KC.INPUTS}
    assert np.allclose(ET.connected(prod, n), 0)
    bell = {i: {"0000": 500, "0011": 500} for i in KC.INPUTS}
    c = ET.connected(bell, n)
    assert abs(c[0] - 1) < 1e-12 and np.allclose(c[1:], 0)


def test_defect_present():
    phys = list(range(10))
    a = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    assert ET.defect_present(phys, a, [3, 4, 9]) is True
    assert ET.defect_present(phys, a, [7, 8]) is False


def test_defect_present_weak_line():
    phys = list(range(10))
    a = np.array([0.4, 0.4, 0.15, 0.15, 0.15, 0.4, 0.4, 0.4, 0.4, 0.4])
    assert ET.defect_present(phys, a, [2, 3]) is False
    assert ET.defect_present(phys, a, [2, 3], rel=0.5) is True
