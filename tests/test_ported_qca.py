"""Ported from TOQB tests/test_qca.py at 8d44f05 (imports changed to psf-os modules)."""
"""toqb.qca_test (2026-11-01): n layers of (H on all, CZ ring) return every input on the n-ring, and the scorer."""
import pytest

pytest.importorskip("qiskit")
from qiskit.quantum_info import StabilizerState  # noqa: E402

from psf_os import circuits as C  # noqa: E402
from psf_os import job as Q  # noqa: E402
from psf_os import scan as KC  # noqa: E402


@pytest.mark.parametrize("n", [8, 20, 40])
def test_qca_is_the_identity_on_the_inputs(n):
    for i in KC.INPUTS:
        bits = KC.input_bits(i, n)
        qc = C.qca_circuit(bits, n).remove_final_measurements(inplace=False)
        (key, p), = StabilizerState(qc).probabilities_dict().items()
        assert p == 1.0 and [int(key[n - 1 - j]) for j in range(n)] == list(bits)


def test_scores_of_a_perfect_run():
    n = 6
    cc = {i: {"".join(str(b) for b in reversed(KC.input_bits(i, n))): 100} for i in KC.INPUTS}
    a, se = Q.site_signal_forward(cc, n)
    assert all(abs(x - 1) < 1e-12 for x in a) and Q.full_match(cc, n) == 1.0


@pytest.mark.parametrize("n,D", [(8, 6), (20, 12), (40, 24)])
def test_sh6_is_the_identity_on_the_inputs(n, D):
    for i in KC.INPUTS:
        bits = KC.input_bits(i, n)
        qc = C.sh6_circuit(bits, n, D).remove_final_measurements(inplace=False)
        (key, p), = StabilizerState(qc).probabilities_dict().items()
        assert p == 1.0 and [int(key[n - 1 - j]) for j in range(n)] == list(bits)
