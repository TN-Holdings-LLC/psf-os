"""P4 (dual-unitary kicked Ising chain): the CZ-form circuit gives the closed-form answer; the layer order matters."""
import numpy as np
import pytest

from psf_os import dual as D


@pytest.mark.parametrize("h", [0.0, 0.05, 0.15, 0.3])
def test_closed_form_on_small_chain(h):
    for t in range(6):
        assert np.allclose(D.check_statevector(11, t, h), D.exact(11, t, h), atol=1e-9)


def test_cz_count_of_logical_circuit():
    from psf_os.circuits import q2_count
    for t in range(6):
        assert q2_count(D.chain_circuit(11, t, 0.1)) == D.cz_count(11, t)


def test_off_point_is_not_the_closed_form():
    # b away from pi/4: the closed form no longer holds (the second stage has no formula)
    v = D.check_statevector(11, 4, 0.1, kick=np.pi / 4 - 0.1)
    assert not np.allclose(v, D.exact(11, 4, 0.1), atol=1e-3)


def test_exact_refuses_beyond_half_chain():
    with pytest.raises(ValueError):
        D.exact(11, 6, 0.1)
