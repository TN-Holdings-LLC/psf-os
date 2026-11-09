"""grid3d layout "line" (psf_os.grid3d_line; VS Claude 2026-11-08, ported 2026-11-09): exact without noise, fewer CZ
than the fold, the 2D acceptance checks, reachable through prepare/build, never chosen by "auto"."""
import dataclasses

import pytest

from psf_os import device as Dv
from psf_os import grid3d as G


def _fez(**marks):
    return dataclasses.replace(Dv.fetch_fake("FakeFez"), **{**dict(failed_qubits=[], failed_couplers=[]), **marks})


@pytest.mark.parametrize("t", [1, 2, 3])
def test_line_equals_the_lattice_without_noise(t):
    dev = _fez()
    qc, placed, used, _ = G.physical(dev, 4, 2, 2, t, 0.6, layout="line")
    assert G.exact_placed_check(qc, placed, used, 4, 2, 2, t, 0.6) < 1e-9


def test_line_cz_and_acceptance():
    dev = _fez()
    line = [G.prepare(dev, 4, 2, 2, t, 0.6, layout="line")["cz"] for t in (1, 2)]
    fold = [G.prepare(dev, 4, 2, 2, t, 0.6, layout="fold")["cz"] for t in (1, 2)]
    assert line == [60, 120] and fold == [108, 216]
    p = G.prepare(dev, 4, 3, 2, 1, 0.6, layout="line")
    assert p["cz"] == 126 and all(p["checks"].values()) and p["where"]["layout"] == "line"


def test_line_at_the_folds_place():
    dev = _fez()
    f = G.prepare(dev, 4, 2, 2, 2, 0.6, layout="fold")["where"]
    at = [(f["r0"], f["c0"])]
    p = G.prepare(dev, 4, 2, 2, 2, 0.6, layout="line", at=at)
    assert (p["where"]["r0"], p["where"]["c0"]) == at[0]
    qc, placed, used, _ = G.physical(dev, 4, 2, 2, 2, 0.6, layout="line", at=at)
    assert G.exact_placed_check(qc, placed, used, 4, 2, 2, 2, 0.6) < 1e-9


def test_line_moves_off_a_failed_qubit_and_auto_stays_fold():
    dev = _fez()
    *_, w0 = G.physical(dev, 4, 2, 2, 1, 0.6, layout="line", info=True)
    first = 20 * w0["r0"] + w0["c0"]
    *_, w1 = G.physical(_fez(failed_qubits=[first]), 4, 2, 2, 1, 0.6, layout="line", info=True)
    assert (w1["r0"], w1["c0"]) != (w0["r0"], w0["c0"])
    *_, wa = G.physical(dev, 4, 2, 2, 1, 0.6, info=True)
    assert wa["layout"].startswith("fold")
    with pytest.raises(G.Refused):
        G.physical(dev, 4, 4, 3, 1, 0.6, layout="line")
