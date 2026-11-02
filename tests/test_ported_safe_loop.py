"""Ported from TOQB tests/test_safe_loop.py at 8d44f05 (imports changed to psf-os modules)."""
"""toqb.safe_loop (2026-10-31): the weak line, one defect seen again is not listed twice, and the depth verdicts."""
import numpy as np

from psf_os import scan as S
from psf_os import strict as SL


def test_weak_runs_use_the_ring_median():
    a = np.array([0.4] * 10 + [0.15, 0.10, 0.18] + [0.4] * 7)
    runs, line = S.weak_runs(a)
    assert abs(line - 0.2) < 1e-12 and runs == [(10, 3)]
    assert S.weak_runs(np.array([0.05] * 6 + [0.1] * 6))[1] == 0.12


def test_add_paths_merges_the_same_defect():
    lst = dict(params={}, paths=[dict(device="d", path=[28, 29, 38, 49])])
    scans = [dict(device="d", label="x", n=40, weak=[dict(centre=[27, 28, 29, 38], mean=0, length=19),
                                                     dict(centre=[35, 19, 15, 14], mean=0.1, length=4)])]
    new = SL.add_paths(lst, scans)
    assert [p["path"] for p in new["paths"]] == [[28, 29, 38, 49], [35, 19, 15, 14]]
    assert SL.paths_of(new, "d") == [(28, 29, 38, 49), (35, 19, 15, 14)]


def test_depth_verdicts():
    es = {"walls": {"0.0": {"fits": [dict(n=20, L_star=24.0, se=0.5), dict(n=40, L_star=24.2, se=0.4),
                                     dict(n=72, L_star=20.4, se=0.2)]}}}
    v = SL.depth_verdicts(es)
    assert v[40]["depth"] == "REACHED" and v[72]["depth"] == "SHORTER"
