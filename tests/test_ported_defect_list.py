"""Ported from TOQB tests/test_defect_list.py at 8d44f05 (imports changed to psf-os modules)."""
"""toqb.defect_list (2026-10-31 draft): dead runs on a cyclic ring, centre paths, path containment, and the list's
status rules (listed after repeats, candidate after one sighting, expired after the expiry time, contradictions)."""
import datetime as dt

import numpy as np

from psf_os import registry as R
from psf_os import scan as S


def test_dead_runs_wrap_and_min_length():
    a = np.array([0.0, 0.0, 0.4, 0.4, 0.0, 0.4, 0.4, 0.0, 0.05])  # run over the wrap: 7, 8, 0, 1
    assert S.dead_runs(a, 0.12, 3) == [(7, 4)]
    assert sorted(S.dead_runs(a, 0.12, 1)) == [(4, 1), (7, 4)]
    assert S.dead_runs(np.zeros(5), 0.12, 3) == [(0, 5)]
    assert S.dead_runs(np.ones(5), 0.12, 1) == []


def test_centre_path_and_containment():
    phys = (10, 11, 12, 13, 14, 15, 16, 17)
    assert S.centre_path(phys, 1, 6, 4) == (12, 13, 14, 15)
    assert S.centre_path(phys, 6, 4, 2) == (17, 10)
    assert S.centre_path(phys, 0, 2, 4) == (10, 11)
    cyc = [5, 12, 13, 14, 15, 9]
    assert S.contains_path(cyc, (12, 13, 14, 15))
    assert S.contains_path(cyc, (15, 14, 13, 12))
    assert S.contains_path(cyc, (9, 5, 12))
    assert not S.contains_path(cyc, (12, 14))


def _obs(job, when, ring, dead_centre=None, device="dev"):
    dead = [] if dead_centre is None else [dict(start=0, length=5, qubits=list(dead_centre), centre=list(dead_centre),
                                                mean=-0.1)]
    return dict(device=device, job=job, label=job, submitted=f"x ({when} CET)", n=len(ring), ring=list(ring),
                median=0.3, dead=dead)


def test_entries_status_rules():
    c = (3, 4, 5, 6)
    obs = [_obs("j1", "2026-10-30 12:00:00", [1, 2, 3, 4, 5, 6, 7], c),
           _obs("j2", "2026-10-30 14:00:00", [1, 2, 3, 4, 5, 6, 7], tuple(reversed(c))),
           _obs("j3", "2026-10-30 15:00:00", [9, 3, 4, 5, 6, 8]),             # healthy ring through the centre path
           _obs("j4", "2026-10-30 15:00:00", [20, 21, 22, 23], (21, 22, 23))]
    ent = {tuple(e["centre"]): e for e in R.entries(obs, min_repeats=2, expiry_days=7)}
    assert ent[c]["status"] == "LISTED" and ent[c]["jobs"] == ["j1", "j2"] and ent[c]["sightings"] == 2
    assert ent[c]["contradictions"] == [dict(job="j3", n=6)]
    assert ent[(21, 22, 23)]["status"] == "CANDIDATE"
    later = dt.datetime(2026, 11, 8, 12, 0)
    ent2 = {tuple(e["centre"]): e for e in R.entries(obs, 2, 7, now=later)}
    assert ent2[c]["status"] == "EXPIRED"


def test_shifted_runs_merge_to_their_common_core():
    obs = [_obs("j1", "2026-10-30 12:00:00", list(range(80, 92)), (87, 86, 85, 84)),
           _obs("j2", "2026-10-30 14:00:00", list(range(80, 92)), (88, 87, 86, 85))]
    ent = R.entries(obs, min_repeats=2, expiry_days=7)
    assert len(ent) == 1 and ent[0]["status"] == "LISTED" and ent[0]["centre"] == [87, 86, 85]


def test_usable_and_coverage_verdict():
    a = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.05, 0.5, 0.0, 0.0])  # one dead run (1-3); 5 and 7-8 are too short
    assert S.usable(a, 0.12, 3) == [0, 4, 5, 6, 7, 8]
    assert S.usable(a, 0.12, 2) == [0, 4, 5, 6]
    assert S.coverage_verdict(22, 40) == "WIDER"
    assert S.coverage_verdict(22, 23) == "SAME"
    assert S.coverage_verdict(40, 30) == "NARROWER"


def test_path_present():
    phys = [10, 11, 12, 13, 14, 15]
    a = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.5])
    assert S.path_present(phys, a, (11, 12, 13, 14)) is True
    assert S.path_present(phys, a, (13, 14, 15)) is False
