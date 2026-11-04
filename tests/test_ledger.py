"""psf_os.ledger: counts per kind, exclusion only after two jobs, the ring kind equal to the registry."""
import datetime as dt
import os

import numpy as np
import pytest

from psf_os import allocate as A
from psf_os import device as Dv
from psf_os import ledger as Lg
from psf_os import registry as R

TOQB = os.environ.get("TOQB_DIR")


def floor_readings(bad=(), n=30, seed=0):
    rng = np.random.default_rng(seed)
    labels = list(range(n))
    z0 = 0.985 + 0.004 * rng.standard_normal(n)
    for q in bad:
        z0[q] = 0.70
    se = np.full(n, 0.003)
    return Lg.read_floor(labels, z0, se)


def test_three_values_and_the_two_job_rule():
    r1, line = floor_readings(bad=(4,))
    assert dict((q, v) for q, v, *_ in r1)[4] == "DEAD"
    led = Lg.add(Lg.empty(), "fake_dev", "floor", "job1", "2026-11-03 06:00:00", r1, line)
    now = dt.datetime(2026, 11, 3, 12, 0)
    s = Lg.summary(led, "fake_dev", now)
    assert s[("floor", 4)]["status"] == "WATCH"                       # one job is not enough
    r2, line2 = floor_readings(bad=(4,), seed=1)
    led2 = Lg.add(led, "fake_dev", "floor", "job2", "2026-11-03 11:00:00", r2, line2)
    q, c, s2 = Lg.exclusions(led2, "fake_dev", now)
    assert q == [4] and c == []
    later = dt.datetime(2026, 11, 11, 12, 0)                           # more than 7 days after both
    assert Lg.exclusions(led2, "fake_dev", later)[0] == []
    # a reading near the line is UNCERTAIN and never excludes
    labels, z0, se = [0, 1, 2, 3, 4, 5, 6, 7], np.array([0.99] * 7 + [0.93]), np.array([0.003] * 7 + [0.02])
    rd, ln = Lg.read_floor(labels, z0, se)
    assert dict((q, v) for q, v, *_ in rd)[7] == "UNCERTAIN"
    led3 = Lg.add(Lg.add(Lg.empty(), "d", "floor", "a", "2026-11-03 06:00:00", rd, ln), "d", "floor", "b",
                  "2026-11-03 07:00:00", rd, ln)
    assert Lg.exclusions(led3, "d", now)[0] == []


def test_mirror_reading_flags_large_drops():
    labels = list(range(20))
    d = np.full(20, 0.30)
    d[[3, 9]] = [0.75, 0.80]
    rd, line = Lg.read_mirror(labels, d, np.full(20, 0.01))
    assert [q for q, v, *_ in rd if v == "DEAD"] == [3, 9]


@pytest.mark.skipif(not TOQB, reason="needs TOQB_DIR (the registry files of the coverage days)")
def test_ring_kind_equals_the_registry():
    dev = Dv.fetch_fake("FakeKingston")
    dev.name = "ibm_kingston"
    for f in ("results/coverage-days-2026-11-01-day2/list_after.json", "results/coverage-days-2026-11-02-day3/list_after.json"):
        reg = R.load(os.path.join(TOQB, f))
        for now in (dt.datetime(2026, 11, 2, 12, 0), dt.datetime(2026, 11, 3, 12, 0), dt.datetime(2026, 11, 12, 12, 0)):
            want = A.mask(dev, R.avoid_paths(reg, dev.name, now))[1]
            got = Lg.exclusions(Lg.empty(), dev.name, now, registry=reg)[1]
            assert got == want, (f, now)


def test_apply_marks_excluded_qubits_failed():
    dev = Dv.fetch_fake("FakeKingston")
    r1, l1 = floor_readings(bad=(6,), n=40)
    r2, l2 = floor_readings(bad=(6,), n=40, seed=2)
    led = Lg.add(Lg.add(Lg.empty(), dev.name, "floor", "j1", "2026-11-03 06:00:00", r1, l1), dev.name, "floor", "j2",
                 "2026-11-03 09:00:00", r2, l2)
    d, q, c = Lg.apply(dev, led, dt.datetime(2026, 11, 3, 12, 0))
    assert q == [6] and not d.qubit_ok(6) and dev.qubit_ok(6)
