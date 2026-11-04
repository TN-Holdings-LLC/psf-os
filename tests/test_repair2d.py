"""psf_os.repair2d: a planted bad qubit is excluded and an unplanted one is not; rule a needs two jobs, rule b one;
release by expiry and early; the whole flow job 1 -> chain -> job 2 -> score on a fake device with synthetic counts."""
import datetime as dt
import json
import os

import numpy as np
import pytest

from psf_os import device as Dv
from psf_os import ising2d as I
from psf_os import job as J
from psf_os import ledger as Lg
from psf_os import repair2d as Rp

N = len(I.QUBITS)


@pytest.fixture(scope="module")
def dev():
    return Dv.fetch_fake("FakeKingston")


def floor_of(m, bad=(), seed=0, base=0.98, badval=0.70):
    """Per-label floor readings (sorted labels), with the physical qubits in `bad` low."""
    rng = np.random.default_rng(seed)
    z = base + 0.004 * rng.standard_normal(N)
    for k, q in enumerate(I.QUBITS):
        if m[q] in bad:
            z[k] = badval
    return z, np.full(N, 0.003)


def job_readings(m, bad, seed):
    z, se = floor_of(m, bad, seed)
    phys = [m[q] for q in I.QUBITS]
    return Rp.readings(phys, z, se)


def test_planted_bad_qubit_is_excluded_and_others_are_not(dev):
    A = Rp.replace(dev, I.EDGES, I.QUBITS, I.OFFSETS)
    m = A["mapping"]
    bad = m[I.QUBITS[5]]
    led = Rp.record(Lg.empty(), dev.name, "job1", "2026-11-04 10:00:00", job_readings(m, {bad}, 1))
    now = dt.datetime(2026, 11, 4, 10, 30)
    log = Rp.chain(dev, led, now, I.EDGES, I.QUBITS, rule="b", offsets=I.OFFSETS)
    assert list(log["excluded"]) == [bad]
    assert log["status"] == "TESTABLE" and bad not in log["B"]["mapping"].values()
    assert bad in log["A"]["mapping"].values() and log["price"]["excluded_in_A"] == 1
    # rule a: one job is not enough
    log_a = Rp.chain(dev, led, now, I.EDGES, I.QUBITS, rule="a", offsets=I.OFFSETS)
    assert log_a["excluded"] == {} and log_a["status"].startswith("NOT TESTED")


def test_rule_a_two_jobs_and_the_two_releases(dev):
    m = Rp.replace(dev, I.EDGES, I.QUBITS, I.OFFSETS)["mapping"]
    bad = m[I.QUBITS[7]]
    led = Rp.record(Lg.empty(), dev.name, "j1", "2026-11-04 08:00:00", job_readings(m, {bad}, 1))
    led = Rp.record(led, dev.name, "j2", "2026-11-04 12:00:00", job_readings(m, {bad}, 2))
    t = dt.datetime(2026, 11, 4, 13, 0)
    assert list(Rp.confirmed(led, dev.name, t, "a")[0]) == [bad]
    # a later HEALTHY reading releases it only under "early"
    led2 = Rp.record(led, dev.name, "j3", "2026-11-04 18:00:00", job_readings(m, set(), 3))
    t2 = dt.datetime(2026, 11, 4, 19, 0)
    assert list(Rp.confirmed(led2, dev.name, t2, "a", "expiry")[0]) == [bad]
    exc, rel = Rp.confirmed(led2, dev.name, t2, "a", "early")
    assert exc == {} and "released early" in rel[("floor", bad)]
    # expiry: 7 days after the last DEAD
    t3 = dt.datetime(2026, 11, 11, 12, 30)
    assert Rp.confirmed(led, dev.name, t3, "a")[0] == {}
    # records after `now` are ignored (no look-ahead)
    assert Rp.confirmed(led, dev.name, dt.datetime(2026, 11, 4, 9, 0), "a")[0] == {}


def _counts(z, rng, shots=4000):
    p1 = (1 - np.asarray(z)) / 2
    b = (rng.random((shots, len(z))) < p1).astype(int)
    keys, cnt = np.unique(b[:, ::-1], axis=0, return_counts=True)
    return {"".join(map(str, k)): int(c) for k, c in zip(keys, cnt)}


def test_flow_job1_chain_job2_score(dev, tmp_path):
    """Synthetic counts only (product states from made-up answers; plumbing, not physics). The planted qubit reads
    0.4 lower in both jobs; the chain (rule b) moves B off it and the score says HELPS."""
    from psf_os.compile import RuleCompiler
    rng = np.random.default_rng(7)
    exact = {t: [0.86 - 0.05 * t] * N for t in (1, 2, 3)}
    rc = RuleCompiler(dev)
    A = Rp.replace(dev, I.EDGES, I.QUBITS, I.OFFSETS)
    bad = A["mapping"][I.QUBITS[10]]
    hurt = {k for k, q in enumerate(I.QUBITS) if A["mapping"][q] == bad}

    def planted(z, mapping):
        z = np.array(z, float)
        for k, q in enumerate(I.QUBITS):
            if mapping[q] == bad:
                z[k] -= 0.4
        return z
    # job 1: the floor (theta_x 0) on A's copy
    z0 = planted([0.98] * N, A["mapping"])
    c0 = _counts(z0, rng)
    from psf_os.kim import z_values
    zf, sef = z_values(c0, N)
    rd = Rp.readings([A["mapping"][q] for q in I.QUBITS], zf, sef)
    led = Rp.record(Lg.empty(), dev.name, "job1", "2026-11-04 20:00:00", rd)
    log = Rp.chain(dev, led, dt.datetime(2026, 11, 4, 20, 5), I.EDGES, I.QUBITS, rule="b", offsets=I.OFFSETS)
    assert list(log["excluded"]) == [bad], log["excluded"]
    Rp.save_log(log, str(tmp_path / "chain.json"))
    with pytest.raises(FileExistsError):
        Rp.save_log(log, str(tmp_path / "chain.json"))
    # job 2: build, assemble, synthetic counts, read back, score
    runs, jd = str(tmp_path / "runs"), str(tmp_path / "job")
    rec = Rp.build(dev, log, dict(ts=[1, 2, 3], theta_x=0.53, shots=4000), runs, compiler=rc)
    assert rec["valid"]
    man = J.assemble(dev, runs, jd, max_seconds=70)
    rows = []
    for p in man["pubs"]:
        pt = rec["points"][p["point"]]
        mp = {int(k): v for k, v in rec[f"mapping_{pt['arm']}"].items()}
        rows.append(dict(counts=_counts(planted(exact[pt["t"]], mp), rng)))
    json.dump(dict(rows=rows), open(os.path.join(jd, "raw_counts.json"), "w"))
    cells = {}
    for name, (r, cc) in J.read(jd, runs).items():
        for pt in r["points"]:
            cells[(pt["arm"], pt["t"])] = cc[(pt["input"], "z")]
    s = Rp.score(cells, exact, N, nboot=50)
    assert s["verdict"] == "HELPS" and s["ratio"] < 0.8 and hurt
    # with a between-run spread larger than the difference the verdict falls back to NO EFFECT
    s2 = Rp.score(cells, exact, N, nboot=20, spread=1.0)
    assert s2["verdict"] == "NO EFFECT"
