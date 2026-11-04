"""The self-repair chain (workplace, 2026-11-04; draft, not declared): find -> confirm -> avoid -> re-place -> verify,
every step a rule fixed in advance and applied mechanically (owner, 2026-11-04 07:57 CET: not counted as search).

  find     a job's floor (theta = 0 cells) and mirror (mirror2d, d_q = floor - mirror) give per-qubit readings in the
           ledger's three-valued form (psf_os.ledger.read_floor / read_mirror: line from the job's own median and MAD,
           DEAD if the value is beyond the line by 3 SE, HEALTHY if inside by 3 SE, UNCERTAIN otherwise). The ring
           sentinel goes to the registry under its existing rules (unchanged here: a LISTED or same-day gray path masks
           its middle coupler).
  confirm  rule "a": DEAD in two different jobs within 7 days in one kind (the ledger's rule);
           rule "b": DEAD in one job (three-valued, value + 3 SE < line for the floor) is enough.
           The declaration picks one.
  release  "expiry": an exclusion lapses 7 days after its last DEAD reading (the ledger's rule);
           "early": also lapses as soon as a later job of the same kind reads the qubit HEALTHY.
           The owner picks one.
  avoid    confirmed qubits are marked failed in a copy of the device record (with the registry's masks).
  re-place lattice2d.place on that record: the first qualifying copy in the fixed order, no search. None qualifying
           -> the caller's rule (the next device by queue).
  verify   the next job holds the same problem twice: A on the copy chosen without the chain, B on the copy chosen with
           it. A and B on the same copy -> NOT TESTED.

What the chain cannot do (stated first, the workplace's five points): readout faults move within hours, so a flag may
be stale by the next job; on a device's first job there is nothing to confirm from (inside one job only rule R, a
scoring drop, acts); a static ZZ is not seen by a mirror; the patch avoids, it does not correct; and a repair chosen
after seeing the same job's results would be a choice after the fact, so the chain acts only on later jobs.

Every run of the chain returns a log: readings, confirmations with their reasons, released exclusions with reasons,
the excluded qubits, the copy chosen for A and for B and the copies passed over with their reasons.
"""
from __future__ import annotations

import copy as _copy
import datetime as dt
import json
import os

import numpy as np

from psf_os import ledger as Lg

RULES = ("a", "b")
RELEASES = ("expiry", "early")


# ------------------------------------------------------------------------------------------------ find

def readings(labels, z_floor, se_floor, z_mirror=None, se_mirror=None):
    """{kind: (readings, line)} for one job: floor always, mirror when given (d = floor - mirror)."""
    out = {"floor": Lg.read_floor(labels, z_floor, se_floor)}
    if z_mirror is not None:
        d = np.asarray(z_floor, float) - np.asarray(z_mirror, float)
        sd = np.sqrt(np.asarray(se_floor, float) ** 2 + np.asarray(se_mirror, float) ** 2)
        out["mirror"] = Lg.read_mirror(labels, d, sd)
    return out


def record(led, device, job, when, rd):
    """The ledger with one job's readings appended (all kinds)."""
    for kind, (r, line) in rd.items():
        led = Lg.add(led, device, kind, job, when, r, line)
    return led


# ------------------------------------------------------------------------------------------------ confirm

def confirmed(led, device, now, rule="a", release="expiry"):
    """({qubit: [reasons]}, {(kind, qubit): release reason}) at `now` (records after `now` are ignored)."""
    if rule not in RULES or release not in RELEASES:
        raise ValueError(f"rule {rule!r} / release {release!r}")
    need = 2 if rule == "a" else 1
    by = {}
    for r in led["records"]:
        if r["device"] != device:
            continue
        t = dt.datetime.fromisoformat(r["when"])
        if t > now:
            continue
        by.setdefault((r["kind"], r["id"]), []).append((t, r["job"], r["verdict"]))
    out, released = {}, {}
    for (kind, q), rs in sorted(by.items()):
        rs.sort()
        dead = [(t, j) for t, j, v in rs if v == "DEAD" and (now - t).total_seconds() <= Lg.EXPIRY_DAYS * 86400]
        jobs = sorted({j for _, j in dead})
        if len(jobs) < need:
            if any(v == "DEAD" for _, _, v in rs) and not dead:
                released[(kind, q)] = "expired (last DEAD more than 7 days ago)"
            continue
        last = max(t for t, _ in dead)
        if release == "early":
            h = [j for t, j, v in rs if v == "HEALTHY" and t > last]
            if h:
                released[(kind, q)] = f"released early: HEALTHY in {h[0]} after the last DEAD"
                continue
        out.setdefault(q, []).append(f"{kind}: DEAD in {', '.join(jobs)}")
    return out, released


# ------------------------------------------------------------------------------------------------ avoid, re-place

def avoid(dev, qubits):
    d = _copy.deepcopy(dev)
    d.failed_qubits = sorted(set(d.failed_qubits) | set(int(q) for q in qubits))
    return d


def replace(dev, edges, qubits=None, offsets=None):
    """dict(copy=(drow, dcol) or None, mapping, passed_over)."""
    from psf_os import lattice2d as L2
    try:
        dr, dc, m, why = L2.place(dev, edges, qubits, offsets=offsets)
        return dict(copy=(dr, dc), mapping={int(k): int(v) for k, v in m.items()}, passed_over=why)
    except RuntimeError as e:
        return dict(copy=None, mapping=None, passed_over=str(e))


def chain(dev, led, now, edges, qubits=None, rule="a", release="expiry", offsets=None):
    """One run of the chain on `dev` (already carrying the registry's masks). Returns the log."""
    exc, rel = confirmed(led, dev.name, now, rule, release)
    A = replace(dev, edges, qubits, offsets)
    B = replace(avoid(dev, exc), edges, qubits, offsets)
    in_A = sorted(q for q in exc if A["mapping"] and q in set(A["mapping"].values()))
    status = ("NO COPY (A)" if A["copy"] is None else "NO COPY (B): the caller's next-device rule" if B["copy"] is None
              else "NOT TESTED (A and B on the same copy)" if A["copy"] == B["copy"] else "TESTABLE")
    return dict(device=dev.name, now=now.isoformat(sep=" "), rule=rule, release=release,
                excluded={int(q): r for q, r in exc.items()}, excluded_in_A=in_A,
                released={f"{k[0]}:{k[1]}": v for k, v in rel.items()}, A=A, B=B, status=status,
                price=dict(excluded=len(exc), excluded_in_A=len(in_A), copy_moved=_moved(A, B)))


def _moved(A, B):
    if A["copy"] is None or B["copy"] is None:
        return None
    return dict(drow=B["copy"][0] - A["copy"][0], dcol=B["copy"][1] - A["copy"][1],
                shared_qubits=len(set(A["mapping"].values()) & set(B["mapping"].values())))


# ------------------------------------------------------------------------------------------------ job 2 (verify)

def build(dev, log, job, runs, compiler=None, name=None):
    """One record holding the P6 problem on A's and B's copies, interleaved (A t1, B t1, A t2, ...). job = {"ts",
    "theta_x", "shots"}. Acceptance: 2 x 38 x t CZs and every patch qubit measured, per point."""
    from psf_os import circuits as C
    from psf_os import ising2d as I
    from psf_os.compile import VERSION, RuleCompiler, measured_qubits
    from psf_os.crosstalk2d import _write
    if log["status"] != "TESTABLE":
        raise RuntimeError(f"chain status {log['status']}: B is not built")
    rc = compiler or RuleCompiler(dev)
    ts = [int(t) for t in job["ts"]]
    tx = float(job.get("theta_x", I.THETA_X))
    cs, points, ok = [], [], True
    for t in ts:
        for arm in ("A", "B"):
            m = {int(k): v for k, v in log[arm]["mapping"].items()}
            c = rc.pm.run(I.circuit(t, mapping=m, width=dev.num_qubits, tx=tx))
            want = {k: m[q] for k, q in enumerate(I.QUBITS)}
            ok &= measured_qubits(c) == want and C.q2_count(c) == 2 * len(I.EDGES) * t
            cs.append(c)
            points.append(dict(input=f"{arm}{t}", setting="z", t=t, arm=arm))
    name = name or f"prep2d_tx{tx:g}_qedma-patch"
    rec = dict(kind="psfos", psfos_role="user", psfos_job=dict(job, kind="repair2d"), arm="patch:A/B",
               version=VERSION, device=dev.name, masked=dev.masked, chain=log,
               mapping_A={str(k): v for k, v in log["A"]["mapping"].items()},
               mapping_B={str(k): v for k, v in log["B"]["mapping"].items()},
               echo=dict(kind="repair2d", accepted=bool(ok)), points=points, valid=bool(ok),
               shots=job.get("shots", 4000))
    return _write(runs, name, cs, rec)


def mse(z_by_t, exact_by_t):
    """Mean over t and qubits of (z - exact)^2."""
    sq = [(np.asarray(z_by_t[t]) - np.asarray(exact_by_t[t])) ** 2 for t in z_by_t if t in exact_by_t]
    return float(np.mean(np.concatenate(sq)))


def verdict(mse_a, mse_b, se_diff, spread=None, lo=0.8, hi=1.25):
    """HELPS / NO EFFECT / HURTS on the ratio B / A (0.8 / 1.25), with the owner's uncertainty rule: the difference
    A - B must also exceed 2 x the larger of its shot SE and the between-run spread, otherwise NO EFFECT."""
    u = max(se_diff, spread or 0.0)
    r = mse_b / mse_a
    if r <= lo and mse_a - mse_b > 2 * u:
        return "HELPS", r, u
    if r >= hi and mse_b - mse_a > 2 * u:
        return "HURTS", r, u
    return "NO EFFECT", r, u


def score(cells, exact, n, nboot=200, seed=0, spread=None):
    """cells: {("A"|"B", t): counts}; exact: {t: [<Z> per label]}. MSE per arm, the bootstrap SE of A - B (shots,
    fixed seed) and the verdict."""
    from psf_os import crosstalk2d as X
    parsed = {k: X.bits(c, n) for k, c in cells.items()}

    def stat(p):
        z = {arm: {t: X.zz(c, n, [])[0] for (a, t), c in p.items() if a == arm} for arm in ("A", "B")}
        return mse(z["A"], exact), mse(z["B"], exact)
    ma, mb = stat(parsed)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(nboot):
        a, b = stat({k: X.resample(c, rng) for k, c in parsed.items()})
        diffs.append(a - b)
    se = float(np.std(diffs, ddof=1))
    v, r, u = verdict(ma, mb, se, spread)
    return dict(mse_A=ma, mse_B=mb, ratio=r, se_diff=se, spread=spread, uncertainty=u, verdict=v, nboot=nboot,
                seed=seed)


def save_log(log, path):
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: a chain log is never overwritten")
    json.dump(log, open(path, "w"), indent=1)
