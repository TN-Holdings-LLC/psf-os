"""Reliability ledger v0: per device, per qubit (and coupler), counts of DEAD / HEALTHY / UNCERTAIN readings from the
OS's own checks, kept apart by kind. Used for one thing only: whether to leave a qubit or coupler out (as the
registry's LISTED does). No score is made, nothing is ranked, no route is optimised (owner, 10/14: no effort on
time-varying maps and searches around them; and the data: a gray stretch went from 0.10 back to 0.46 in 1.5 h).

Kinds (own measurements only; IBM's quality numbers are never read):
  ring    the ring echo (morning scan, M6 sentinel): read through the existing registry (registry.avoid_paths), so a
          LISTED or same-day gray path masks the same middle coupler as today (tests/test_ledger.py checks equality);
  floor   the theta = 0 floor <Z_q> of a job (readout): a qubit-level reading;
  mirror  the mirror sentinel's d_q = floor - mirror (gates): a qubit-level reading.

Three-valued reading of a job (the three-valued scan draft's rule, line +- 3 SE), with the job's own line (rule R's
form: median and MAD over the job's qubits):
  floor:  line = median(z0) - max(5 x 1.4826 MAD, 0.05); DEAD if z0 + 3 SE < line, HEALTHY if z0 - 3 SE >= line;
  mirror: line = median(d) + max(5 x 1.4826 MAD, 0.05); DEAD if d - 3 SE > line, HEALTHY if d + 3 SE <= line;
  UNCERTAIN otherwise. UNCERTAIN is reported, never acted on.
Exclusion rule (proposal of 11/04, same form as LISTED): a qubit is left out if, in one kind, two different jobs read
it DEAD within the last 7 days.

File: {"version": "ledger v0", "records": [{device, kind, unit "qubit", id, verdict, value, se, line, job, when}]},
append-only, oldest first (the registry's convention: a ledger file is never overwritten, `save` refuses).
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

KINDS = ("floor", "mirror")   # qubit-level kinds stored here; "ring" stays in the registry file
EXPIRY_DAYS = 7
MIN_JOBS = 2
K_SE = 3.0


def empty():
    return dict(version="ledger v0", records=[])


def load(path):
    return json.load(open(path))


def save(led, path):
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: a ledger file is never overwritten")
    json.dump(led, open(path, "w"), indent=1)


def _line(values, sign, k=5.0, margin=0.05):
    v = np.asarray(values, dtype=float)
    med = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - med)))
    return med - sign * max(k * sig, margin)


def read_floor(labels, z0, se):
    """[(id, verdict, value, se)] and the line, for one job's floor (higher is better)."""
    line = _line(z0, +1)
    out = []
    for q, v, s in zip(labels, z0, se):
        verdict = "DEAD" if v + K_SE * s < line else "HEALTHY" if v - K_SE * s >= line else "UNCERTAIN"
        out.append((int(q), verdict, float(v), float(s)))
    return out, line


def read_mirror(labels, d, se):
    """The same for the mirror's d_q = floor - mirror (lower is better)."""
    line = _line(d, -1)
    out = []
    for q, v, s in zip(labels, d, se):
        verdict = "DEAD" if v - K_SE * s > line else "HEALTHY" if v + K_SE * s <= line else "UNCERTAIN"
        out.append((int(q), verdict, float(v), float(s)))
    return out, line


def add(led, device, kind, job, when, readings, line):
    """A new ledger dict with one job's readings appended (`when`: 'YYYY-MM-DD HH:MM:SS' CET)."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r}: the ring kind lives in the registry")
    new = dict(led, records=list(led["records"]))
    for q, verdict, v, s in readings:
        new["records"].append(dict(device=device, kind=kind, unit="qubit", id=q, verdict=verdict, value=v, se=s,
                                   line=line, job=job, when=when))
    return new


def summary(led, device, now):
    """{(kind, id): {"dead": n, "healthy": n, "uncertain": n, "dead_jobs": [...], "last_seen": when, "status": ...}}
    status: EXCLUDED (DEAD in >= MIN_JOBS distinct jobs within EXPIRY_DAYS of now), WATCH (any DEAD or UNCERTAIN),
    OK otherwise."""
    out = {}
    for r in led["records"]:
        if r["device"] != device:
            continue
        key = (r["kind"], r["id"])
        e = out.setdefault(key, dict(dead=0, healthy=0, uncertain=0, dead_jobs=[], last_seen=None))
        e[r["verdict"].lower()] += 1
        t = dt.datetime.fromisoformat(r["when"])
        if r["verdict"] == "DEAD" and (now - t).total_seconds() <= EXPIRY_DAYS * 86400 and r["job"] not in e["dead_jobs"]:
            e["dead_jobs"].append(r["job"])
        e["last_seen"] = max(e["last_seen"] or r["when"], r["when"])
    for e in out.values():
        e["status"] = ("EXCLUDED" if len(e["dead_jobs"]) >= MIN_JOBS else
                       "WATCH" if e["dead"] or e["uncertain"] else "OK")
    return out


def exclusions(led, device, now, registry=None):
    """(qubits to leave out, couplers to mask, details). Couplers come from the ring kind (the registry's avoid_paths,
    unchanged); qubits from the floor and mirror kinds."""
    from psf_os import allocate as A
    from psf_os import registry as R
    couplers = []
    if registry is not None:
        avoid = R.avoid_paths(registry, device, now)
        couplers = sorted({A.middle_edge(a["path"]) for a in avoid if len(a["path"]) >= 2})
    s = summary(led, device, now)
    qubits = sorted({q for (kind, q), e in s.items() if e["status"] == "EXCLUDED"})
    return qubits, couplers, s


def apply(dev, led, now, registry=None):
    """A Device with the ledger's exclusions marked: excluded qubits as failed qubits, ring couplers masked."""
    import copy
    qubits, couplers, _ = exclusions(led, dev.name, now, registry)
    d = dev.with_mask(couplers, reason="ledger ring kind (registry)") if couplers else copy.deepcopy(dev)
    d.failed_qubits = sorted(set(d.failed_qubits) | set(qubits))
    return d, qubits, couplers
