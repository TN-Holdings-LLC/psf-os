"""L5: sentinels (OS plan M6, prototype). A few mirror-echo circuits on the user's own ring, in the user's own job.

The morning scan cannot see a gray defect that appears an hour later. A sentinel rides in the same submission as the
user's circuits, on the same physical qubits, so it sees the ring at the time the job ran:

- what is added: the rule's echo (D = 24, hz 0, h 0.6, the four inputs) placed by the same compiler on the same
  device record, so it lands on the user's ring (checked). No readout references are added: the per-site signal
  needs none, and the user's own references are in the job. At most 8 circuits (4 here).
- the user's circuits are not touched: the sentinel is a separate record, assembled after them (the manifest says
  which cells and pubs are the user's).
- the verdict uses the registry's rules unchanged (TOQB defect_list / safe_loop): a dead run (a_j < 0.12 on 3 or more
  consecutive sites) -> DEAD; else a weak run (a_j < max(0.12, half the ring median)) -> WEAK; else CLEAN. The sites
  are reported as ring positions (logical bits) and physical qubits.
- what it writes: the runs it found go into a new registry file as gray entries of the job's day. They are not black
  observations and do not count towards LISTED (black entries come from scans only).
"""
from __future__ import annotations

import os

import numpy as np

from psf_os import job as J
from psf_os import registry as R
from psf_os import scan as S

SENTINEL_JOB = dict(kind="echo", D=24, hz=0.0, h=0.6)
MAX_CIRCUITS = 8


def name_for(n, D):
    return f"ksent_n{n}_D{D}-fold_rule"


FALLBACKS = (None, "same_qubits")


def build(dev, runs, user_ring, shots, compiler, spec=None):
    """Add the sentinel record to `runs` (the user's record is already there). Returns the record.

    spec may carry "fallback" (2026-11-13, not used unless asked; the rest of spec is the sentinel's job as before):
      None (default): the sentinel must land on the user's ring in the same order, as always;
      "same_qubits": if the order differs but the physical qubits are the same set, the sentinel is kept and its record
        says so ("fallback" in the record). Seen on ibm_marrakesh 2026-11-12 (14 failed qubits): a 40-site ring needs a
        cycle of 32 with 8 pendants, and the echo's pendants come in another order than the user's model circuit's,
        so every job was refused although the sentinel watched exactly the user's qubits. The verdict reads the
        sentinel's own order (its record), so dead and weak runs stay runs of neighbours on the sentinel's ring.
    When the rings are equal nothing changes: the same record and the same circuits as without the option."""
    spec = dict(spec or SENTINEL_JOB)
    fallback = spec.pop("fallback", None)
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback {fallback!r}: one of {FALLBACKS}")
    spec = dict(spec, n=len(user_ring), shots=shots)
    rec = J.build(dev, spec, runs, compiler=compiler, role="sentinel", refs=False, name=name_for(spec["n"], spec["D"]))
    mq = rec["echo"]["measured_qubits"]
    ring = [int(mq[str(j)]) for j in range(spec["n"])]
    if ring != list(user_ring):
        if fallback == "same_qubits" and sorted(ring) == sorted(int(q) for q in user_ring):
            rec["fallback"] = dict(rule="same_qubits", note="the sentinel's ring order differs from the user's; same "
                                   "physical qubits (psf_os.sentinel.build, fallback same_qubits)",
                                   user_ring=[int(q) for q in user_ring], sentinel_ring=ring)
            import json
            json.dump(rec, open(os.path.join(runs, name_for(spec["n"], spec["D"]) + ".json"), "w"), indent=1)
            return rec
        raise RuntimeError("the sentinel did not land on the user's ring")
    if rec["export"]["circuits"] > MAX_CIRCUITS:
        raise RuntimeError(f"the sentinel has {rec['export']['circuits']} circuits, over {MAX_CIRCUITS}")
    return rec


def verdict(job_dir, runs):
    """{"status": CLEAN | WEAK | DEAD | NONE, per sentinel cell: ring, signal, runs (positions and qubits)}."""
    out = {"status": "NONE", "cells": {}}
    order = {"NONE": 0, "CLEAN": 1, "WEAK": 2, "DEAD": 3}
    for name, (rec, cc) in J.read(job_dir, runs).items():
        if rec.get("psfos_role") != "sentinel":
            continue
        n = rec["task"]["n"]
        mq = rec["echo"]["measured_qubits"]
        ring = [int(mq[str(j)]) for j in range(n)]
        a = S.site_signal({i: cc[(i, "echo")] for i in S.INPUTS}, n)
        dead = S.dead_runs(a)
        weak, line = S.weak_runs(a)
        dead_sites = {(s0 + t) % n for s0, L0 in dead for t in range(L0)}
        weak = [(s0, L0) for s0, L0 in weak if not ({(s0 + t) % n for t in range(L0)} & dead_sites)]
        st = "DEAD" if dead else "WEAK" if weak else "CLEAN"

        def runs_of(rs):
            return [dict(positions=[(s + t) % n for t in range(L)], qubits=[ring[(s + t) % n] for t in range(L)],
                         centre=list(S.centre_path(ring, s, L)), mean=float(np.mean([a[(s + t) % n] for t in range(L)])))
                    for s, L in rs]
        out["cells"][name] = dict(status=st, n=n, ring=ring, a=[float(x) for x in a], median=float(np.median(a)),
                                  weak_line=line, dead=runs_of(dead), weak=runs_of(weak))
        if order[st] > order[out["status"]]:
            out["status"] = st
    return out


def to_registry(reg, device, job_dir, runs):
    """A copy of the registry with the sentinel's runs (dead and weak) added as gray entries of the job's day."""
    import copy
    import json
    sub = json.load(open(os.path.join(job_dir, "submission.json")))
    day = R._time(sub["submitted"]).date().isoformat()
    new = copy.deepcopy(reg)
    for name, c in verdict(job_dir, runs)["cells"].items():
        for kind in ("dead", "weak"):
            for r in c[kind]:
                new.setdefault("gray", []).append(dict(device=device, day=day, job=sub.get("job_id"), n=c["n"],
                                                       line=c["weak_line"], path=r["centre"], qubits=r["qubits"],
                                                       mean=r["mean"], source=f"sentinel ({kind})"))
    return new
