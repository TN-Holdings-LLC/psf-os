"""Ring sentinel ledger (W3, prototype; not used by run2d). The 2D jobs' ring sentinel (echo D = 24 on the rule's
n = 40 ring) has said DEAD on ibm_fez in every job since 2026-11-03, always on the same stretch. A sentinel that is
always DEAD carries no news. This module:

  history(jobs)      the sentinel verdict of each finished job: device, date, status, the dead qubits in ring order;
  persistent(h, ...) per device, the qubits dead in at least `min_jobs` jobs over at least `min_days` days, and the
                     ring paths they form (consecutive in ring order);
  ring_after(dev, paths)
                     where the rule's ring lands once those paths are masked (allocate.mask, the registry's mechanism),
                     and whether a given user edge set would still be placed on the same copy.

Nothing here changes a job. Masking for the sentinel through the registry also masks for the user's placement (the
same device record is used for both in user2d.build_batch); `ring_after` reports that effect so the owner can decide.
"""
from __future__ import annotations

import numpy as np

from psf_os import sentinel


def history(jobs):
    """jobs: [(date, job_dir, runs_dir)]. Returns [dict(date, device, status, ring, dead=[qubits], a=[signal])]."""
    from psf_os import job as J
    out = []
    for date, jd, runs in jobs:
        v = sentinel.verdict(jd, runs)
        if v["status"] == "NONE":
            continue
        dev = None
        for _, (rec, _) in J.read(jd, runs).items():
            dev = rec.get("device") or dev
            break
        for c in v["cells"].values():
            out.append(dict(date=date, device=dev, status=c["status"], ring=c["ring"], median=c["median"],
                            dead=[q for d in c["dead"] for q in d["qubits"]], a=c["a"]))
    return out


def persistent(h, min_jobs=3, min_days=2):
    """{device: dict(qubits=[...], paths=[[...]], jobs=n, days=n)} of the qubits dead often enough."""
    out = {}
    for dev in sorted({r["device"] for r in h}):
        rows = [r for r in h if r["device"] == dev]
        count, days = {}, {}
        for r in rows:
            for q in r["dead"]:
                count[q] = count.get(q, 0) + 1
                days.setdefault(q, set()).add(r["date"])
        qs = {q for q in count if count[q] >= min_jobs and len(days[q]) >= min_days}
        if not qs:
            out[dev] = dict(qubits=[], paths=[], jobs=len(rows), days=len({r["date"] for r in rows}))
            continue
        ring = rows[-1]["ring"]
        n = len(ring)
        paths, cur = [], []
        start = next((k for k in range(n) if ring[k] not in qs), 0)
        for j in range(n):
            q = ring[(start + j) % n]
            if q in qs:
                cur.append(q)
            elif cur:
                paths.append(cur)
                cur = []
        if cur:
            paths.append(cur)
        out[dev] = dict(qubits=sorted(qs), paths=paths, jobs=len(rows), days=len({r["date"] for r in rows}),
                        counts={str(q): count[q] for q in sorted(qs)})
    return out


def ring_after(dev, paths, edges_for_placement=None, ring=40):
    """dict(ring_before, ring_after, avoids=[...], copy_before, copy_after) with `paths` masked."""
    from psf_os import allocate as A
    from psf_os import circuits as C
    from psf_os import lattice2d as L
    from psf_os.compile import RuleCompiler, measured_qubits

    def ring_on(d):
        rc = RuleCompiler(d)
        mq = measured_qubits(rc.compile(C.model_circuit(list(np.zeros(C.n_model_angles(ring, 1))), ring, 1)))
        return [int(mq[j]) for j in range(ring)]
    masked, edges = A.mask(dev, [p for p in paths if len(p) >= 2], reason="ring2d prototype: persistent sentinel DEAD")
    out = dict(masked_couplers=[list(e) for e in edges], ring_before=ring_on(dev), ring_after=ring_on(masked))
    bad = {q for p in paths for q in p}
    out["after_avoids_dead"] = not (set(out["ring_after"]) & bad)
    if edges_for_placement is not None:
        def copy(d):
            try:
                dr, dc, _, _ = L.place(d, edges_for_placement)
                return [dr, dc]
            except RuntimeError:
                return None
        out["copy_before"], out["copy_after"] = copy(dev), copy(masked)
    return out


def verdict_beside_ledger(row, known_qubits):
    """The sentinel verdict recomputed with the ledger's known-dead qubits set aside (report only; no circuit changes).
    row: one entry of history(). The known sites take the ring median, so they can neither make nor hide a run; the
    rest is judged by the registry's rules unchanged. Returns dict(status_all, status_new, new_dead=[qubits])."""
    from psf_os import scan as S
    ring, a = row["ring"], np.array(row["a"], dtype=float)
    known = [k for k, q in enumerate(ring) if q in set(known_qubits)]
    b = a.copy()
    b[known] = float(np.median(a))
    n = len(ring)
    dead = S.dead_runs(b)
    weak, _ = S.weak_runs(b)
    new_dead = [ring[(s + t) % n] for s, L in dead for t in range(L)]
    return dict(status_all=row["status"], status_new="DEAD" if dead else "WEAK" if weak else "CLEAN",
                new_dead=new_dead, set_aside=[ring[k] for k in known])
