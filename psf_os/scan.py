"""L1: reading the health scan. The rule's mirror echo, read per ring site, and its dead and weak runs.

Ported from TOQB toqb/defect_list.py (site_signal, job_profiles, dead_runs, centre_path, contains_path, usable,
observe) and toqb/safe_loop.py (weak_runs, scan_job) at 8d44f05. The arithmetic is kept identical so that the
registry made here matches TOQB's lists exactly (test T1).

A job folder is TOQB's format: manifest.json (cells, pubs), raw_counts.json[.gz] (rows), submission.json; the run
records (one JSON per cell) hold the inputs, the settings and the measured qubits.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from collections import defaultdict

import numpy as np

from psf_os import RULES

ECHO_NAME = re.compile(r"kecho_n(\d+)_D(\d+)_hz([0-9.]+)_h([0-9.]+)-(.+)$")
INPUTS = ("zero", "one", "neel", "antineel")


def input_bits(name, n):
    """The four inputs of the echo and the forward tests (TOQB toqb/kising_charge.py)."""
    if name == "zero":
        return [0] * n
    if name == "one":
        return [1] * n
    if name == "neel":
        return [q % 2 for q in range(n)]
    if name == "antineel":
        return [(q + 1) % 2 for q in range(n)]
    raise ValueError(name)


def load_counts(job):
    for name in ("raw_counts.json", "raw_counts.json.gz"):
        p = os.path.join(job, name)
        if os.path.exists(p):
            return json.load(gzip.open(p) if name.endswith(".gz") else open(p))["rows"]
    raise FileNotFoundError(f"no raw counts in {job}")


def site_signal(counts_by_input, n):
    """Per-site echo signal a_j = mean over the 4 inputs of <-z_in,j Z_j> (1 perfect, 0 random, < 0 inverted)."""
    a = np.zeros(n)
    for i in INPUTS:
        zin = 1 - 2 * np.array(input_bits(i, n))
        c = counts_by_input[i]
        N = float(sum(c.values()))
        m = np.zeros(n)
        for key, k in c.items():
            b = key.replace(" ", "")
            m += k * np.array([1 - 2 * int(b[n - 1 - j]) for j in range(n)])
        a += -zin * m / N / 4
    return a


def job_profiles(job, runs, D=24, arm="fold_rule"):
    """{n: (physical qubits in ring order, a_j averaged over the hz present)} for the job's echo cells at depth D,
    with the job id, backend and submission time."""
    man = json.load(open(os.path.join(job, "manifest.json")))
    rows = load_counts(job)
    sub = json.load(open(os.path.join(job, "submission.json")))
    per = defaultdict(list)
    phys_of = {}
    for cid, c in enumerate(man["cells"]):
        m = ECHO_NAME.match(c["name"])
        if not m or int(m.group(2)) != D or m.group(5) != arm:
            continue
        rec = json.load(open(os.path.join(runs, c["name"] + ".json")))
        if not rec.get("valid"):
            continue
        n = rec["task"]["n"]
        cc = {}
        for k, p in enumerate(man["pubs"]):
            if p["cell"] == cid:
                pt = rec["points"][p["point"]]
                if pt["setting"] == "echo":
                    cc[pt["input"]] = rows[k]["counts"]
        mq = rec["echo"]["measured_qubits"]
        phys = tuple(int(mq[str(j)]) for j in range(n))
        if n in phys_of and phys_of[n] != phys:
            raise ValueError(f"{job}: two rings for n = {n}")
        phys_of[n] = phys
        per[n].append(site_signal(cc, n))
    when = sub.get("submitted", "")
    return {n: (phys_of[n], np.mean(per[n], axis=0)) for n in per}, sub.get("job_id"), sub.get("backend"), when


def dead_runs(a, thr=RULES["dead_threshold"], min_len=RULES["min_run"]):
    """Contiguous runs (cyclic) of sites with a < thr, at least min_len long: [(start, length)]."""
    n = len(a)
    dead = [x < thr for x in a]
    if all(dead):
        return [(0, n)]
    s0 = next(i for i in range(n) if not dead[i])  # start after a live site so no run is cut at index 0
    out, i = [], 0
    while i < n:
        j = (s0 + i) % n
        if dead[j]:
            k = 0
            while k < n and dead[(j + k) % n]:
                k += 1
            if k >= min_len:
                out.append((j, k))
            i += k
        else:
            i += 1
    return out


def weak_runs(a, thr=RULES["dead_threshold"], rel=RULES["weak_relative"], min_len=RULES["min_run"]):
    """Runs of at least min_len consecutive sites (cyclic) below max(thr, rel x median(a)), and that line."""
    a = np.asarray(a, float)
    line = max(thr, rel * float(np.median(a)))
    return dead_runs(a, line, min_len), line


def centre_path(phys, start, length, width=RULES["centre"]):
    """The `width` qubits at the centre of a run, in ring order (fewer if the run is shorter)."""
    w = min(width, length)
    first = start + (length - w) // 2
    return tuple(phys[(first + t) % len(phys)] for t in range(w))


def contains_path(cycle, path):
    """True if `path` appears as consecutive qubits of the cycle, in either direction."""
    m, w = len(cycle), len(path)
    if w > m:
        return False
    for seq in (path, tuple(reversed(path))):
        for s in range(m):
            if all(cycle[(s + t) % m] == seq[t] for t in range(w)):
                return True
    return False


def usable(a, thr=RULES["dead_threshold"], min_len=RULES["min_run"]):
    """Ring sites outside every dead run."""
    n = len(a)
    dead = set()
    for st, ln in dead_runs(np.asarray(a), thr, min_len):
        dead.update((st + t) % n for t in range(ln))
    return [j for j in range(n) if j not in dead]


def observe(device_job_runs, thr=RULES["dead_threshold"], min_len=RULES["min_run"], width=RULES["centre"]):
    """One observation per (job, n): the ring, its dead runs and their centre paths (black candidates)."""
    obs = []
    for device, job, runs, label in device_job_runs:
        prof, job_id, backend, when = job_profiles(job, runs)
        if backend and backend != device:
            raise ValueError(f"{job}: submitted to {backend}, not {device}")
        for n, (phys, a) in sorted(prof.items()):
            runs_ = dead_runs(a, thr, min_len)
            obs.append(dict(device=device, job=job_id, label=label, submitted=when, n=n, ring=list(phys),
                            median=float(np.median(a)),
                            dead=[dict(start=s, length=L, qubits=[phys[(s + t) % n] for t in range(L)],
                                       centre=list(centre_path(phys, s, L, width)),
                                       mean=float(np.mean([a[(s + t) % n] for t in range(L)]))) for s, L in runs_]))
    return obs


def weak_scan(device, job, runs, label, thr=RULES["dead_threshold"], rel=RULES["weak_relative"],
              min_len=RULES["min_run"], width=RULES["centre"]):
    """One entry per ring size: the weak line and the weak runs with their centre paths (gray, same day)."""
    prof, job_id, backend, when = job_profiles(job, runs)
    if backend and backend != device:
        raise ValueError(f"{job}: submitted to {backend}, not {device}")
    out = []
    for n, (phys, a) in sorted(prof.items()):
        rs, line = weak_runs(a, thr, rel, min_len)
        out.append(dict(device=device, job=job_id, label=label, submitted=when, n=n, ring=list(phys),
                        median=float(np.median(a)), line=line,
                        weak=[dict(start=s, length=L, qubits=[phys[(s + t) % n] for t in range(L)],
                                   centre=list(centre_path(phys, s, L, width)),
                                   mean=float(np.mean([a[(s + t) % n] for t in range(L)]))) for s, L in rs]))
    return out
