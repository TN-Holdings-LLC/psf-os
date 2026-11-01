"""L2: the defect registry.

The file keeps TOQB's list format, so that a registry made here and a TOQB list are the same file (test T1):

    {"params": {"thr": 0.12, "min_len": 3, "centre": 4},
     "observations": [...],                 # scan.observe rows, append-only, oldest first
     "base": {"file": ..., "sha256": ...} or null}

and, only when present, two keys of the OS layer:

    "gray":    [{"device", "day", "job", "n", "line", "path", "qubits", "mean"}]   # weak runs, valid on `day` only
    "repairs": [{"device", "path", "flip", "helped_on": [days]}]                   # applied after 2 days

Black entries are derived, never stored: `entries` groups the dead runs of the observations (ported from TOQB
toqb/defect_list.py at 8d44f05: LISTED after `min_repeats` jobs, EXPIRED after `expiry_days`, CANDIDATE otherwise;
a healthy ring through the path is a contradiction, and the entry stays).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re

from psf_os import RULES
from psf_os import scan as S


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _time(s):
    m = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) CET\)", s or "")
    if m:
        return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return dt.datetime.fromisoformat(s[:19]) if s else None


def entries(obs, min_repeats=RULES["min_repeats"], expiry_days=RULES["expiry_days"], now=None,
            overlap=RULES["overlap"]):
    """Group dead runs per device: two sightings belong to one entry when their centre paths share at least `overlap`
    qubits. The entry's path is the part of the centre common to all its sightings, in ring order (the first
    sighting's centre if that is shorter than `overlap`). LISTED: seen in >= min_repeats jobs and last seen within
    expiry_days of `now`; CANDIDATE: fewer jobs; EXPIRED: older. A ring observed healthy that contains the entry's path
    is reported as a contradiction (the entry stays)."""
    groups = []
    for o in obs:
        for d in o["dead"]:
            c = list(d["centre"])
            sight = dict(job=o["job"], label=o["label"], n=o["n"], submitted=o["submitted"], mean=d["mean"],
                         length=d["length"], centre=c, ring=o["ring"])
            for g in groups:
                if g["device"] == o["device"] and any(len(set(c) & set(s["centre"])) >= overlap for s in g["seen"]):
                    g["seen"].append(sight)
                    g["qubits"].update(d["qubits"])
                    break
            else:
                groups.append(dict(device=o["device"], seen=[sight], qubits=set(d["qubits"])))
    t_now = now or max((_time(o["submitted"]) for o in obs if o["submitted"]), default=None)
    out = []
    for g in groups:
        first = g["seen"][0]["centre"]
        common = [q for q in first if all(q in s["centre"] for s in g["seen"])]
        path = common if len(common) >= overlap else first
        device = g["device"]
        jobs = {s["job"] for s in g["seen"]}
        last = max(_time(s["submitted"]) for s in g["seen"])
        contra = [dict(job=o["job"], n=o["n"]) for o in obs if o["device"] == device and not o["dead"]
                  and S.contains_path(o["ring"], path)]
        if t_now is not None and (t_now - last).total_seconds() > expiry_days * 86400:
            status = "EXPIRED"
        elif len(jobs) >= min_repeats:
            status = "LISTED"
        else:
            status = "CANDIDATE"
        out.append(dict(device=device, centre=list(path), status=status, jobs=sorted(j for j in jobs if j),
                        sightings=len(g["seen"]), last_seen=last.isoformat(sep=" "), qubits=sorted(g["qubits"]),
                        contradictions=contra,
                        seen=[{k: v for k, v in s.items() if k != "ring"} for s in g["seen"]]))
    out.sort(key=lambda e: (e["device"], e["status"], e["centre"]))
    return out


# ------------------------------------------------------------------------------------------------ the file

def params_dict():
    return dict(thr=RULES["dead_threshold"], min_len=RULES["min_run"], centre=RULES["centre"])


def empty():
    return dict(params=params_dict(), observations=[], base=None)


def load(path):
    reg = json.load(open(path))
    if reg["params"] != params_dict():
        raise ValueError(f"{path} was made with other parameters {reg['params']}")
    return reg


def ingest(obs_specs, base=None):
    """A new registry: the base file's observations, then those of the scan jobs `obs_specs` = [(device, job, runs,
    label)]. A job already in the base is refused. Same result as `toqb.defect_list ingest --base`."""
    obs = S.observe(obs_specs)
    prev = []
    reg_base = None
    extra = {}
    if base:
        b = json.load(open(base))
        if b["params"] != params_dict():
            raise ValueError(f"{base} was made with other parameters {b['params']}")
        prev = b["observations"]
        seen = {(o["device"], o["job"], o["n"]) for o in prev}
        dup = [o for o in obs if (o["device"], o["job"], o["n"]) in seen]
        if dup:
            raise ValueError(f"job {dup[0]['job']} n {dup[0]['n']} is already in {base}")
        reg_base = dict(file=os.path.basename(base), sha256=sha256_file(base))
        extra = {k: b[k] for k in ("gray", "repairs") if b.get(k)}
    reg = dict(params=params_dict(), observations=prev + obs, base=reg_base)
    reg.update(extra)
    return reg, obs


def add_gray(reg, device, job, runs, label):
    """Weak runs of a scan job, valid on the scan's day only (not carried over)."""
    for s in S.weak_scan(device, job, runs, label):
        day = _time(s["submitted"]).date().isoformat()
        for w in s["weak"]:
            reg.setdefault("gray", []).append(dict(device=device, day=day, job=s["job"], n=s["n"], line=s["line"],
                                                   path=w["centre"], qubits=w["qubits"], mean=w["mean"]))
    return reg


def add_repair(reg, device, path, flip, day):
    for r in reg.setdefault("repairs", []):
        if r["device"] == device and r["path"] == list(path) and r["flip"] == list(flip):
            if day not in r["helped_on"]:
                r["helped_on"].append(day)
            return r
    r = dict(device=device, path=list(path), flip=list(flip), helped_on=[day])
    reg["repairs"].append(r)
    return r


# ------------------------------------------------------------------------------------------------ what applies now

def avoid_paths(reg, device, now):
    """The paths the allocator avoids on `now`'s day: LISTED black paths, and gray paths seen that day."""
    out = []
    for e in entries(reg["observations"], now=now):
        if e["device"] == device and e["status"] == "LISTED":
            out.append(dict(path=e["centre"], kind="black", jobs=len(e["jobs"]), last_seen=e["last_seen"]))
    day = now.date().isoformat()
    for g in reg.get("gray", []):
        if g["device"] == device and g["day"] == day:
            if not any(len(set(g["path"]) & set(o["path"])) >= RULES["overlap"] for o in out):
                out.append(dict(path=g["path"], kind="gray", jobs=1, last_seen=g["day"]))
    return out


def repairs_ready(reg, device, min_days=2):
    return [r for r in reg.get("repairs", []) if r["device"] == device and len(set(r["helped_on"])) >= min_days]


def last_scan(reg, device):
    ts = [_time(o["submitted"]) for o in reg["observations"] if o["device"] == device and o["submitted"]]
    return max(ts) if ts else None


def is_stale(reg, device, now):
    """Stale when the device has no scan on `now`'s day (the OS plan: scan only if the registry is old)."""
    t = last_scan(reg, device)
    return t is None or t.date() != now.date()


def status_lines(reg, device=None, now=None):
    """Like `toqb.defect_list status`."""
    out = []
    for e in entries(reg["observations"], now=now):
        if device and e["device"] != device:
            continue
        out.append(f"{e['device']} centre {e['centre']}: {e['status']} ({e['sightings']} sightings in "
                   f"{len(e['jobs'])} jobs, last {e['last_seen']}; contradictions {len(e['contradictions'])})")
    return out


def save(reg, path):
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: a registry file is never overwritten (write the next day's file)")
    json.dump(reg, open(path, "w"), indent=1)
