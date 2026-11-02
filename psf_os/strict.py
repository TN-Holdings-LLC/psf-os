"""L2 (strict variant): the safe-ring list, where one sighting of a weak run lists its centre path at once, and the
depth verdicts against the n = 20 control.

Ported from TOQB toqb/safe_loop.py at 8d44f05 (paths_of, add_paths, depth_verdicts; weak_runs and scan_job are in
psf_os.scan). Unchanged except for imports.
"""
from __future__ import annotations

import math


def paths_of(lst, device):
    """The listed paths of a device, without duplicates (a path and its reverse are the same)."""
    seen, out = set(), []
    for p in lst["paths"]:
        if p["device"] != device:
            continue
        key = min(tuple(p["path"]), tuple(reversed(p["path"])))
        if key not in seen:
            seen.add(key)
            out.append(tuple(p["path"]))
    return out


def add_paths(lst, scans, overlap=2):
    """A new list: the old paths plus the centre path of every weak run in the scans (one sighting is enough). A centre
    sharing at least `overlap` qubits with a path already listed on the device is the same defect seen again (a run's
    ends move by a site or two) and is not added, so one defect masks one coupler."""
    new = dict(params=lst["params"], paths=list(lst["paths"]), scans=list(lst.get("scans", [])) + scans)
    for s in scans:
        for w in s["weak"]:
            if not any(p["device"] == s["device"] and len(set(p["path"]) & set(w["centre"])) >= overlap
                       for p in new["paths"]):
                new["paths"].append(dict(device=s["device"], path=w["centre"], source=f"{s['label']} n {s['n']}",
                                         mean=w["mean"], length=w["length"]))
    return new


def depth_verdicts(echo_score, hz="0.0", control=20):
    """Per n: L*, SE and REACHED / SHORTER against the control's L* (2 combined SE), from kising_echo's walls."""
    w = echo_score["walls"][hz]
    f = {x["n"]: (x["L_star"], x["se"]) for x in w["fits"]}
    if control not in f or f[control][0] is None:
        return {}
    l0, s0 = f[control]
    out = {}
    for n, (L, se) in sorted(f.items()):
        if n == control:
            continue
        ok = L is not None and L >= l0 - 2 * math.sqrt(se ** 2 + s0 ** 2)
        out[n] = dict(L_star=L, se=se, L_star_control=l0, se_control=s0, depth="REACHED" if ok else "SHORTER")
    return out
