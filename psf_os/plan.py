"""The chain behind `psf-os plan`: registry stale -> a scan job first; registry fresh -> avoid, place, (repair),
sentinels (none in M5), assemble. Every decision goes into report.json with the category and the entries applied.
"""
from __future__ import annotations

import datetime as dt
import json
import os

from psf_os import allocate as A
from psf_os import job as J
from psf_os import registry as R
from psf_os import repair, sentinel
from psf_os.compile import RuleCompiler
from psf_os.scan import contains_path

SCAN_SIZES = (20, 40, 72)
SCAN_JOB = dict(kind="echo", D=24, hz=0.0, h=0.6, shots=1000)


def _report(out, rep):
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    return rep


def scan_job(dev, out, sizes=SCAN_SIZES):
    """The L1 scan: the rule's standard rings (no mask), the echo at D = 24, one record per size that places."""
    runs = os.path.join(out, "scan", "runs")
    built, refused = [], {}
    rc = RuleCompiler(dev)
    for n in sizes:
        try:
            J.build(dev, dict(SCAN_JOB, n=n), runs, compiler=rc)
            built.append(n)
        except Exception as e:  # noqa: BLE001  (a size that does not place is reported, not replaced)
            refused[n] = f"{type(e).__name__}: {e}"
    if not built:
        raise RuntimeError(f"the scan placed no ring: {refused}")
    man = J.assemble(dev, runs, os.path.join(out, "scan", "job"))
    return dict(sizes=built, refused=refused, circuits=len(man["pubs"]), qpu_seconds_model=man["qpu_seconds_model"],
                job=os.path.join(out, "scan", "job"), runs=runs)


def plan(dev, logical, reg, out, now, standard=False, scan_sizes=SCAN_SIZES):
    """Returns the report. `reg` is a registry dict (R.load / R.empty()); `standard` skips the registry entirely."""
    if os.path.exists(os.path.join(out, "report.json")):
        raise FileExistsError(f"{out} already holds a plan")
    rep = dict(written=now.isoformat(sep=" "), device=dev.name, fake=dev.fake, job=logical,
               rules="TOQB defect_list / safe_loop at 8d44f05")
    if not standard and (R.is_stale(reg, dev.name, now)):
        rep["category"] = "scan first"
        rep["why"] = f"no scan of {dev.name} on {now.date().isoformat()} in the registry"
        rep["scan"] = scan_job(dev, out, scan_sizes)
        rep["next"] = ["psf-os submit --job " + rep["scan"]["job"],
                       "psf-os scan-ingest --job ... --runs ... --registry REG --out NEW_REG",
                       "psf-os plan again with NEW_REG"]
        return _report(out, rep)
    avoid = [] if standard else R.avoid_paths(reg, dev.name, now)
    rep["category"] = "standard" if standard else "self-checked"
    rep["avoided"] = avoid
    pa = A.place_avoiding(dev, logical["n"], avoid)
    rep["skipped_outlines"] = pa[3] if pa else None
    dev_use, masked = A.mask(dev, avoid, reason=f"registry at {now.isoformat(sep=' ')}") if avoid else (dev, [])
    rep["masked_couplers"] = [list(e) for e in masked]
    rc = RuleCompiler(dev_use)
    # L4: only for a mirror circuit, only with entries that helped on two days, only if they lie on this ring
    job = dict(logical)
    if job["kind"] == "echo" and not standard:
        from psf_os import circuits as C
        from psf_os.compile import measured_qubits
        probe = rc.compile(C.echo_circuit([0] * job["n"], job["n"], job["D"] // 2))
        mq = measured_qubits(probe)
        ring = [mq[j] for j in range(job["n"])]
        flip, why = repair.plan(job["kind"], ring, R.repairs_ready(reg, dev.name))
        if flip:
            job["flip"] = flip
    else:
        flip, why = repair.plan(job["kind"], [], [])
    rep["repair"] = dict(flip=flip, reason=why)
    runs = os.path.join(out, "job", "runs")
    rec = J.build(dev_use, job, runs, compiler=rc)
    mq = rec["echo"]["measured_qubits"]
    ring = [int(mq[str(j)]) for j in range(job["n"])]
    rep["ring"] = ring
    rep["placement"] = rec["echo"]["placement"]
    rep["accepted"] = rec["echo"]["accepted"]
    rep["avoided_on_ring"] = [a["path"] for a in avoid if contains_path(ring, a["path"])]
    rep["sentinels"] = sentinel.plan(dev_use, ring, reg, now)
    man = J.assemble(dev_use, runs, os.path.join(out, "job", "job"))
    rep["user_job"] = dict(job=os.path.join(out, "job", "job"), runs=runs, circuits=len(man["pubs"]),
                           qpu_seconds_model=man["qpu_seconds_model"])
    rep["next"] = ["psf-os submit --job " + rep["user_job"]["job"], "psf-os result --job ... --runs ..."]
    return _report(out, rep)


def scan_ingest(dev_name, job_dir, runs, base, label):
    """A new registry from `base` (a file, or None) plus this scan job's dead runs (black observations) and weak runs
    (gray, that day only)."""
    reg, obs = R.ingest([(dev_name, job_dir, runs, label)], base=base)
    R.add_gray(reg, dev_name, job_dir, runs, label)
    return reg, obs


def now_cet():
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo("Europe/Zurich")).replace(tzinfo=None, microsecond=0)
