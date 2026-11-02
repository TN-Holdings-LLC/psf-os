"""L0 on an IBM device: fetch the device record, submit a job folder, collect its counts.

The same rules as TOQB's s2rc_device (owner, 2026-11-02, the step after M6): gate-level circuits through SamplerV2 with
dynamical decoupling and twirling off; a job is only ever tried on the device it was placed for (psf-os jobs "stay");
at submission the job is checked against the elements that device marks as failed now, and nothing is sent if any
circuit touches one. No quality number is read or stored: the device record keeps the coupling graph and the failed
elements only (device.from_target), and the submission check reads only the failed mark.

The saved IBM account is used (QiskitRuntimeService()); psf-os never reads, writes or asks for a key. Every function
takes the service as an argument, so that tests can pass a stand-in.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from zoneinfo import ZoneInfo

from psf_os import device as Dv
from psf_os.registry import sha256_file

OPTIONS = {"dynamical_decoupling.enable": False, "twirling.enable_gates": False, "twirling.enable_measure": False}
OPTIONS_TEXT = "dynamical decoupling off, twirling off (gates and measure); SamplerV2 applies no other error mitigation"


def _stamp():
    t = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    return f"{t.isoformat()} ({t.astimezone(ZoneInfo('Europe/Zurich')).strftime('%Y-%m-%d %H:%M:%S')} CET)"


def service():
    """The saved IBM account. Raises SystemExit with a plain message if there is none."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        return QiskitRuntimeService()
    except Exception as exc:  # noqa: BLE001 - any failure to load the saved account
        raise SystemExit("no usable saved IBM account (" + type(exc).__name__ + "); save it once in a terminal with "
                         "QiskitRuntimeService.save_account(...), never in a file or a chat")


def calibration_time(backend):
    """The device's calibration timestamp (a time, not a quality number), or None."""
    try:
        return str(backend.properties().last_update_date)
    except Exception:  # noqa: BLE001
        return None


def fetch(svc, backend_name, out):
    """Write OUT/device.json for an IBM device: coupling graph and failed elements, nothing else."""
    backend = svc.backend(backend_name)
    dev = Dv.from_target(backend.target, backend.name, fake=False,
                         source=dict(fetched=_stamp(), calibration_time=calibration_time(backend)))
    dev.save(out)
    return dev


def problems(circuits, target):
    """Circuit elements that the device does not support or marks as failed now (the failed mark only)."""
    _, failed_c, failed_q = Dv.failed_elements(target)
    fc = {tuple(sorted(e)) for e in failed_c}
    fq = set(failed_q)
    ops = set(target.operation_names)
    keys = {}  # the qubit tuples each instruction is offered on (keys only; no property is read)

    def supported(name, qa):
        if name not in ops:
            return False
        if name not in keys:
            keys[name] = set(iter(target[name]))
        return None in keys[name] or qa in keys[name]
    out = []
    for i, qc in enumerate(circuits):
        for inst in qc.data:
            name = inst.operation.name
            if name == "barrier":
                continue
            qa = tuple(qc.find_bit(q).index for q in inst.qubits)
            if not supported(name, qa):
                out.append(f"circuit {i}: {name}{qa} not supported")
            elif len(qa) == 2 and tuple(sorted(qa)) in fc:
                out.append(f"circuit {i}: {name}{qa} on a failed coupler")
            elif len(qa) == 1 and qa[0] in fq:
                out.append(f"circuit {i}: {name}{qa} on a failed qubit")
    return out


def _read_job(job_dir):
    man = json.load(open(os.path.join(job_dir, "manifest.json")))
    if sha256_file(os.path.join(job_dir, "circuits.qpy")) != man["circuits_sha256"]:
        raise SystemExit("STOP: circuits.qpy differs from the manifest")
    return man


def submit(svc, job_dir, sampler_cls=None):
    """Send the job folder to the device it was placed for, or refuse (submission_refused.json). Returns the job id.
    `sampler_cls` is for tests only (default: qiskit_ibm_runtime.SamplerV2)."""
    from qiskit import qpy
    if sampler_cls is None:
        from qiskit_ibm_runtime import SamplerV2 as sampler_cls
    man = _read_job(job_dir)
    sub_file = os.path.join(job_dir, "submission.json")
    if os.path.exists(sub_file):
        raise SystemExit(f"STOP: {sub_file} exists: this job was already submitted")
    if man["fetch"].get("fake") and not getattr(svc, "allows_fake", False):
        raise SystemExit("STOP: the job was placed on a fake device; run it with --local")
    if man.get("stay") is not True:
        raise SystemExit("STOP: the manifest does not say stay (not a psf-os job folder)")
    with open(os.path.join(job_dir, "circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    if len(circs) != len(man["pubs"]):
        raise SystemExit("STOP: the number of circuits differs from the manifest")
    name = man["fetch"]["backend"]
    backend = svc.backend(name)
    if backend.name != name:
        raise SystemExit(f"STOP: asked for {name}, got {backend.name}")
    cal = calibration_time(backend)
    why = problems(circs, backend.target)
    if why:
        json.dump(dict(written=_stamp(), backend=name, calibration_at_submit=cal, problems=why[:50], stay=True),
                  open(os.path.join(job_dir, "submission_refused.json"), "w"), indent=1)
        raise SystemExit(f"STOP: {name} does not take the job now ({len(why)} problems; submission_refused.json); "
                         "plan it again from a new fetch")
    sampler = sampler_cls(mode=backend)
    for k, v in OPTIONS.items():
        obj = sampler.options
        for part in k.split(".")[:-1]:
            obj = getattr(obj, part)
        setattr(obj, k.split(".")[-1], v)
    t = _stamp()
    job = sampler.run([(c, None, man["shots"]) for c in circs])
    json.dump(dict(job_id=job.job_id(), submitted=t, local=False, backend=name, stay=True, calibration_at_submit=cal,
                   options=OPTIONS_TEXT, sampler_options=OPTIONS, psfos="psf_os.ibm"),
              open(sub_file, "w"), indent=1)
    return job.job_id()


def collect(svc, job_dir, wait_s=0, poll_s=30):
    """Write raw_counts.json when the job is done (TOQB's format, so either side can score it). Returns the status."""
    sub = json.load(open(os.path.join(job_dir, "submission.json")))
    raw_file = os.path.join(job_dir, "raw_counts.json")
    if os.path.exists(raw_file):
        raise SystemExit(f"STOP: {raw_file} exists")
    job = svc.job(sub["job_id"])
    t_end = time.time() + wait_s
    while True:
        st = str(job.status())
        if any(x in st.upper() for x in ("DONE", "ERROR", "CANCEL")) or time.time() >= t_end:
            break
        time.sleep(poll_s)
    if "DONE" not in st.upper():
        return st
    res = job.result()
    rows = [dict(pub=i, counts=dict(r.join_data().get_counts())) for i, r in enumerate(res)]
    try:
        metrics = job.metrics()
    except Exception as exc:  # noqa: BLE001
        metrics = dict(error=type(exc).__name__)
    json.dump(dict(fetched=_stamp(), job_id=sub["job_id"], status=st, metrics=metrics, rows=rows),
              open(raw_file, "w"), default=str)
    return st
