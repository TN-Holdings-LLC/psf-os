"""The circuit as sent, checked and described (0.3.3rc1, candidate; the owner's decision of 2026-11-23).

The compiler may move, merge or cancel gates across a wait, so what is written is not always what waits (the
workplace VS on the compiled IPE, 2026-11-20). psf-os 0.3.0 and 0.3.1 decide the parking on the circuit as written
and check that the compiled CZ count and the noiseless output do not change; they do not look at the compiled circuit's
states. This module does, noiselessly and from the circuit alone (no noise model, no calibration number):

  waits(cc)                    every wait of a compiled circuit (a mid-circuit read with the conditionals that follow
                               it, or a run of conditionals on bits read earlier), and per waiting qubit its
                               probability of |1> over the branches (at the wait's start and the largest during it) and
                               whether it is entangled with other qubits in any branch
  check_parked(cc, plan)       refuses (ValueError) when a qubit that psf-os parked is not parked in the compiled
                               circuit: P(|1>) above 0 during the wait for a qubit parked because it was in |1> on
                               every branch, 0.5 or more for a qubit parked because it was more likely in |1>
  summary(c)                   CZ, depth and the SHA-256 of the gate sequence (conditional blocks opened, RZ in
                               (-pi, pi]: psf_os.gatehash), for the manifest and for verify across versions
  schedule(c, timing)          the circuit's length and each conditional's wait in dt from the device record's
                               durations (as soon as possible, per qubit); the decision latency of a conditional is
                               not in the record and is not counted, so a conditional's wait is a lower bound

A wait is located in a compiled circuit by its anchors, which compilation keeps: each measurement by its classical bit
and the how-many-th write to that bit, each conditional by its test (bits and value) and the how-many-th conditional
with that test. `cc` is the compiled circuit restricted to the item's qubits in logical order (feedforward.restrict).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

EPS = 1e-9


# ------------------------------------------------------------------------------------------------ anchors

def _cond_key(qc, op):
    tgt, val = op.condition
    bits = ([qc.find_bit(x).index for x in tgt] if type(tgt).__name__ == "ClassicalRegister"
            else [qc.find_bit(tgt).index])
    return tuple(bits), int(val)


def anchors(qc):
    """{instruction index: anchor} for every top-level measurement ("m", clbit, k) and conditional ("c", bits, value,
    k), k counting from 0 per clbit / per test."""
    seen, out = {}, {}
    for i, ins in enumerate(qc.data):
        name = ins.operation.name
        if name == "measure":
            key = ("m", qc.find_bit(ins.clbits[0]).index)
        elif name == "if_else":
            key = ("c",) + _cond_key(qc, ins.operation)
        else:
            continue
        k = seen.get(key, 0)
        seen[key] = k + 1
        out[i] = key + (k,)
    return out


def _skip(name):
    return name in ("barrier", "delay")


def wait_spans(qc):
    """The waits of qc in program order: dict(kind "read" | "conditional", start, end (instruction indices, both part
    of the wait), involved (qubit indices measured or inside the conditionals), keys (anchors)). A read is a group of
    measurements (barriers and delays between them allowed) followed by more than measurements, with the conditionals
    right after it; a run of conditionals not right after a read is a wait of its own."""
    data = list(qc.data)
    anc = anchors(qc)
    n = len(data)

    def later_ops(j):
        return any(data[x].operation.name not in ("measure", "barrier") for x in range(j, n))

    out = []
    i = 0
    while i < n:
        name = data[i].operation.name
        if name not in ("measure", "if_else"):
            i += 1
            continue
        involved, keys = set(), []
        start = i
        kind = "read" if name == "measure" else "conditional"
        j = i
        last = i
        while j < n and (data[j].operation.name == "measure" or _skip(data[j].operation.name)):
            if data[j].operation.name == "measure":
                involved.add(qc.find_bit(data[j].qubits[0]).index)
                keys.append(anc[j])
                last = j
            j += 1
        if kind == "read" and not later_ops(last + 1):
            break                                    # the final measurements are not a wait
        k = last + 1 if kind == "read" else i
        while k < n and (data[k].operation.name == "if_else" or _skip(data[k].operation.name)):
            if data[k].operation.name == "if_else":
                involved |= {qc.find_bit(q).index for q in data[k].qubits}
                keys.append(anc[k])
                last = k
            k += 1
        out.append(dict(kind=kind, start=start, end=last, involved=sorted(involved), keys=keys))
        i = last + 1
    return out


# ------------------------------------------------------------------------------------------------ states

def _states(qc, stops):
    from psf_os.footprint import _simulate
    branches, sim = _simulate(qc, stops)
    at = dict(sim["at"])
    return at, branches, sim


def _qubit_view(brs, k, n):
    """(P(|1>) over the branches with their weights, entangled in any branch) of active qubit k."""
    from psf_os.footprint import _reduced
    p1, ent = 0.0, False
    for p, sv, _ in brs:
        if p < 1e-15:
            continue
        rho = _reduced(sv.data, k, n)
        p1 += p * float(rho[1, 1].real)
        if not ent and float(np.real(np.trace(rho @ rho))) < 1 - 1e-7:
            ent = True
    return p1, ent


def waits(cc):
    """Every wait of the compiled circuit cc (logical qubit order) with, per waiting active qubit, p1 at the wait's
    start, p1_max during it and entangled during it. dict(counted=True, waits=[...]) or dict(counted=False, why) above
    the footprint's bounds (psf_os.footprint.MAX_QUBITS, MAX_WORK)."""
    spans = wait_spans(cc)
    if not spans:
        return dict(counted=True, waits=[])
    stops = sorted({x for w in spans for x in range(w["start"], w["end"] + 2)})
    try:
        at, final, sim = _states(cc, stops)
    except ValueError as e:
        return dict(counted=False, why=str(e))
    act, pos = sim["act"], sim["pos"]
    n = max(len(act), 1)
    out = []
    for w in spans:
        qs = {}
        for q in act:
            if q in w["involved"]:
                continue
            k = pos[q]
            p_start, ent = _qubit_view(at.get(w["start"], final), k, n)
            p_max = p_start
            for x in range(w["start"] + 1, w["end"] + 2):
                p, e = _qubit_view(at.get(x, final), k, n)
                p_max, ent = max(p_max, p), ent or e
            qs[q] = (p_start, p_max, ent)
        out.append(dict(kind=w["kind"], at=list(w["keys"][0]), qubits=qs))
    return dict(counted=True, waits=out)


def waits_record(cc, physical):
    """waits(cc) for a manifest: qubits as physical qubits (physical[k] for logical k), probabilities to 1e-4."""
    r = waits(cc)
    if not r["counted"]:
        return r
    rows = []
    for w in r["waits"]:
        rows.append(dict(kind=w["kind"], at=w["at"],
                         qubits=[dict(q=physical[q], p1=round(a, 4), p1_max=round(b, 4), entangled=bool(e))
                                 for q, (a, b, e) in sorted(w["qubits"].items())]))
    return dict(counted=True, waits=rows)


# ------------------------------------------------------------------------------------------------ the parking check

def plan_anchors(f, windows, kind):
    """The parking plan of a fenced logical circuit f as anchored waits: windows from footprint._windows with their
    parked qubits under "qubits" (logical); kind "always" or "likely"."""
    anc = anchors(f)
    out = []
    for w in windows:
        if not w.get("qubits"):
            continue
        keys = [anc[i] for i in range(w["start"] + 1, w["end"] + 1) if i in anc]
        out.append(dict(kind=kind, qubits=list(w["qubits"]), keys=keys))
    return out


def check_parked(cc, plan, tag=""):
    """Refuses (ValueError) when a parked qubit is not parked in the compiled circuit cc during its wait: from the
    first to the last anchor of the wait (both included, and right after the last), P(|1>) over the branches must be
    0 (to 1e-9) for an "always" park and below 0.5 for a "likely" park. Returns dict(waits, qubits) checked; an empty
    plan is not simulated."""
    if not plan:
        return dict(waits=0, qubits=0)
    anc = {v: i for i, v in anchors(cc).items()}
    spans = []
    for w in plan:
        idx = []
        for key in w["keys"]:
            key = tuple(key) if not isinstance(key, tuple) else key
            if key not in anc:
                raise ValueError(f"{tag}: a parked wait ({key}) is not found in the compiled circuit: not built")
            idx.append(anc[key])
        spans.append((min(idx), max(idx), w))
    stops = sorted({x for s, e, _ in spans for x in range(s, e + 2)})
    at, final, sim = _states(cc, stops)
    act, pos = sim["act"], sim["pos"]
    n = max(len(act), 1)
    count = 0
    for s, e, w in spans:
        for q in w["qubits"]:
            if q not in pos:
                raise ValueError(f"{tag}: parked qubit {q} is idle in the compiled circuit: not built")
            for x in range(s, e + 2):
                p1, _ = _qubit_view(at.get(x, final), pos[q], n)
                if (w["kind"] == "always" and p1 > EPS) or (w["kind"] == "likely" and p1 >= 0.5 - EPS):
                    raise ValueError(f"{tag}: qubit {q}, parked ({w['kind']}) at the wait {list(w['keys'][0])}, has "
                                     f"P(|1>) {p1:.4f} in the compiled circuit during the wait: the compiler did not "
                                     "keep the parking; not built")
            count += 1
    return dict(waits=len(spans), qubits=count)


# ------------------------------------------------------------------------------------------------ summaries

def summary(c):
    """dict(cz, depth, gates_sha256) of a circuit as sent: the SHA-256 over psf_os.gatehash's open gate list
    (conditional blocks opened, RZ in (-pi, pi], parameters to 1e-10), comparable across qiskit versions."""
    from psf_os.gatehash import _gates_open
    from psf_os.feedforward import cz_count
    doc = _gates_open(c)
    return dict(cz=cz_count(c), depth=c.depth(),
                gates_sha256=hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode())
                .hexdigest())


def summary_difference(a, b, names=None):
    """One line on two lists of summaries (the recorded job's and the rebuilt one's)."""
    if len(a) != len(b):
        return f"summary: {len(a)} circuits recorded, {len(b)} rebuilt"
    bad = [k for k, (x, y) in enumerate(zip(a, b)) if x != y]
    if not bad:
        return f"summary: the same for all {len(a)} circuits"
    k = bad[0]
    x, y = a[k], b[k]
    what = ", ".join(f"{f} {x[f]} -> {y[f]}" for f in ("cz", "depth") if x[f] != y[f]) or "gates only"
    nm = f" ({names[k]})" if names and k < len(names) else ""
    return f"summary: {len(bad)} of {len(a)} circuits differ; the first, circuit {k}{nm}: {what}"


# ------------------------------------------------------------------------------------------------ lengths of time

def _dur(timing, name, qs):
    d = (timing.get("durations") or {}).get(name)
    if d is None:
        return None
    return d.get(",".join(str(q) for q in qs))


def schedule(c, timing):
    """(length in dt, [per conditional: dict(at, measure_dt, body_dt, wait_dt_lower)]) of a compiled device circuit c
    by the durations of the device record (as soon as possible per qubit; a barrier aligns its qubits; rz takes no
    time; a delay its length). A conditional starts when its qubits are free and its bits are measured; its wait is
    the measurement that wrote its bit plus its longest block: a lower bound, the decision latency being no part of
    the record. None when a duration is missing (the reason under "missing")."""
    if not timing or not timing.get("durations"):
        return None
    t = {}
    ready, meas_len = {}, {}
    conds = []
    missing = set()
    anc = anchors(c)

    def block_len(blk, qmap):
        tb = {}
        for x in blk.data:
            nm = x.operation.name
            qs = [qmap[blk.find_bit(q).index] for q in x.qubits]
            if nm == "barrier":
                m = max((tb.get(q, 0) for q in qs), default=0)
                for q in qs:
                    tb[q] = m
                continue
            if nm == "rz":
                continue
            d = int(x.operation.duration) if nm == "delay" else _dur(timing, nm, qs)
            if d is None:
                missing.add(nm)
                d = 0
            st = max((tb.get(q, 0) for q in qs), default=0)
            for q in qs:
                tb[q] = st + d
        return max(tb.values(), default=0)

    for i, x in enumerate(c.data):
        nm = x.operation.name
        qs = [c.find_bit(q).index for q in x.qubits]
        if nm == "barrier":
            m = max((t.get(q, 0) for q in qs), default=0)
            for q in qs:
                t[q] = m
            continue
        if nm == "rz":
            continue
        if nm == "if_else":
            bits, _val = _cond_key(c, x.operation)
            st = max([t.get(q, 0) for q in qs] + [ready.get(b, 0) for b in bits])
            body = max((block_len(b, {j: qs[j] for j in range(len(qs))}) for b in x.operation.blocks
                        if b is not None), default=0)
            m = max((meas_len.get(b, 0) for b in bits), default=0)
            conds.append(dict(at=list(anc[i]), measure_dt=m, body_dt=body, wait_dt_lower=m + body))
            for q in qs:
                t[q] = st + body
            continue
        d = int(x.operation.duration) if nm == "delay" else _dur(timing, nm, qs)
        if d is None:
            missing.add(nm)
            d = 0
        st = max((t.get(q, 0) for q in qs), default=0)
        for q in qs:
            t[q] = st + d
        if nm == "measure":
            b = c.find_bit(x.clbits[0]).index
            ready[b] = st + d
            meas_len[b] = d
    if missing:
        return dict(missing=sorted(missing))
    return dict(length_dt=max(t.values(), default=0), conditionals=conds)


def lengths(c, timing, dt):
    """schedule(c, timing) for a manifest: the length in dt and µs and the conditionals' waits (lower bounds) as
    [measure_dt, body_dt] pairs; None without durations."""
    s = schedule(c, timing)
    if s is None:
        return None
    if "missing" in s:
        return dict(missing=s["missing"])
    us = (lambda k: round(k * dt * 1e6, 4)) if dt else (lambda k: None)
    out = dict(length_dt=s["length_dt"], length_us=us(s["length_dt"]))
    if s["conditionals"]:
        w = [x["wait_dt_lower"] for x in s["conditionals"]]
        out["conditionals"] = dict(n=len(w), wait_dt_lower=[[x["measure_dt"], x["body_dt"]] for x in s["conditionals"]],
                                   longest_wait_us_lower=us(max(w)),
                                   note="per conditional [measurement, longest block] in dt: a lower bound (the "
                                        "decision latency is not in the device record)")
    return out
