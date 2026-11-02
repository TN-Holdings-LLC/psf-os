"""L1 (diagnostic part): reading the scan echo in X, Y and Z, sign-aligned site vectors, connected pair correlations
and the verdicts used to tell a lost signal from a turned one.

Ported from TOQB toqb/echo_tomo.py at 8d44f05 (functions only; the build and score commands stay in TOQB).
Unchanged except for imports.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from psf_os.registry import sha256_file
from psf_os.scan import INPUTS, dead_runs, input_bits


def with_basis(circ, basis):
    """The circuit with a basis change on every measured qubit just before its measurement (X or Y); Z unchanged."""
    if basis == "Z":
        return circ
    import numpy as _np
    from qiskit import QuantumCircuit
    out = QuantumCircuit(*circ.qregs, *circ.cregs, name=circ.name)
    for i in circ.data:
        if i.operation.name == "measure":
            q = out.qubits[circ.find_bit(i.qubits[0]).index]
            if basis == "X":
                out.rz(_np.pi / 2, q)
                out.sx(q)
                out.rz(_np.pi / 2, q)
            elif basis == "Y":
                out.sx(q)
                out.rz(_np.pi / 2, q)
            else:
                raise ValueError(basis)
        out.append(i.operation, [out.qubits[circ.find_bit(q).index] for q in i.qubits],
                   [out.clbits[circ.find_bit(c).index] for c in i.clbits])
    return out


def _pm(counts, n):
    """Per bit (clbit index) <(-1)^b> and its shot-noise SE from one counts dict."""
    m, N = np.zeros(n), 0.0
    for key, k in counts.items():
        b = key.replace(" ", "")
        m += k * np.array([1 - 2 * int(b[n - 1 - j]) for j in range(n)])
        N += k
    m /= N
    return m, np.sqrt(np.maximum(1 - m ** 2, 0) / N)


def aligned(counts_by_input, n):
    """Sign-aligned component a_j = mean over inputs of (-z_in,j) <P_j> and its SE (P the record's basis)."""
    a, v = np.zeros(n), np.zeros(n)
    for i in INPUTS:
        zin = 1 - 2 * np.array(input_bits(i, n))
        m, se = _pm(counts_by_input[i], n)
        a += -zin * m / 4
        v += se ** 2 / 16
    return a, np.sqrt(v)


def connected(counts_by_input, n):
    """Per ring pair (j, j+1 mod n): mean over the inputs of |<P_j P_j+1> - <P_j><P_j+1>| (P the record's basis)."""
    c = np.zeros(n)
    for i in INPUTS:
        Z, w = [], []
        for key, k in counts_by_input[i].items():
            b = key.replace(" ", "")
            Z.append([1 - 2 * int(b[n - 1 - j]) for j in range(n)])
            w.append(k)
        Z, w = np.array(Z, float), np.array(w, float)
        w /= w.sum()
        m = w @ Z
        for j in range(n):
            jj = (j + 1) % n
            c[j] += abs(float(w @ (Z[:, j] * Z[:, jj]) - m[j] * m[jj])) / len(INPUTS)
    return c


def read_job(job, runs):
    """{(n, D, basis): (phys, a, se, corr)} for every echo record of the job (corr: connected pair correlations)."""
    import gzip
    man = json.load(open(os.path.join(job, "manifest.json")))
    p = os.path.join(job, "raw_counts.json")
    raw = json.load(open(p)) if os.path.exists(p) else json.load(gzip.open(p + ".gz"))
    rows = {r["pub"]: r["counts"] for r in raw["rows"]}
    out = {}
    for cid, c in enumerate(man["cells"]):
        m = NAME.match(c["name"])
        if not m:
            continue
        rf = os.path.join(runs, c["name"] + ".json")
        if sha256_file(rf) != c["run_sha256"]:
            raise SystemExit(f"STOP: {rf} differs from the manifest")
        rec = json.load(open(rf))
        n, D, basis = rec["task"]["n"], rec["echo"]["D"], rec["echo"].get("basis", "Z")
        if (n, D, basis) in out:
            raise SystemExit(f"STOP: two records for n {n} D {D} basis {basis}")
        cc = {}
        for k, pub in enumerate(man["pubs"]):
            if pub["cell"] == cid:
                pt = rec["points"][pub["point"]]
                if pt["setting"] == "echo":
                    cc[pt["input"]] = rows[k]
        mq = rec["echo"]["measured_qubits"]
        a, se = aligned(cc, n)
        out[(n, D, basis)] = ([int(mq[str(j)]) for j in range(n)], a, se, connected(cc, n))
    return out


def bloch(data, n, D):
    """Per site (vx, vy, vz), |v| and the transverse fraction |v_xy| / |v|, at (n, D) if all three bases were read."""
    if not all((n, D, b) in data for b in "XYZ"):
        return None
    phys = data[(n, D, "Z")][0]
    if any(data[(n, D, b)][0] != phys for b in "XY"):
        raise SystemExit(f"STOP: the X, Y and Z records of n {n} D {D} are on different rings")
    vx, vy, vz = (data[(n, D, b)][1] for b in "XYZ")
    r = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    tf = np.sqrt(vx ** 2 + vy ** 2) / np.maximum(r, 1e-12)
    return dict(phys=phys, vx=vx, vy=vy, vz=vz, r=r, transverse=tf)


def verdict_length(r_seg, r_rest, keep=0.8, lost=0.5):
    """KEPT: the segment's single-site vector length is >= keep x the rest's (the signal was turned, a per-site
    counter-rotation could bring it back); SHRUNK: < lost x the rest's; otherwise MIXED."""
    if r_seg < lost * r_rest:
        return "SHRUNK"
    if r_seg >= keep * r_rest:
        return "KEPT"
    return "MIXED"


def crossings(vals, ses, k=2.0):
    """Signs of the values that are beyond k SE, in depth order, with repeats merged (e.g. '+-+')."""
    out = ""
    for v, s in zip(vals, ses):
        t = "+" if v > k * s else "-" if v < -k * s else ""
        if t and (not out or out[-1] != t):
            out += t
    return out


def verdict_depth(vals, ses):
    """OSCILLATES: + then - then + again over depth (a turning that comes back: a periodic coherent error);
    CROSSES ONCE: + then - (turned past zero, not back within the depths read); STARTS INVERTED: the first value
    beyond 2 SE is already negative; NO CROSSING otherwise."""
    p = crossings(vals, ses)
    if p.startswith("-"):
        return "STARTS INVERTED"
    if "+-+" in p:
        return "OSCILLATES"
    if "+-" in p:
        return "CROSSES ONCE"
    return "NO CROSSING"


def verdict_corr(c_seg, c_rest_pairs, k=3.0):
    """ENTANGLED: the segment's mean connected correlation exceeds the mean of the rest's pairs by more than k x their
    spread / sqrt(#rest pairs) and by at least 0.05; NOT SEEN otherwise."""
    r = np.asarray(c_rest_pairs, float)
    se = float(np.std(r, ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0
    return "ENTANGLED" if c_seg > r.mean() + max(k * se, 0.05) else "NOT SEEN"


def sign_change(depths, vals, ses):
    """(validation helper) OSCILLATES if the values reach both > +2 SE and < -2 SE over the depths, else MONOTONE."""
    pos = any(v > 2 * s for v, s in zip(vals, ses))
    neg = any(v < -2 * s for v, s in zip(vals, ses))
    return "OSCILLATES" if pos and neg else "MONOTONE"


def defect_present(phys, a, path, thr=0.12, min_len=3, need=2, rel=None):
    """True if a dead run (>= min_len consecutive ring sites below thr, cyclic) holds >= need qubits of path. With
    `rel`, the line is max(thr, rel x the ring's median) (a weak run, toqb.safe_loop's rule)."""
    if rel is not None:
        thr = max(thr, rel * float(np.median(a)))
    for st, ln in dead_runs(np.asarray(a), thr=thr, min_len=min_len):
        q = {phys[(st + t) % len(phys)] for t in range(ln)}
        if len(q & set(path)) >= need:
            return True
    return False


def _sites(phys, qs):
    return [j for j, q in enumerate(phys) if q in qs]


def _pairs(phys, qs, inside=True):
    n = len(phys)
    if inside:
        return [j for j in range(n) if phys[j] in qs and phys[(j + 1) % n] in qs]
    return [j for j in range(n) if phys[j] not in qs and phys[(j + 1) % n] not in qs]
