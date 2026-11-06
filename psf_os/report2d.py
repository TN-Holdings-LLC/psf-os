"""The extended answer card (report only; nothing here changes a value or a verdict of `user2d.card`).

`psf-os result2d --extended` prints `user2d.card` unchanged, plus two blocks:

flags (always):
  - unreadable: qubits whose floor is below 0.3 (the floor is the user's circuit without its non-diagonal one-qubit
    gates, ideal <Z> = 1; a qubit near 0 there carries no signal: 11-05, ibm_fez q28, floor -0.003);
  - rule_R: qubits below median - max(5 x 1.4826 x MAD, 0.05) of the floor (P6's rule; until now only in TOQB's
    scorers);
  - neighbours: the circuit's own neighbours (distance 1 on its edges) of the qubits above, which may lose signal to
    a stuck partner through the problem's own two-qubit gates (TOQB 05b: the loss the mirror does not see sat there);
  - M over every qubit and M without the flagged ones, side by side (and M_exact over the same qubits if answered).
retention (only when answers are given; the definitions of TOQB gpu/daily_score.py):
  retention M / M_exact (only if |M_exact| >= 0.1), normalised error |M - M_exact| / |M_exact|, per-qubit slope
  sum(z a) / sum(a^2), the MSE left after that one scale, CZ layers of the compiled circuit and the decay per layer
  -ln(slope) / layers.
notes: what the mirror is (a fault locator, not a measure of the error) and the bias of an absolute criterion.
"""
from __future__ import annotations

import math
import os

import numpy as np

FLOOR_UNREADABLE = 0.3
NOTES = [
    "The mirror (U, barrier, U^dagger) locates faulty qubits; its mean is not a measure of the forward error (device "
    "runs of 2026-11-04 and 2026-11-05: mirror means far below the forward results' quality).",
    "A criterion on the absolute difference |M - M_exact| is biased: the same relative loss gives a larger difference "
    "when the exact answer is larger, and a problem with a small answer can agree while losing much; retention and "
    "the normalised error show the loss relative to the answer.",
    "The flags and M without flagged qubits are reported only; no verdict uses them.",
]


def rule_r(z0):
    """(keep mask, dict(median, mad, line)) of P6's rule R on per-qubit floor values z0."""
    z0 = np.asarray(z0, dtype=float)
    med = float(np.median(z0))
    mad = float(np.median(np.abs(z0 - med)))
    line = med - max(5 * 1.4826 * mad, 0.05)
    return z0 >= line, dict(median=med, mad=mad, line=line)


def cz_layers(circuit):
    """CZ layers of a compiled circuit: each CZ one layer after the latest CZ on either of its qubits."""
    last, n = {}, 0
    for ins in circuit.data:
        if ins.operation.name == "cz":
            qs = [circuit.find_bit(x).index for x in ins.qubits]
            k = max(last.get(q, 0) for q in qs) + 1
            for q in qs:
                last[q] = k
            n = max(n, k)
    return n


def flags(labels, edges, floor, z, answers=None, unreadable_below=FLOOR_UNREADABLE):
    """The flag block from per-qubit floor and measured values (lists in label order)."""
    floor, z = np.asarray(floor, float), np.asarray(z, float)
    unread = [q for q, f in zip(labels, floor) if f < unreadable_below]
    keep, rr = rule_r(floor)
    dropped = [q for q, k in zip(labels, keep) if not k]
    bad = set(unread) | set(dropped)
    adj = {q: set() for q in labels}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    nb = sorted({n for q in bad for n in adj[q]} - bad)
    flagged = bad | set(nb)
    rest = [k for k, q in enumerate(labels) if q not in flagged]
    out = dict(unreadable=dict(below=unreadable_below, qubits=sorted(unread)),
               rule_R=dict(rr, dropped=sorted(dropped)), neighbours=nb, flagged=sorted(flagged),
               M_all=float(z.mean()), M_without_flagged=float(z[rest].mean()) if rest else None,
               qubits_without_flagged=len(rest))
    if answers is not None:
        a = np.array([answers.get(str(q), answers.get(q, np.nan)) for q in labels], dtype=float)
        ok_all = ~np.isnan(a)
        ok_rest = np.array([k in rest for k in range(len(labels))]) & ok_all
        out["M_exact_all"] = float(a[ok_all].mean()) if ok_all.any() else None
        out["M_exact_without_flagged"] = float(a[ok_rest].mean()) if ok_rest.any() else None
        if ok_rest.any():
            out["difference_without_flagged"] = float(z[ok_rest].mean() - a[ok_rest].mean())
    return out


def retention(z, a, layers=None):
    """The daily_score.py block from measured z and exact a (answered qubits only)."""
    z, a = np.asarray(z, float), np.asarray(a, float)
    M, Mx = float(z.mean()), float(a.mean())
    slope = float((z * a).sum() / (a * a).sum()) if (a * a).sum() else None
    out = dict(M=M, M_exact=Mx, difference=M - Mx,
               retention=M / Mx if abs(Mx) >= 0.1 else None,
               norm_error=abs(M - Mx) / abs(Mx) if Mx else None,
               slope=slope,
               mse_after_scale=float(np.mean((z - slope * a) ** 2)) if slope is not None else None,
               cz_layers=layers,
               decay_per_layer=(-math.log(slope) / layers) if slope and slope > 0 and layers else None)
    if out["retention"] is None:
        out["retention_note"] = "small answer (|M_exact| < 0.1): retention not given"
    return out


def extended(job_dir, runs, answers=None, tag=None):
    """user2d.card (unchanged) with the flag, retention and notes blocks."""
    from qiskit import qpy
    from psf_os import job as J
    from psf_os import user2d as U
    c = U.card(job_dir, runs, answers, tag=tag)
    labels = c["labels"]
    edges = None
    for name, (rec, _) in J.read(job_dir, runs).items():
        if rec.get("psfos_job", {}).get("kind") == "user2d" and U._tag_of(name, rec) == c["tag"]:
            edges = [tuple(e) for e in rec["edges"]]
            qf = os.path.join(runs, rec["export"]["qpy"])
            with open(qf, "rb") as f:
                layers = cz_layers(qpy.load(f)[0])
            break
    floor = [x["floor"] for x in c["qubits"]]
    z = [x["z"] for x in c["qubits"]]
    out = dict(c, flags=flags(labels, edges, floor, z, answers), notes=list(NOTES))
    out["cz_layers"] = layers
    if answers:
        q = [x for x in c["qubits"] if "exact" in x]
        out["retention"] = retention([x["z"] for x in q], [x["exact"] for x in q], layers)
    return out


def extended_cards(job_dir, runs, answers_by_tag=None):
    from psf_os import user2d as U
    answers_by_tag = answers_by_tag or {}
    return {t: extended(job_dir, runs, answers_by_tag.get(t), tag=t) for t in U.tags_of(job_dir, runs)}
