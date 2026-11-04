"""Crosstalk test pieces (workplace, 2026-11-04; draft, not declared): P5's circuit with each colour class split into
sublayers of mutually non-adjacent couplers, barriers between the layers, acceptance on the compiled circuit, and
readout-pattern controls with known answers.

Which crosstalk this can measure, stated first:
  - between CZs driven at the same time on neighbouring couplers: splitting a colour class removes such pairs, so this
    part shrinks; the test compares the split arm with the standard arm in the same job;
  - the static (always-on) ZZ between neighbouring qubits: splitting does not remove it; it acts while a qubit waits,
    so the longer split schedule can give it more time. A mirror (U, barrier, U^dagger) can cancel a static ZZ, so a
    mirror drop is not a measurement of the static ZZ.
  - readout crosstalk: a correlation added at measurement. Independent readout errors only scale a connected
    correlation down (C_meas = b_i b_j C for <Z_q>_meas = a_q + b_q <Z_q>); an amplification needs a correlated part.
    The pattern controls measure the pair readout (confusion) matrices from the device's own counts.

Layers. The standard arm is P5 (kim.ops) with a barrier between colour classes, so that the CZs that run together are
known. The split arm splits each colour class greedily, in sorted order (no search): an edge joins the first sublayer
in which it shares no qubit with, and has no device coupler (working or failed: the geometry) to, any edge already
there. The RZZs (CZ + RZ(THETA_J) on both ends) commute, so both arms give the same state (tests).

Acceptance (both arms, and their mirrors): in the compiled circuit, the CZs between consecutive barriers are exactly
the designed groups, in order (a missing barrier merges two groups and is refused).

Readout patterns: X on a fixed set of lattice qubits, then measure; ideal <Z_q> = -1 on the set, +1 elsewhere, and
every connected correlation 0. Patterns (fixed, no seed): all0, all1, A and B (the two sides of the lattice's
bipartition, BFS from its smallest label: every edge sees 01 and 10), M0, M1, M2 (q mod 3 == 0, 1, 2, for pairs at
distance 2 and 3; same-side pairs need them). On the fake snapshots' suspect lattices this covers all four input
states for every distance-1 and distance-3 pair and about 83 % of distance-2 pairs (pairs with i = j mod 3 are not
covered; reported, not searched further). The states include the X gate's error: "readout" here means preparation and measurement.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

from psf_os import kim as K

PATTERNS = ("all0", "all1", "A", "B", "M0", "M1", "M2")


# ------------------------------------------------------------------------------------------------ layers

def split_class(edges, geometry):
    """Split one matching into sublayers of mutually non-adjacent edges (greedy, sorted order)."""
    geo = {tuple(sorted(e)) for e in geometry}

    def near(e, f):
        if set(e) & set(f):
            return True
        return any(tuple(sorted((x, y))) in geo for x in e for y in f)

    subs = []
    for e in sorted(tuple(sorted(x)) for x in edges):
        for S in subs:
            if not any(near(e, f) for f in S):
                S.append(e)
                break
        else:
            subs.append([e])
    return subs


def groups(edges, geometry, split):
    """The CZ groups of one step, in order: the colour classes (standard) or their sublayers (split)."""
    out = []
    for L in K.colour(edges):
        out.extend(split_class(L, geometry) if split else [sorted(tuple(sorted(e)) for e in L)])
    return out


def circuit(qubits, edges, geometry, theta, t, split, width=None, measure=True):
    """P5 with a barrier after every CZ group (the barrier also closes each step before the RX layer)."""
    from qiskit import QuantumCircuit
    width = width or (max(qubits) + 1)
    qc = QuantumCircuit(width, len(qubits) if measure else 0)
    gs = groups(edges, geometry, split)
    for _ in range(t):
        for G in gs:
            for a, b in G:
                qc.cz(a, b)
                qc.rz(K.THETA_J, a)
                qc.rz(K.THETA_J, b)
            qc.barrier(qubits)
        for q in qubits:
            qc.rx(float(theta), q)
    if measure:
        for k, q in enumerate(qubits):
            qc.measure(q, k)
    return qc


def designed(edges, geometry, t, split, mirror=False):
    gs = groups(edges, geometry, split) * t
    return gs + gs[::-1] if mirror else gs


def segments(c):
    """The CZ edges between consecutive barriers of a (compiled) circuit, empty segments dropped."""
    out, cur = [], []
    for ins in c.data:
        name = ins.operation.name
        if name == "barrier":
            if cur:
                out.append(cur)
            cur = []
        elif name == "cz":
            cur.append(tuple(sorted(c.find_bit(q).index for q in ins.qubits)))
    if cur:
        out.append(cur)
    return out


def accept(c, want, geometry):
    """(ok, reason): the compiled circuit's CZ segments equal `want` in order, each edge once, and each segment's edges
    pairwise disjoint (and, for the split arm, the caller's groups are also non-adjacent by construction)."""
    seg = segments(c)
    if len(seg) != len(want):
        return False, f"{len(seg)} CZ segments, designed {len(want)}"
    for k, (s, w) in enumerate(zip(seg, want)):
        if len(s) != len(set(s)):
            return False, f"segment {k}: an edge twice"
        if sorted(s) != sorted(tuple(sorted(e)) for e in w):
            return False, f"segment {k}: {sorted(s)[:4]}... is not the designed group"
        qs = [q for e in s for q in e]
        if len(qs) != len(set(qs)):
            return False, f"segment {k}: two CZs share a qubit"
    return True, "ok"


def is_split(gs, geometry):
    """True if every group's edges are mutually non-adjacent (no shared qubit, no device coupler between them)."""
    geo = {tuple(sorted(e)) for e in geometry}
    for G in gs:
        for i, e in enumerate(G):
            for f in G[i + 1:]:
                if set(e) & set(f) or any(tuple(sorted((x, y))) in geo for x in e for y in f):
                    return False
    return True


def mirror(qubits, edges, geometry, theta, t, split, width=None):
    from psf_os import mirror2d as M
    u = circuit(qubits, edges, geometry, theta, t, split, width=width, measure=False)
    return M.mirror(u, qubits)


# ------------------------------------------------------------------------------------------------ readout patterns

def bipartition(qubits, edges):
    """{q: 0 or 1} by BFS from the smallest label of each connected part. Raises if the lattice is not bipartite."""
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    side = {}
    for s in sorted(qubits):
        if s in side:
            continue
        side[s] = 0
        fr = [s]
        while fr:
            nxt = []
            for x in fr:
                for y in sorted(adj[x]):
                    if y not in side:
                        side[y] = 1 - side[x]
                        nxt.append(y)
                    elif side[y] == side[x]:
                        raise ValueError(f"not bipartite at edge ({x}, {y})")
            fr = nxt
    return side


def pattern_set(name, qubits, edges):
    side = bipartition(qubits, edges)
    return sorted({"all0": [], "all1": list(qubits), "A": [q for q in qubits if side[q] == 0],
                   "B": [q for q in qubits if side[q] == 1], "M0": [q for q in qubits if q % 3 == 0],
                   "M1": [q for q in qubits if q % 3 == 1],
                   "M2": [q for q in qubits if q % 3 == 2]}[name])


def pattern_circuit(qubits, flipped, width=None):
    from qiskit import QuantumCircuit
    width = width or (max(qubits) + 1)
    qc = QuantumCircuit(width, len(qubits))
    for q in flipped:
        qc.x(q)
    qc.barrier(qubits)
    for k, q in enumerate(qubits):
        qc.measure(q, k)
    return qc


def pairs_at(edges, d):
    """Pairs (i < j) at lattice distance exactly d."""
    qs = sorted({q for e in edges for q in e})
    out = []
    for i in qs:
        b = set(K.ball(edges, i, d)) - set(K.ball(edges, i, d - 1))
        out += [(i, j) for j in sorted(b) if j > i]
    return out


def coverage(qubits, edges, pairs, names=PATTERNS):
    """{pair: set of input states (bit_i, bit_j) seen in the patterns}."""
    sets = {n: set(pattern_set(n, qubits, edges)) for n in names}
    return {p: {(int(p[0] in s), int(p[1] in s)) for s in sets.values()} for p in pairs}


# ------------------------------------------------------------------------------------------------ after the run

def bits(counts, n):
    """(array shots x n of 0/1 for clbit k, weights); a (bits, weights) pair is passed through."""
    if isinstance(counts, tuple):
        return counts
    keys = list(counts)
    w = np.array([counts[k] for k in keys], dtype=float)
    b = np.array([[int(k.replace(" ", "")[-1 - j]) for j in range(n)] for k in keys], dtype=np.int8)
    return b, w


def zz(counts, n, pairs):
    """<Z_k>, and {(k, l): <Z_k Z_l>} for clbit positions."""
    b, w = bits(counts, n)
    z = 1 - 2 * b.astype(float)
    tot = w.sum()
    m = (w[:, None] * z).sum(0) / tot
    return m, {(k, l): float((w * z[:, k] * z[:, l]).sum() / tot) for k, l in pairs}


def connected(counts, n, pairs):
    m, c2 = zz(counts, n, pairs)
    return {p: c2[p] - m[p[0]] * m[p[1]] for p in pairs}


def pair_confusion(pattern_counts, flipped_by_pattern, pos, pair):
    """M[out, in] for one pair (in, out = 2 bit_i + bit_j), averaged over the patterns that prepare each input; None for
    an input no pattern prepares."""
    i, j = pair
    ki, kj = pos[i], pos[j]
    M = np.zeros((4, 4))
    seen = np.zeros(4)
    for name, counts in pattern_counts.items():
        fl = set(flipped_by_pattern[name])
        s = 2 * int(i in fl) + int(j in fl)
        b, w = bits(counts, len(pos))
        o = 2 * b[:, ki] + b[:, kj]
        M[:, s] += np.bincount(o, weights=w, minlength=4) / w.sum()
        seen[s] += 1
    if (seen == 0).any():
        return None
    return M / seen[None, :]


def spectator_spread(pattern_counts, flipped_by_pattern, pos, pair):
    """The largest total-variation distance between the pair's output distributions over patterns that prepare the
    same input state (their other qubits differ). 0 if readout of the pair does not depend on its neighbours' states;
    the pair-confusion model assumes it does not. None if no input state is prepared by two patterns."""
    i, j = pair
    by = collections.defaultdict(list)
    for name, counts in pattern_counts.items():
        fl = set(flipped_by_pattern[name])
        s = 2 * int(i in fl) + int(j in fl)
        b, w = bits(counts, len(pos))
        o = 2 * b[:, pos[i]] + b[:, pos[j]]
        by[s].append(np.bincount(o, weights=w, minlength=4) / w.sum())
    tv = [0.5 * np.abs(a - c).sum() for v in by.values() for x, a in enumerate(v) for c in v[x + 1:]]
    return float(max(tv)) if tv else None


def predicted_connected(M, zi, zj, zij):
    """The connected correlation after readout M, for the true pair values (zi, zj, zij): the diagonal two-qubit
    distribution is fixed by them, p(bi, bj) = (1 + si zi + sj zj + si sj zij) / 4 with s = (-1)^b."""
    p = np.array([(1 + si * zi + sj * zj + si * sj * zij) / 4 for si in (1, -1) for sj in (1, -1)])
    q = M @ p
    s_i = np.array([1, 1, -1, -1])
    s_j = np.array([1, -1, 1, -1])
    mi, mj, mij = (s_i * q).sum(), (s_j * q).sum(), (s_i * s_j * q).sum()
    return float(mij - mi * mj)


def resample(counts, rng):
    if isinstance(counts, tuple):
        b, w = counts
        N = int(w.sum())
        return b, rng.multinomial(N, w / N).astype(float)
    keys = list(counts)
    p = np.array([counts[k] for k in keys], dtype=float)
    N = int(p.sum())
    draw = rng.multinomial(N, p / N)
    return {k: int(c) for k, c in zip(keys, draw) if c}


def amplification(counts_by_t, exact_by_t, pos, pairs, min_exact=0.01):
    """R = mean over (t, pair) with |exact C| >= min_exact of measured C / exact C (the correlations note's ratio).
    exact_by_t[t] = {pair: (zi, zj, zij)} with pairs as physical labels; pos[q] = clbit position."""
    rat = []
    for t, counts in counts_by_t.items():
        ex = exact_by_t.get(t, {})
        use = [p for p in pairs if p in ex]
        if not use:
            continue
        cm = connected(counts, len(pos), [(pos[a], pos[b]) for a, b in use])
        for a, b in use:
            zi, zj, zij = ex[(a, b)]
            ce = zij - zi * zj
            if abs(ce) >= min_exact:
                rat.append(cm[(pos[a], pos[b])] / ce)
    return float(np.mean(rat)) if rat else float("nan"), len(rat)


def readout_amplification(confusions, exact_by_t, pairs, min_exact=0.01):
    """The same ratio if readout (the pair confusion matrices) were the only error."""
    rat = []
    for t, ex in exact_by_t.items():
        for p in pairs:
            if p not in ex or confusions.get(p) is None:
                continue
            zi, zj, zij = ex[p]
            ce = zij - zi * zj
            if abs(ce) >= min_exact:
                rat.append(predicted_connected(confusions[p], zi, zj, zij) / ce)
    return float(np.mean(rat)) if rat else float("nan"), len(rat)


def bootstrap(fn, cells, n=200, seed=0):
    """SE of fn(cells) by resampling every cell's shots (multinomial), fixed seed (declared, never changed)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        vals.append(fn({k: resample(c, rng) for k, c in cells.items()}))
    return float(np.std(vals, ddof=1))


def statistics(cells, exact, qubits, edges, flipped, min_exact=0.01):
    """Every judged and reported number of the crosstalk note from one set of cells.
    cells: {("S"|"D", theta, t): counts, ("mS"|"mD", t): counts, ("pat", name): counts} (counts or (bits, weights));
    exact: {theta: {t: {"z": [<Z_q> per qubit], "pairs": {(a, b): (za, zb, zab)}}}} for the non-zero thetas."""
    pos = {q: k for k, q in enumerate(qubits)}
    n = len(qubits)
    d1 = [tuple(sorted(e)) for e in edges]
    out = {}
    for arm in ("S", "D"):
        for th in exact:
            sub = {t: cells[(arm, th, t)] for t in exact[th] if (arm, th, t) in cells}
            ex = {t: exact[th][t]["pairs"] for t in sub}
            out[f"R_{arm}_{th:g}"], out[f"R_n_{th:g}"] = amplification(sub, ex, pos, d1, min_exact)
        se2, cnt = 0.0, 0
        for th in exact:
            for t in exact[th]:
                if (arm, th, t) not in cells:
                    continue
                m, _ = zz(cells[(arm, th, t)], n, [])
                ze = np.asarray(exact[th][t]["z"])
                se2 += float(((m - ze) ** 2).sum())
                cnt += n
                out[f"Merr_{arm}_{th:g}_t{t}"] = float(m.mean() - ze.mean())
        out[f"MSE_{arm}"] = se2 / max(cnt, 1)
        fl = [zz(c, n, [])[0] for k, c in cells.items() if k[0] == arm and k[1] == 0.0]
        floor = np.mean(fl, axis=0) if fl else np.ones(n)
        out[f"floor_{arm}"] = float(floor.mean())
        for k, c in cells.items():
            if k[0] == "m" + arm:
                out[f"mirror_drop_{arm}_t{k[1]}"] = float(floor.mean() - zz(c, n, [])[0].mean())
    out["MSE_ratio_D_over_S"] = out["MSE_D"] / out["MSE_S"] if out.get("MSE_S") else float("nan")
    pats = {k[1]: c for k, c in cells.items() if k[0] == "pat"}
    if pats:
        conf = {p: pair_confusion(pats, flipped, pos, p) for p in d1}
        for th in exact:
            ex = {t: exact[th][t]["pairs"] for t in exact[th]}
            out[f"R_readout_{th:g}"], _ = readout_amplification(conf, ex, d1, min_exact)
        sp = [x for x in (spectator_spread(pats, flipped, pos, p) for p in d1) if x is not None]
        out["spectator_tv_mean"] = float(np.mean(sp)) if sp else float("nan")
        out["spectator_tv_max"] = float(np.max(sp)) if sp else float("nan")
        for name, c in pats.items():
            cc = connected(c, n, [(pos[a], pos[b]) for a, b in d1])
            v = np.array(list(cc.values()))
            out[f"pattern_C_mean_{name}"] = float(v.mean())
            out[f"pattern_C_absmax_{name}"] = float(np.abs(v).max())
    for th in exact:
        rs, rr = out.get(f"R_S_{th:g}"), out.get(f"R_readout_{th:g}")
        out[f"closer_{th:g}"] = abs(out[f"R_S_{th:g}"] - 1) - abs(out[f"R_D_{th:g}"] - 1)
        if rr is not None and rs is not None and abs(rs - 1) > 1e-12:
            out[f"F_readout_{th:g}"] = (rr - 1) / (rs - 1)
    return out


def analyse(cells, exact, qubits, edges, flipped, theta_q1=0.4, nboot=200, seed=0, min_exact=0.01):
    """The statistics, their bootstrap SEs (shots resampled in every cell together, fixed seed) and the proposed
    verdicts (draft thresholds; the declaration fixes them)."""
    n = len(qubits)
    parsed = {k: bits(c, n) for k, c in cells.items()}
    est = statistics(parsed, exact, qubits, edges, flipped, min_exact)
    rng = np.random.default_rng(seed)
    boot = collections.defaultdict(list)
    for _ in range(nboot):
        st = statistics({k: resample(c, rng) for k, c in parsed.items()}, exact, qubits, edges, flipped, min_exact)
        for key, v in st.items():
            boot[key].append(v)
    se = {k: float(np.nanstd(v, ddof=1)) for k, v in boot.items() if k in est}
    th = f"{theta_q1:g}"
    v = {}
    rs, s_rs = est[f"R_S_{th}"], se[f"R_S_{th}"]
    if rs - 1 <= 2 * s_rs:
        v["Q1"] = "NOT TESTED (no amplification in the standard arm)"
    else:
        c, sc = est[f"closer_{th}"], se[f"closer_{th}"]
        v["Q1"] = "CLOSER" if c > 2 * sc else "FARTHER" if c < -2 * sc else "NO CHANGE"
    r = est["MSE_ratio_D_over_S"]
    v["Q2_magnetisation"] = "HELPS" if r <= 0.8 else "HURTS" if r >= 1.25 else "NO EFFECT"
    if v["Q1"].startswith("NOT TESTED") or f"F_readout_{th}" not in est:
        v["Q3"] = "NOT DEFINED (no amplification to explain)" if v["Q1"].startswith("NOT TESTED") else "NO PATTERNS"
    else:
        F = est[f"F_readout_{th}"]
        v["Q3"] = "READOUT" if F >= 0.75 else "NOT READOUT" if F < 0.25 else "PARTLY READOUT"
    return dict(estimate=est, se=se, verdicts=v, nboot=nboot, seed=seed)


# ------------------------------------------------------------------------------------------------ records

def _write(runs, name, cs, rec):
    from qiskit import qpy
    from psf_os.registry import sha256_file
    for k, c in enumerate(cs):
        c.name = f"{name}-{k}"
    os.makedirs(runs, exist_ok=True)
    jf, qf = os.path.join(runs, name + ".json"), os.path.join(runs, name + ".qpy")
    for f in (jf, qf):
        if os.path.exists(f):
            raise FileExistsError(f"{f} exists: a run never overwrites another")
    with open(qf, "wb") as f:
        qpy.dump(cs, f)
    rec["export"] = dict(qpy=os.path.basename(qf), circuits=len(cs), sha256=sha256_file(qf))
    json.dump(rec, open(jf, "w"), indent=1)
    return rec


def build(dev, job, runs, compiler=None, name=None, tag=""):
    """One record per theta with the two arms interleaved (points S t1, D t1, S t2, D t2, ...), so that a drift
    during the job falls on both arms alike. job = {"theta", "ts", "shots", "suspect", "mirror": bool}.
    Acceptance: every compiled circuit's CZ segments are its arm's designed groups, and the split groups are
    mutually non-adjacent."""
    from psf_os.compile import VERSION, RuleCompiler, measured_qubits
    rc = compiler or RuleCompiler(dev)
    qubits, edges = K.region(dev, suspect=bool(job.get("suspect")))
    geo = dev.all_couplers()
    theta = float(job["theta"])
    ts = [int(t) for t in job["ts"]]
    mir = bool(job.get("mirror"))
    cs, points, checks = [], [], []
    for t in ts:
        for arm, sp in (("S", False), ("D", True)):
            lc = (mirror(qubits, edges, geo, theta, t, sp, width=dev.num_qubits) if mir
                  else circuit(qubits, edges, geo, theta, t, sp, width=dev.num_qubits))
            c = rc.pm.run(lc)
            ok, why = accept(c, designed(edges, geo, t, sp, mirror=mir), geo)
            cs.append(c)
            points.append(dict(input=f"{'m' if mir else ''}{arm}{t}", setting="z", t=t, arm=arm))
            checks.append(dict(arm=arm, t=t, ok=ok, why=why))
    ok_meas = all(measured_qubits(c) == {k: q for k, q in enumerate(qubits)} for c in cs)
    gD = groups(edges, geo, True)
    accepted = ok_meas and all(x["ok"] for x in checks) and is_split(gD, geo)
    name = name or f"pxtalk{tag}_{'mir_' if mir else ''}th{theta:.6f}-lattice"
    rec = dict(kind="psfos", psfos_role="user" if not mir else "mirror",
               psfos_job=dict(job, kind="xtalk_mirror" if mir else "xtalk_split"), arm="lattice:region",
               version=VERSION, device=dev.name, masked=dev.masked, qubits=qubits, edges=[list(e) for e in edges],
               groups_standard=[[list(e) for e in G] for G in groups(edges, geo, False)],
               groups_split=[[list(e) for e in G] for G in gD],
               echo=dict(kind="xtalk", checks=checks, accepted=bool(accepted)),
               points=points, valid=bool(accepted), shots=job.get("shots", 4000))
    return _write(runs, name, cs, rec)


def build_patterns(dev, job, runs, compiler=None, name=None):
    """One record of the readout-pattern controls on the same lattice."""
    from psf_os.compile import VERSION, RuleCompiler, measured_qubits
    rc = compiler or RuleCompiler(dev)
    qubits, edges = K.region(dev, suspect=bool(job.get("suspect")))
    names = list(job.get("patterns", PATTERNS))
    flipped = {n: pattern_set(n, qubits, edges) for n in names}
    cs, points = [], []
    for n in names:
        c = rc.pm.run(pattern_circuit(qubits, flipped[n], width=dev.num_qubits))
        cs.append(c)
        points.append(dict(input=f"pat_{n}", setting="z", pattern=n))
    xs = [sorted(c.find_bit(i.qubits[0]).index for i in c.data if i.operation.name == "x") for c in cs]
    ok = all(measured_qubits(c) == {k: q for k, q in enumerate(qubits)} for c in cs) and \
        all(x == flipped[n] for x, n in zip(xs, names))
    rec = dict(kind="psfos", psfos_role="user", psfos_job=dict(job, kind="xtalk_readout"), arm="lattice:region",
               version=VERSION, device=dev.name, masked=dev.masked, qubits=qubits, edges=[list(e) for e in edges],
               flipped=flipped, echo=dict(kind="xtalk_readout", accepted=bool(ok)), points=points, valid=bool(ok),
               shots=job.get("shots", 4000))
    return _write(runs, name or "pxtalk_readout-lattice", cs, rec)
