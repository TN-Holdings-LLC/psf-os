"""L3 (placement part): the rule's ring placement on a heavy-hex device, from the coupling graph and the failed
elements only.

Ported from TOQB toqb/fold.py at 8d44f05 (lattice, shapes, band_shapes, shapes_for, fold_place, pend_place,
readout_start, spread_holes, read_ring, hole_circuit, pend_circuit, and the `fold:rule` choice of FoldLoop). The code
is unchanged except that it reads a psf_os.device.Device (working couplers, all couplers, failed qubits) instead of
a Qiskit Target, so no error value can enter. The outlines are constructive and in a fixed order: no search, no
quality number.
"""
from __future__ import annotations

from collections import defaultdict


class FoldRefused(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ outlines

def lattice(edges, nq):
    """Heavy-hex read off the coupling graph: bridges (two neighbours, indices not consecutive), rows (components
    without the bridges), and per row pair its bridges (upper qubit, bridge, lower qubit) left to right."""
    nb = defaultdict(set)
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    bridges = {q for q in range(nq) if len(nb[q]) == 2 and all(abs(x - q) > 1 for x in nb[q])}
    row_of, nrows = {}, 0
    for q in range(nq):
        if q in bridges or q in row_of or not nb[q]:
            continue
        stack = [q]
        row_of[q] = nrows
        while stack:
            u = stack.pop()
            for v in nb[u]:
                if v not in bridges and v not in row_of:
                    row_of[v] = nrows
                    stack.append(v)
        nrows += 1
    pairs = defaultdict(list)
    for b in bridges:
        u, v = sorted(nb[b], key=lambda x: row_of.get(x, -1))
        if u in row_of and v in row_of and row_of[u] != row_of[v]:
            pairs[(row_of[u], row_of[v])].append((u, b, v))
    for k in pairs:
        pairs[k].sort()
    return pairs


def _cell_edges(rp, j, pairs):
    (u1, b1, v1), (u2, b2, v2) = pairs[rp][j], pairs[rp][j + 1]
    e = set()
    for a in range(min(u1, u2), max(u1, u2)):
        e.add((a, a + 1))
    for a in range(min(v1, v2), max(v1, v2)):
        e.add((a, a + 1))
    for x, y in ((u1, b1), (b1, v1), (u2, b2), (b2, v2)):
        e.add((min(x, y), max(x, y)))
    return e


def _as_cycle(e):
    nb = defaultdict(list)
    for a, b in e:
        nb[a].append(b)
        nb[b].append(a)
    if not nb or any(len(v) != 2 for v in nb.values()):
        return None
    start = min(nb)
    cyc, prev, cur = [start], None, start
    while True:
        nxt = nb[cur][0] if nb[cur][0] != prev else nb[cur][1]
        if nxt == start:
            break
        cyc.append(nxt)
        prev, cur = cur, nxt
    return cyc if len(cyc) == len(nb) else None


def shapes(pairs):
    """Outlines of k cells side by side in a row pair (8k + 4 qubits), and of k cells plus k2 side by side in the next
    row pair down when the outline is one cycle (24, 32, 36, ...): [(length, name, qubits in cycle order)], in the
    fixed order (row pairs top to bottom, cells left to right)."""
    out, keys = [], sorted(pairs)
    for rp in keys:
        nc = len(pairs[rp]) - 1
        for k in range(1, nc + 1):
            for j in range(nc - k + 1):
                e = set()
                for c in range(j, j + k):
                    e ^= _cell_edges(rp, c, pairs)
                cyc = _as_cycle(e)
                if cyc:
                    out.append((len(cyc), ("row", rp, j, k), cyc))
                for rb in (kk for kk in keys if kk[0] == rp[1]):
                    nb2 = len(pairs[rb]) - 1
                    for k2 in range(1, nb2 + 1):
                        for c in range(nb2 - k2 + 1):
                            e2 = set(e)
                            for cc in range(c, c + k2):
                                e2 ^= _cell_edges(rb, cc, pairs)
                            cyc2 = _as_cycle(e2)
                            if cyc and cyc2:
                                out.append((len(cyc2), ("row+row", rp, j, k, rb, c, k2), cyc2))
    return out


def band_shapes(pairs, min_pairs=3):
    """Outlines of bands over t >= `min_pairs` consecutive row pairs (2026-10-30; the owner's "find n's upper limit"):
    in each row pair of the chain one run of cells (an interval of consecutive cells), each run overlapping the next
    one along the row the two pairs share; kept when the outline is one cycle (`_as_cycle`). Constructive, no search
    and no error value: [(length, ("band", first row pair, t, ((j, k), ...)), qubits in cycle order)] in the fixed
    order: first row pair top to bottom, then t, then the runs (cells from the left: j, then k) from the top pair down."""
    from itertools import product
    keys = sorted(pairs)
    runs = {}
    for rp in keys:
        nc = len(pairs[rp]) - 1
        runs[rp] = [(j, k) for j in range(nc) for k in range(1, nc - j + 1)]

    def cols(rp, j, k, side):  # column span (qubit indices along a row) of the run's upper (0) or lower (1) row
        a_, b_ = pairs[rp][j], pairs[rp][j + k]
        i = 0 if side == 0 else 2
        return min(a_[i], b_[i]), max(a_[i], b_[i])
    out = []
    for a0 in range(len(keys)):
        for t in range(min_pairs, len(keys) - a0 + 1):
            chain = keys[a0:a0 + t]
            if any(chain[i][1] != chain[i + 1][0] for i in range(t - 1)):
                continue
            for combo in product(*(runs[rp] for rp in chain)):
                ok = True
                for i in range(t - 1):
                    lo1, hi1 = cols(chain[i], *combo[i], 1)
                    lo2, hi2 = cols(chain[i + 1], *combo[i + 1], 0)
                    if max(lo1, lo2) >= min(hi1, hi2):  # the runs must share at least one edge of the common row
                        ok = False
                        break
                if not ok:
                    continue
                e = set()
                for rp, (j, k) in zip(chain, combo):
                    for c in range(j, j + k):
                        e ^= _cell_edges(rp, c, pairs)
                cyc = _as_cycle(e)
                if cyc:
                    out.append((len(cyc), ("band", chain[0], t, combo), cyc))
    return out


_BAND_CACHE = {}
EXISTING_MAX = 36  # the longest outline `shapes` can produce on a Heron heavy-hex (row + row): placements up to it are unchanged


def shapes_for(pairs, n, pairs_all=None):
    """The candidate outlines for a ring of n (2026-10-30). For n <= EXISTING_MAX: `shapes` exactly as before, so every
    placement (and every refusal) up to 36 is unchanged. For n > EXISTING_MAX: `shapes` followed by `band_shapes` read
    off `pairs_all`, the lattice of all couplers (the device's geometry), since a failed coupler inside a row would
    otherwise split the row and hide every band across it; the callers keep only outlines whose qubits and couplers
    all work, so failed elements are never used."""
    base = shapes(pairs)
    if n <= EXISTING_MAX:
        return base
    src = pairs_all if pairs_all is not None else pairs
    key = tuple(sorted((rp, tuple(v)) for rp, v in src.items()))
    if key not in _BAND_CACHE:
        _BAND_CACHE[key] = band_shapes(src)
    return base + _BAND_CACHE[key]


def _all_pairs(dev):
    return lattice(dev.all_couplers(), dev.num_qubits)


def fold_place(dev, n):
    """(m, shape name, cycle qubits) for a ring of n, or None: the shortest outline of m >= n qubits, the first in
    the fixed order whose qubits and couplers all work."""
    edges = dev.working_couplers()
    cands = shapes_for(lattice(edges, dev.num_qubits), n, _all_pairs(dev))
    for m in sorted({c[0] for c in cands}):
        if m < n:
            continue
        for length, name, cyc in cands:
            if length != m:
                continue
            if any(not dev.qubit_ok(q) for q in cyc):
                continue
            if all(tuple(sorted((cyc[i], cyc[(i + 1) % m]))) in edges for i in range(m)):
                return m, name, cyc
    return None


def pend_place(dev, n, packed=False):
    """(m, shape name, cycle qubits, {cycle position: pendant qubit}) for a ring of n, or None: the longest cycle of
    m <= n qubits whose qubits and couplers all work and which has n - m cycle nodes with a working outside
    neighbour; pendants spread evenly over those nodes in cycle order, each node's smallest working outside neighbour
    not used before (packed=True: the first n - m such nodes)."""
    edges = dev.working_couplers()
    nb = defaultdict(set)
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    ok = dev.qubit_ok
    cands = shapes_for(lattice(edges, dev.num_qubits), n, _all_pairs(dev))
    for m in sorted({c[0] for c in cands}, reverse=True):
        if m > n:
            continue
        k = n - m
        for length, name, cyc in cands:
            if length != m or not all(ok(q) for q in cyc):
                continue
            if not all(tuple(sorted((cyc[i], cyc[(i + 1) % m]))) in edges for i in range(m)):
                continue
            inside, used, spots = set(cyc), set(), []
            for p, q in enumerate(cyc):
                outs = sorted(x for x in nb[q] if x not in inside and ok(x))
                if outs:
                    spots.append((p, outs))
            if len(spots) < k:
                continue
            pick = spots[:k] if packed else ([spots[(j * len(spots)) // k] for j in range(k)] if k else [])
            pend = {}
            for p, outs in pick:
                free = [x for x in outs if x not in used]
                if not free:
                    break
                pend[p] = free[0]
                used.add(free[0])
            if len(pend) == k:
                return m, name, cyc, pend
    return None


def rule_choice(dev, n):
    """The `fold:rule` choice: of the extra two-qubit gates per layer (pendants n - m_below, holes 2 (m_above - n),
    the zipper (5n - 8)/2 - n or (5n - 7)/2 - n) the fewest decides (ties: pendants, then holes); the zipper is not
    built (FoldRefused). Returns dict(kind, m, shape, cycle, pend)."""
    below, above = pend_place(dev, n), fold_place(dev, n)
    ep = (n - below[0]) if below else None
    eh = 2 * (above[0] - n) if above else None
    ez = (5 * n - 8) // 2 - n if n % 2 == 0 else (5 * n - 7) // 2 - n
    best = min(e for e in (ep, eh, ez) if e is not None)
    if ep is not None and ep == best:
        m, shape, cyc, pend = below
        return dict(kind="pend", m=m, shape=shape, cycle=list(cyc), pend=dict(pend))
    if eh is not None and eh == best:
        m, shape, cyc = above
        return dict(kind="holes", m=m, shape=shape, cycle=list(cyc), pend={})
    raise FoldRefused(f"rule: the zipper has the fewest extra gates at n = {n} (not built by this arm)")


# ------------------------------------------------------------------------------------------------ schedules

def spread_holes(n, m):
    """Cycle position -> ring index (None for a hole): the m - n holes spread evenly."""
    h = m - n
    at = {round((k + 0.5) * m / h) % m for k in range(h)} if h else set()
    while len(at) < h:
        at.add(max(set(range(m)) - at))
    occ, q = [], 0
    for p in range(m):
        if p in at:
            occ.append(None)
        else:
            occ.append(q)
            q += 1
    return occ


def read_ring(qc):
    """TOQB's ring classifier as (n, layers: per layer the single-qubit instructions per qubit, final ones, measures);
    FoldRefused for anything else."""
    n = qc.num_qubits
    ring = {frozenset((i, (i + 1) % n)) for i in range(n)}
    layers, cur, seen, meas = [], [[] for _ in range(n)], set(), []
    for ins in qc.data:
        nm = ins.operation.name
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if nm == "cz":
            e = frozenset(qs)
            if e not in ring or e in seen:
                raise FoldRefused("not a CZ ring in layers")
            seen.add(e)
            if len(seen) == n:
                layers.append(cur)
                cur, seen = [[] for _ in range(n)], set()
        elif nm == "measure":
            meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))
        elif nm == "barrier":
            continue
        elif len(qs) == 1 and not seen:
            if meas:
                raise FoldRefused("a gate after a measurement")
            cur[qs[0]].append(ins.operation)
        else:
            raise FoldRefused(f"{nm} on {qs} inside a ring layer")
    if seen:
        raise FoldRefused("an incomplete ring layer")
    return n, layers, cur, meas


def hole_circuit(qc, cyc, nq):
    """The hole schedule of `qc` on the cycle `cyc` (device qubits in cycle order) of an nq-qubit device:
    (circuit, initial device qubit of each ring index, final one)."""
    from qiskit import QuantumCircuit
    n, layers, final, meas = read_ring(qc)
    m = len(cyc)
    occ = spread_holes(n, m)
    start = {r: cyc[p] for p, r in enumerate(occ) if r is not None}
    out = QuantumCircuit(nq, qc.num_clbits)
    for layer in layers:
        pos = {r: p for p, r in enumerate(occ) if r is not None}
        for r in range(n):
            for op in layer[r]:
                out.append(op, [cyc[pos[r]]])
        edges = [p for p in range(m) if occ[p] is not None and occ[(p + 1) % m] is not None
                 and ((occ[p] + 1) % n == occ[(p + 1) % m] or (occ[(p + 1) % m] + 1) % n == occ[p])]
        rounds = []
        for p in edges:  # commuting CZs in rounds of disjoint pairs, so the schedule is not one long chain
            for rd in rounds:
                if all({p, (p + 1) % m}.isdisjoint({q, (q + 1) % m}) for q in rd):
                    rd.append(p)
                    break
            else:
                rounds.append([p])
        for rd in rounds:
            for p in rd:
                out.cz(cyc[p], cyc[(p + 1) % m])
        for p in [p for p in range(m) if occ[p] is None and occ[(p - 1) % m] is not None]:
            cur, pb = (p - 1) % m, (p + 1) % m
            while occ[pb] is None:
                pb = (pb + 1) % m
            while occ[(cur + 1) % m] is None:
                nx = (cur + 1) % m
                out.cx(cyc[cur], cyc[nx])
                out.cx(cyc[nx], cyc[cur])
                occ[nx], occ[cur] = occ[cur], None
                cur = nx
            out.cz(cyc[cur], cyc[pb])
    pos = {r: p for p, r in enumerate(occ) if r is not None}
    for r in range(n):
        for op in final[r]:
            out.append(op, [cyc[pos[r]]])
    for q, c in meas:
        out.measure(cyc[pos[q]], c)
    return out, [start[r] for r in range(n)], [cyc[pos[r]] for r in range(n)]


def readout_start(m, pend):
    """`fold:rulerot` (2026-10-26): the cycle position for ring node 0 (the read-out node): a position without a
    pendant, farthest along the cycle from every pendant's position; the smallest such position on ties."""
    if not pend:
        return 0
    dist = lambda p: min(min((p - h) % m, (h - p) % m) for h in pend)  # noqa: E731
    free = [p for p in range(m) if p not in pend]
    return max(free, key=lambda p: (dist(p), -p))


def pend_circuit(qc, cyc, pend, nq, start=0):
    """The pendant schedule of `qc` on the cycle `cyc` with pendants {cycle position: device qubit}, ring node 0 at
    cycle position `start` (2026-10-26; 0 before): (circuit, initial device qubit of each ring index, final one)."""
    from qiskit import QuantumCircuit
    n, layers, final, meas = read_ring(qc)
    m = len(cyc)
    if len(pend) >= m:  # workplace review 2026-10-26: every block would wait for its neighbour round the cycle
        raise FoldRefused("pend: every cycle node has a pendant (k = m); no schedule")
    oc, op, r = [None] * m, {}, 0
    for p in [(start + t) % m for t in range(m)]:
        oc[p] = r
        r += 1
        if p in pend:
            op[p] = r
            r += 1
    if r != n:
        raise FoldRefused(f"pend: {m} + {len(pend)} places for {n} ring nodes")

    def where():
        w = {v: cyc[p] for p, v in enumerate(oc)}
        w.update({v: pend[p] for p, v in op.items()})
        return w

    start = where()
    out = QuantumCircuit(nq, qc.num_clbits)
    for layer in layers:
        w = where()
        for i in range(n):
            for o in layer[i]:
                out.append(o, [w[i]])
        need = {frozenset((i, (i + 1) % n)) for i in range(n)}
        while need:
            progress = False
            rounds = []
            for p in range(m):  # plain CZ on ring edges between neighbouring cycle nodes
                e = frozenset((oc[p], oc[(p + 1) % m]))
                if e in need:
                    for rd in rounds:
                        if all({p, (p + 1) % m}.isdisjoint({q, (q + 1) % m}) for q in rd):
                            rd.append(p)
                            break
                    else:
                        rounds.append([p])
                    need.discard(e)
            for rd in rounds:
                for p in rd:
                    out.cz(cyc[p], cyc[(p + 1) % m])
                progress = True
            for p in sorted(op):  # a CZ+SWAP block where the cycle node's other ring edge is done
                u, v = oc[p], op[p]
                e = frozenset((u, v))
                other = frozenset((u, (u - 1) % n)) if (u + 1) % n == v else frozenset((u, (u + 1) % n))
                if e in need and other not in need:
                    a, b = cyc[p], pend[p]
                    out.h(b)  # CZ then SWAP on (a, b) = H(b), CX(b, a), CX(a, b), H(a)
                    out.cx(b, a)
                    out.cx(a, b)
                    out.h(a)
                    oc[p], op[p] = v, u
                    need.discard(e)
                    progress = True
            if not progress:
                raise FoldRefused("pend: no schedule step left")
    w = where()
    for i in range(n):
        for o in final[i]:
            out.append(o, [w[i]])
    for q, c in meas:
        out.measure(w[q], c)
    return out, [start[i] for i in range(n)], [w[i] for i in range(n)]
