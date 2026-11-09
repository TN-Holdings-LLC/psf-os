"""Line layout for the kicked Ising model on 4 x M x 2 lattices (grid3d layout "line"; VS Claude 2026-11-08, TOQB
notes/ising_line_2026-11-08; ported into psf-os at the workplace 2026-11-09; not declared).

Reached through psf_os.grid3d.physical / prepare / build / probe_records with layout="line" (never by "auto"), for
L = 4, N = 2, M = 2 or 3:
    physical(dev, L, M, N, t, theta, info=False, avoid=(), offsets=None) -> qc, placed, used, passes[, where]
Lattice row y of both blocks z = 0, 1 shares heavy-hex row r0 + y, columns c0 .. c0 + 7: the eight sites (x, z) are
interleaved along the line. Every x- and z-bond of the row is a RZZ(-pi/2) on neighbouring line positions, taken in
place (one CZ after rzz_exact) or riding on a swap as SWAP . RZZ(-pi/2) (two CX, the psf-os _swap_bond identity), so
a schedule moves sites only on bonds (zero empty swaps for M = 2 with bridges at line positions 1, 5; two per line
otherwise). All M lines run the same schedule. The y-bond of row pair (y, y + 1) is taken when a site stands on one of
that pair's bridge positions: CX(a, b) RZZ(b, q) CX(a, b) (the bridge b copies a in the Z basis and returns to |0>).
The bonds of a step commute (all ZZ); the RX(theta) layer follows each step. Even steps run the schedule reversed.
Bridge positions on the line: heavy-hex bridges sit every 4 columns (even rows at columns = 3 mod 4, odd rows at
1 mod 4), so they fall at {1, 5} or {3, 7} for even c0 and at the mirror images {2, 6} / {0, 4} for odd c0 (mirrored
schedules). Translations (r0, c0) are tried cheapest schedule first, then r0, then c0; the first that touches no
failed qubit or coupler and uses couplers only is used (no search over scores), as grid3d.physical. `offsets`
restricts the translations to a given list, in the given order (e.g. the fold's place, for a comparison on the same
qubits). The schedules are constants (found offline by exhaustive or depth-first search, 2026-11-08); nothing is
searched at placement time.
"""
import numpy as np

NAMES = ["0", "1", "2", "3", "0'", "1'", "2'", "3'"]      # x + 4 z
# schedules found by exhaustive search (ladder_bits.py) or depth-first search (line_dfs.c), 2026-11-08
SCHEDULES = {
    # bridge position sets -> (start arrangement, ops: U in place, SU riding on a swap, SW empty swap; digit = e)
    frozenset([(1, 5)]): (["0'", "1", "0", "1'", "2'", "3", "2", "3'"],
                          "SU5 SU6 U3 SU4 SU5 SU1 SU2 U3 SU0 SU1"),
    frozenset([(3, 7)]): (["0'", "1", "2'", "0", "1'", "2", "3", "3'"],
                          "U5 SU6 SW2 SU3 SU4 U5 SU1 SU2 SU3 SU0 SU1 SW2"),
    frozenset([(1, 5), (3, 7)]): (["2", "3'", "1'", "2'", "3", "0", "1", "0'"],
                                  "SU2 SU1 SU0 SU5 SU6 SW3 SU2 SU1 SU4 SU5 SW2 SU3"),
}
CZ_COST = {"U": 1, "SU": 2, "SW": 3}


def _ops(text):
    return [("SU" if o.startswith("SU") else "SW" if o.startswith("SW") else "U", int(o[-1])) for o in text.split()]


def _mirror_set(s):
    return tuple(sorted(7 - p for p in s))


def schedule_for(pos_sets):
    """(start, ops) for the bridge position sets of the row pairs (mirrored if needed), or None."""
    need = frozenset(tuple(sorted(s)) for s in pos_sets)
    if need in SCHEDULES:
        st, text = SCHEDULES[need]
        return [NAMES.index(v) for v in st], _ops(text)
    mneed = frozenset(_mirror_set(s) for s in need)
    if mneed in SCHEDULES:
        st, text = SCHEDULES[mneed]
        return [NAMES.index(v) for v in reversed(st)], [(k, 6 - e) for k, e in _ops(text)]
    return None


def step_cost(ops, M):
    """CZ per step by operation (rzz_exact, _swap_bond): lines x ops + y-bonds x 3."""
    return M * sum(CZ_COST[k] for k, _ in ops) + (M - 1) * 8 * 3


def _swap_bond(qc, a, b):
    """SWAP . RZZ(-pi/2) with 2 CX (psf_os.grid3d._Fold._swap_bond)."""
    qc.z(a)
    qc.z(b)
    qc.h(b)
    qc.cx(b, a)
    qc.cx(a, b)
    qc.h(a)
    qc.s(a)
    qc.s(b)


def _build(n_qubits, lines, bridge, pos_sets, start, ops, t, theta, M):
    """lines[y][p]: label of line position p in row y; bridge[(j, p)]: label of the bridge of row pair j at p."""
    from qiskit import QuantumCircuit
    from psf_os import grid3d as G
    qc = QuantumCircuit(n_qubits)
    arr = list(start)
    line_bonds = {frozenset((x + 4 * z, x + 1 + 4 * z)) for z in (0, 1) for x in range(3)} | \
                 {frozenset((x, x + 4)) for x in range(4)}
    for step in range(t):
        seq = ops if step % 2 == 0 else list(reversed(ops))
        ydone = {j: set() for j in range(M - 1)}
        done = []

        def ybonds():
            for j in range(M - 1):
                for p in pos_sets[j]:
                    if arr[p] not in ydone[j]:
                        a, b, q = lines[j][p], bridge[(j, p)], lines[j + 1][p]
                        qc.cx(a, b)
                        qc.rzz(-np.pi / 2, b, q)
                        qc.cx(a, b)
                        ydone[j].add(arr[p])
        ybonds()
        for kind, e in seq:
            pair = frozenset((arr[e], arr[e + 1]))
            if kind != "SW":
                if pair not in line_bonds:
                    raise RuntimeError(f"schedule applies a bond to non-neighbours {pair}")
                done.append(pair)
            for y in range(M):
                a, b = lines[y][e], lines[y][e + 1]
                if kind == "SU":
                    _swap_bond(qc, a, b)
                elif kind == "U":
                    qc.rzz(-np.pi / 2, a, b)
                else:
                    qc.swap(a, b)
            if kind != "U":
                arr[e], arr[e + 1] = arr[e + 1], arr[e]
            ybonds()
        if sorted(map(sorted, done)) != sorted(map(sorted, line_bonds)) or any(len(v) != 8 for v in ydone.values()):
            raise RuntimeError("a step missed a bond")
        for y in range(M):
            for p in range(8):
                qc.rx(theta, lines[y][p])
    idx = {s: i for i, s in enumerate(G.sites(4, M, 2))}
    placed = {idx[(arr[p] % 4, y, arr[p] // 4)]: lines[y][p] for y in range(M) for p in range(8)}
    return qc, placed


def line_offsets(dev, M):
    """(cost, r0, c0, pos_sets, bridges, schedule) for every translation with a schedule, cheapest first."""
    from psf_os import grid3d as G
    adj = {}
    for a, b in dev.couplers:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    out = []
    for r0 in range(0, 8 - M + 1):
        br = G.bridges(adj, r0, M)
        for c0 in range(0, 16 - 8 + 1):
            pos_sets, bl = [], {}
            for j in range(M - 1):
                ps = sorted(c - c0 for b, c in br.get(j, []) if 0 <= c - c0 < 8)
                pos_sets.append(tuple(ps))
                for b, c in br.get(j, []):
                    if 0 <= c - c0 < 8:
                        bl[(j, c - c0)] = b
            if any(len(ps) != 2 for ps in pos_sets):
                continue
            sch = schedule_for(pos_sets)
            if sch is None:
                continue
            out.append((step_cost(sch[1], M), r0, c0, pos_sets, bl, sch))
    return sorted(out, key=lambda o: (o[0], o[1], o[2]))


def physical(dev, L, M, N, t, theta, info=False, avoid=(), offsets=None):
    from psf_os import grid3d as G
    if L != 4 or N != 2 or M not in (2, 3):
        raise G.Refused("the line layout is written for 4 x 2 x 2 and 4 x 3 x 2")
    avoid = {tuple(a) for a in avoid}
    bad, badc = set(dev.failed_qubits), {frozenset(c) for c in dev.failed_couplers}
    cpl = {frozenset(c) for c in dev.couplers}
    first = None
    cands = line_offsets(dev, M)
    if offsets is not None:
        by = {(o[1], o[2]): o for o in cands}
        cands = [by[tuple(rc)] for rc in offsets if tuple(rc) in by]
        if not cands:
            raise G.Refused(f"the line layout has no schedule at the translations {list(offsets)}")
    for cost, r0, c0, pos_sets, bl, (start, ops) in cands:
        if (r0, c0) in avoid:
            first = first or "passed over by the probe"
            continue
        lines = [[20 * (r0 + y) + c0 + p for p in range(8)] for y in range(M)]
        n_qubits = 20 * (r0 + M) + 16
        if n_qubits > dev.num_qubits:
            first = first or "larger than the device"
            continue
        qc, placed = _build(n_qubits, lines, bl, pos_sets, start, ops, t, theta, M)
        used = sorted({qc.find_bit(q).index for ins in qc.data for q in ins.qubits})
        pairs = {frozenset(qc.find_bit(q).index for q in ins.qubits) for ins in qc.data if len(ins.qubits) == 2}
        if pairs - cpl or bad & set(used) or badc & pairs:
            first = first or f"touches a failed element or a non-coupler at r0 {r0} c0 {c0}"
            continue
        where = dict(layout="line", r0=r0, c0=c0, bridges=[list(s) for s in pos_sets],
                     empty_swaps=sum(1 for k, _ in ops if k == "SW"), cz_per_step_by_ops=cost)
        return (qc, placed, used, t, where) if info else (qc, placed, used, t)  # as grid3d.physical: passes = t
    raise G.Refused(f"no translation of the line layout fits on {dev.name}; first reason: {first}")
