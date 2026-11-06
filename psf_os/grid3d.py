"""Prototype 3D layer (not declared; `user2d` and `compile` are not touched): a kicked Ising circuit on an
L x M x N cubic lattice placed on the Heron r2 heavy-hex by fixed rules, with its floor and mirror in the same job,
read back with the 2D answer card.

Problem: each step RZZ(-pi/2) on every lattice edge, then RX(theta) on every site; Z readout of every site.

Placement (no search, no calibration number; only the coupling graph and the failed marks). Two fixed layouts:
  * "fold" (L = 4; v3, 2026-11-06): lattice row (y, z) on heavy-hex row r0 + y, block z (4 columns), odd blocks
    mirrored. x-bonds native in the block, z-bonds native across the block boundary (both ends always hold the same
    lattice column), y-bonds CX(a, b) RZZ(b, p) CX(a, b) through the block's bridge b, which is |0> between uses.
    One fixed block schedule per step (FOLD_SCHEDULES; 6 or 10 swaps per lattice row) brings every column to the
    positions it needs. No exchange passes.
  * "rows" (any L multiple of 4, L <= 16; v2): lattice row (y, z) on one heavy-hex row in a snake order; bonds
    between adjacent rows through the bridge as above during "passes" (PASS_ROUNDS, 6 swaps per block and row, the
    blocks end reversed), rows that must bond but are apart exchanged through the empty bridge (2 + 3 + 2 gates).
  "auto" takes "fold" when L = 4. RZZ(+-pi/2) is written as one CZ (compile.rzz_exact), as user2d does.
  Failed elements: the first translation (row offset, column offset in steps of 4) in a fixed order that touches
  none is used; Refused if none does.
v1 (521ac5b, the first prototype: SWAP-RZZ-SWAP bridges, passes back to the identity, no relocation) is kept in
TOQB notes/os3d_sim_2026-11-06 for comparison.

    from psf_os import grid3d
    rep = grid3d.build(dev, L=4, M=2, N=2, t=2, theta=0.8, out="RUN")     # job folder RUN/job, runs RUN/runs
    grid3d.exact(4, 2, 2, 2, 0.8)                                         # exact <Z> per site (n <= 24)
    psf-os run3d --device DEV --L 4 --M 2 --N 2 --t 2 --theta 0.8 --out RUN
    psf-os result2d --job RUN/job --runs RUN/runs [--answers A.json]     # the same answer card as 2D
"""
import itertools
import json
import os

import numpy as np


class Refused(Exception):
    pass


def sites(L, M, N):
    return list(itertools.product(range(L), range(M), range(N)))


def lattice_edges(L, M, N):
    idx = {s: i for i, s in enumerate(sites(L, M, N))}
    out = []
    for (x, y, z), i in idx.items():
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            t = (x + d[0], y + d[1], z + d[2])
            if t in idx:
                out.append((i, idx[t]))
    return out


def logical(L, M, N, t, theta):
    """The lattice's own circuit on site indices (itertools.product order), for exact answers."""
    from qiskit import QuantumCircuit
    n = L * M * N
    qc = QuantumCircuit(n)
    e = lattice_edges(L, M, N)
    for _ in range(t):
        for a, b in e:
            qc.rzz(-np.pi / 2, a, b)
        for q in range(n):
            qc.rx(theta, q)
    return qc


def exact(L, M, N, t, theta):
    """Exact <Z> per site by statevector (n <= 24), as {str(site index): value}."""
    from qiskit.quantum_info import Statevector
    n = L * M * N
    if n > 24:
        raise ValueError("exact answers here only up to 24 sites")
    p = Statevector(logical(L, M, N, t, theta)).probabilities()
    idx = np.arange(len(p))
    return {str(q): float(np.sum(p * (1 - 2 * ((idx >> q) & 1)))) for q in range(n)}


def bridges(adj, r0, rows):
    """{h: [(bridge label, column)]} for heavy-hex row pairs (r0 + h, r0 + h + 1), read from the coupling graph."""
    out = {}
    for h in range(rows - 1):
        r = r0 + h
        row, nxt = set(range(20 * r, 20 * r + 16)), set(range(20 * (r + 1), 20 * (r + 1) + 16))
        for b in range(20 * r + 16, 20 * r + 20):
            up = [q for q in adj.get(b, ()) if q in row]
            dn = [q for q in adj.get(b, ()) if q in nxt]
            if up and dn and up[0] - 20 * r == dn[0] - 20 * (r + 1):
                out.setdefault(h, []).append((b, up[0] - 20 * r))
    return out


# One pass inside every block of 4 columns: rounds of swaps (local column i <-> i + 1). Every value meets local
# column 3 (rounds 0, 1, 3, 5) and local column 1 (rounds 0, 2, 3, 4), the two places a bridge can hang from, and the
# block ends reversed, which keeps every in-block x-neighbour adjacent.
PASS_ROUNDS = ([2], [1], [2, 0], [1], [2])


class _Builder:
    """v2 (2026-11-06): bridges are |0> between uses, so a bond is CX(a, b) RZZ(b, p) CX(a, b) and an exchange moves
    through the empty bridge (2 + 3 + 2 CX-type gates); a pass is PASS_ROUNDS (6 swaps per block and row, all rows
    move together) and leaves the blocks reversed instead of returning to the identity; x-bonds are applied whenever
    their two sites are neighbours (in-block bonds always; bonds across blocks at an identity moment of the step)."""

    def __init__(self, L, M, N, adj, r0=0, c0=0):
        from qiskit import QuantumCircuit
        if L % 4 or not 4 <= L <= 16:
            raise Refused("L must be 4, 8, 12 or 16")
        self.L, self.M, self.N, self.r0, self.c0 = L, M, N, r0, c0
        R = M * N
        if R > 8:
            raise Refused("M x N must be at most 8 (heavy-hex rows of Heron r2)")
        if r0 + R > 8 or c0 + L > 16:
            raise Refused("the layout is larger than the device")
        br = bridges(adj, r0, R)
        if len(br) < R - 1:
            raise Refused("the coupling graph lacks a bridge between two rows")
        self.bcols = {h: {c - c0: b for b, c in br[h] if c0 <= c < c0 + L} for h in br}
        self.qc = QuantumCircuit(20 * (r0 + R) + 16)
        self.rowlab = [(y, z if y % 2 == 0 else N - 1 - z) for y in range(M) for z in range(N)]
        self.col = list(range(L))
        self.passes = 0

    def lab(self, h, i):
        return 20 * (self.r0 + h) + self.c0 + i

    def x_bonds(self, todo):
        R = self.M * self.N
        for i in range(self.L - 1):
            pair = frozenset((self.col[i], self.col[i + 1]))
            if pair in todo:
                for h in range(R):
                    self.qc.rzz(-np.pi / 2, self.lab(h, i), self.lab(h, i + 1))
                todo.discard(pair)

    def _bridge(self, h, kind, c, b):
        a, p = self.lab(h, c), self.lab(h + 1, c)
        if kind == "bond":  # b is |0>: CX copies Z_a onto b, RZZ(b, p) = RZZ(a, p), CX restores b to |0>
            self.qc.cx(a, b)
            self.qc.rzz(-np.pi / 2, b, p)
            self.qc.cx(a, b)
        else:  # exchange a <-> p through the empty b: move a -> b (2 CX), swap b, p, move b -> a (2 CX)
            self.qc.cx(a, b)
            self.qc.cx(b, a)
            self.qc.swap(b, p)
            self.qc.cx(b, a)
            self.qc.cx(a, b)

    def run_pass(self, jobs, xtodo):
        L, R = self.L, self.M * self.N
        todo = {h: set(range(L)) for h in jobs}
        for r in range(len(PASS_ROUNDS) + 1):
            for h, kind in jobs.items():
                for c, b in self.bcols[h].items():
                    if self.col[c] in todo[h]:
                        self._bridge(h, kind, c, b)
                        todo[h].discard(self.col[c])
            if r == len(PASS_ROUNDS):
                break
            for blk in range(0, L, 4):
                for i in PASS_ROUNDS[r]:
                    for h in range(R):
                        self.qc.swap(self.lab(h, blk + i), self.lab(h, blk + i + 1))
                    self.col[blk + i], self.col[blk + i + 1] = self.col[blk + i + 1], self.col[blk + i]
        if any(todo.values()):
            raise RuntimeError("a pass missed a column (a block without a bridge)")
        for h, kind in jobs.items():
            if kind == "exchange":
                self.rowlab[h], self.rowlab[h + 1] = self.rowlab[h + 1], self.rowlab[h]
        self.passes += 1
        self.x_bonds(xtodo)

    def step(self, theta):
        L, M, N = self.L, self.M, self.N
        R = M * N
        xtodo = {frozenset((x, x + 1)) for x in range(L - 1)}
        self.x_bonds(xtodo)
        need = set()
        for y in range(M):
            for z in range(N):
                if z + 1 < N:
                    need.add(frozenset(((y, z), (y, z + 1))))
                if y + 1 < M:
                    need.add(frozenset(((y, z), (y + 1, z))))
        done, k = set(), 0
        while need:
            bonds = {h: "bond" for h in range(R - 1) if frozenset((self.rowlab[h], self.rowlab[h + 1])) in need - done}
            if bonds:
                self.run_pass(bonds, xtodo)
                done |= {frozenset((self.rowlab[h], self.rowlab[h + 1])) for h in bonds}
            if done == need:
                break
            self.run_pass({h: "exchange" for h in range(k % 2, R - 1, 2)}, xtodo)
            k += 1
            if k > 4 * R:
                raise RuntimeError("the row sort did not close")
        if xtodo:  # only when no pass ran in a reversed step (M = N = 1 never reverses)
            raise RuntimeError("x-bonds left over")
        for h in range(R):
            for i in range(L):
                self.qc.rx(theta, self.lab(h, i))

    def placed(self):
        """{site index: physical label} at the end."""
        idx = {s: i for i, s in enumerate(sites(self.L, self.M, self.N))}
        return {idx[(self.col[i], y, z)]: self.lab(h, i) for h, (y, z) in enumerate(self.rowlab) for i in range(self.L)}


# Fixed block schedules for the folded layout, cheapest first (designed offline once by an exhaustive search over
# short swap sequences on 4 positions; constants here, nothing is searched at placement time). Each is applied to
# every block at once and ends on the identity or the reversal; `_covers` checks the positions every value visits.
FOLD_SCHEDULES = (
    ([2], [1], [2, 0], [1], [2]),                  # 6 swaps: every value visits positions 1, 2 and 3
    ([0], [1], [0, 2], [1], [0]),                  # 6 swaps: every value visits positions 0, 1 and 2
    ([0, 2], [0, 2], [1], [0, 2], [1], [0, 2]),    # 10 swaps: every value visits all four positions
)


def _covers(schedule, start, need):
    perm = list(start)
    seen = {x: {perm.index(x)} for x in range(4)}
    for rnd in schedule:
        for i in rnd:
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
        for x in range(4):
            seen[x].add(perm.index(x))
    return all(need <= seen[x] for x in range(4)) and tuple(perm) in ((0, 1, 2, 3), (3, 2, 1, 0))


class _Fold:
    """Folded layout for L = 4 (v3, 2026-11-06): lattice row (y, z) sits on heavy-hex row r0 + y, in block z (columns
    c0 + 4z .. c0 + 4z + 3), odd blocks mirrored. Then a z-bond is the native coupler between two blocks (both ends
    hold the same lattice column x at all times, because the odd block runs the mirror image of the even block's
    swaps), a y-bond is CX(a, b) RZZ(b, p) CX(a, b) through the block's bridge b (|0> between uses), and an x-bond is
    native inside the block. One fixed block schedule per step brings every x to the needed positions; no exchange
    passes, no rows moved apart."""

    def __init__(self, M, N, adj, r0, c0):
        from qiskit import QuantumCircuit
        if M > 8 or N > 4:
            raise Refused("the folded layout takes M <= 8 rows and N <= 4 blocks")
        if r0 + M > 8 or c0 + 4 * N > 16:
            raise Refused("the layout is larger than the device")
        self.M, self.N, self.r0, self.c0 = M, N, r0, c0
        self.L = 4
        br = bridges(adj, r0, M)
        self.bridge = {}  # (y, z) -> (bridge label, position)
        for y in range(M - 1):
            for b, c in br.get(y, []):
                z, off = divmod(c - c0, 4)
                if 0 <= z < N and c >= c0:
                    self.bridge[(y, z)] = (b, off if z % 2 == 0 else 3 - off)
        if len(self.bridge) < (M - 1) * N:
            raise Refused("the coupling graph lacks a bridge for a block")
        self.need = {p for (_, _), (_, p) in self.bridge.items()} | {3 if z % 2 == 0 else 0 for z in range(N - 1)}
        self.col = [0, 1, 2, 3]
        self.qc = QuantumCircuit(20 * (r0 + M) + 16)
        self.passes = 0

    def schedule(self):
        for sch in FOLD_SCHEDULES:
            if _covers(sch, self.col, self.need):
                return sch
        raise RuntimeError("no fixed schedule covers the needed positions")

    def lab(self, y, z, pos):
        return 20 * (self.r0 + y) + self.c0 + 4 * z + (pos if z % 2 == 0 else 3 - pos)

    def _bonds(self, todo):
        M, N = self.M, self.N
        for y in range(M):
            for z in range(N):
                for i in range(3):
                    x, w = self.col[i], self.col[i + 1]
                    key = ("x", y, z, min(x, w))
                    if abs(x - w) == 1 and key in todo:
                        self.qc.rzz(-np.pi / 2, self.lab(y, z, i), self.lab(y, z, i + 1))
                        todo.discard(key)
                if z + 1 < N:
                    jp = 3 if z % 2 == 0 else 0
                    key = ("z", y, z, self.col[jp])
                    if key in todo:
                        self.qc.rzz(-np.pi / 2, self.lab(y, z, jp), self.lab(y, z + 1, jp))
                        todo.discard(key)
                if y + 1 < M:
                    b, bp = self.bridge[(y, z)]
                    key = ("y", y, z, self.col[bp])
                    if key in todo:
                        a, q = self.lab(y, z, bp), self.lab(y + 1, z, bp)
                        self.qc.cx(a, b)
                        self.qc.rzz(-np.pi / 2, b, q)
                        self.qc.cx(a, b)
                        todo.discard(key)

    def step(self, theta):
        M, N = self.M, self.N
        todo = ({("x", y, z, x) for y in range(M) for z in range(N) for x in range(3)}
                | {("z", y, z, x) for y in range(M) for z in range(N - 1) for x in range(4)}
                | {("y", y, z, x) for y in range(M - 1) for z in range(N) for x in range(4)})
        self._bonds(todo)
        if todo:
            for rnd in self.schedule():
                for z in range(N):
                    for i in rnd:
                        for y in range(M):
                            self.qc.swap(self.lab(y, z, i), self.lab(y, z, i + 1))
                for i in rnd:
                    self.col[i], self.col[i + 1] = self.col[i + 1], self.col[i]
                self._bonds(todo)
            self.passes += 1
        if todo:
            raise RuntimeError(f"bonds left over: {sorted(todo)[:4]}")
        for y in range(M):
            for z in range(N):
                for pos in range(4):
                    self.qc.rx(theta, self.lab(y, z, pos))

    def placed(self, swapped):
        """{site index: label}; `swapped`: the lattice's (y, z) were laid out as (z, y)."""
        L, M, N = 4, (self.N if swapped else self.M), (self.M if swapped else self.N)
        idx = {s: i for i, s in enumerate(sites(L, M, N))}
        out = {}
        for y in range(self.M):
            for z in range(self.N):
                for pos in range(4):
                    site = (self.col[pos], z, y) if swapped else (self.col[pos], y, z)
                    out[idx[site]] = self.lab(y, z, pos)
        return out


def fold_offsets(M, N):
    """Fixed order for the folded layout: rows (M or N, whichever fits; N <= 4 blocks preferred as given), then row
    offsets whose schedule is cheaper first, then column offsets."""
    out = []
    for swapped, (m, n) in ((False, (M, N)), (True, (N, M))):
        if n > 4 or m > 8 or (swapped and M == N):
            continue
        for r0 in range(0, 8 - m + 1):
            for c0 in range(0, 16 - 4 * n + 1, 4):
                # positions needed for this row parity (heavy-hex rows: even rows bridge at column 3 of a block)
                need = set()
                for y in range(m - 1):
                    for z in range(n):
                        off = 3 if (r0 + y) % 2 == 0 else 1
                        need.add(off if z % 2 == 0 else 3 - off)
                need |= {3 if z % 2 == 0 else 0 for z in range(n - 1)}
                cost = next((sum(len(r) for r in sch) for sch in FOLD_SCHEDULES if _covers(sch, [0, 1, 2, 3], need)),
                            99)
                out.append((cost, swapped, r0, c0))
    return [o[1:] for o in sorted(out, key=lambda o: (o[0], o[1], o[2], o[3]))]


def exact_placed_check(qc, placed, used, L, M, N, t, theta):
    """max |<Z> placed - exact| without noise (for tests and the dry run)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    m = {u: i for i, u in enumerate(used)}
    small = QuantumCircuit(len(used))
    for ins in qc.data:
        small.append(ins.operation, [m[qc.find_bit(q).index] for q in ins.qubits])
    p = Statevector(small).probabilities()
    idx = np.arange(len(p))
    ex = exact(L, M, N, t, theta)
    return max(abs(float(np.sum(p * (1 - 2 * ((idx >> m[placed[s]]) & 1)))) - ex[str(s)]) for s in placed)


def offsets(L, M, N):
    """The fixed order in which translations are tried (heavy-hex row offset, column offset in steps of 4)."""
    return [(r0, c0) for r0 in range(0, 8 - M * N + 1) for c0 in range(0, 16 - L + 1, 4)]


def physical(dev, L, M, N, t, theta, layout="auto", info=False):
    """The placed circuit on device labels (no measurement), the site -> label map, the labels it touches and the
    number of passes (block schedules). layout "auto": the folded layout for L = 4 (`_Fold`), else the row layout
    (`_Builder`); "fold" or "rows" to force one. The first translation in the fixed order that touches no failed
    qubit or coupler is used (not a search over scores); Refused if none does."""
    adj = {}
    for a, b in dev.couplers:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    bad = set(dev.failed_qubits)
    badc = {frozenset(c) for c in dev.failed_couplers}
    cpl = {frozenset(c) for c in dev.couplers}
    first = None

    def fits(qc):
        used = sorted({qc.find_bit(q).index for ins in qc.data for q in ins.qubits})
        pairs = {frozenset(qc.find_bit(q).index for q in ins.qubits) for ins in qc.data if len(ins.qubits) == 2}
        if qc.num_qubits > dev.num_qubits:
            return None, "the layout is larger than the device"
        if pairs - cpl:  # the labels assume the Heron r2 numbering (20 per row); refuse rather than trust it
            return None, f"the layout uses pairs that are not couplers on {dev.name}: {[sorted(c) for c in pairs - cpl][:4]}"
        if bad & set(used) or badc & pairs:
            return None, (f"the fixed layout touches a failed element on {dev.name}: qubits "
                          f"{sorted(bad & set(used))}, couplers {[sorted(c) for c in badc & pairs]}")
        return used, None

    if L == 4 and layout in ("auto", "fold"):
        for swapped, r0, c0 in fold_offsets(M, N):
            try:
                B = _Fold(N if swapped else M, M if swapped else N, adj, r0, c0)
            except Refused as e:
                first = first or str(e)
                continue
            for _ in range(t):
                B.step(theta)
            used, why = fits(B.qc)
            if used is None:
                first = first or why
                continue
            out = (B.qc, B.placed(swapped), used, B.passes)
            return out + (dict(layout="fold" + ("-swapped" if swapped else ""), r0=r0, c0=c0),) if info else out
        if layout == "fold":
            raise Refused(f"no translation of the folded layout fits on {dev.name}; first reason: {first}")
    if layout == "fold":
        raise Refused("the folded layout needs L = 4")
    for r0, c0 in offsets(L, M, N):
        try:
            B = _Builder(L, M, N, adj, r0, c0)
        except Refused as e:
            if "L must be" in str(e) or "M x N" in str(e):
                raise
            first = first or str(e)
            continue
        for _ in range(t):
            B.step(theta)
        qc = B.qc
        if qc.num_qubits > dev.num_qubits:
            first = first or "the layout is larger than the device"
            continue
        used, why = fits(qc)
        if used is None:
            first = first or why
            continue
        out = (qc, B.placed(), used, B.passes)
        return out + (dict(layout="rows", r0=r0, c0=c0),) if info else out
    raise Refused(f"no translation of the fixed layout fits on {dev.name}; first reason: {first}")


def mirror_of_compiled(rc, cu, phys):
    """U (the compiled circuit), barrier, U^dagger, Z readout of `phys`, put into the basis by a translation-only pass
    manager (level 0): exactly twice the CZs of U. Recompiling the logical mirror at level 1 may cancel a few more
    CZs in the reversed half than in U (seen on the folded layout), which would break the floor/mirror check."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    body = cu.remove_final_measurements(inplace=False)
    out = QuantumCircuit(body.num_qubits, len(phys))
    out.compose(body, qubits=list(range(body.num_qubits)), inplace=True)
    out.barrier()
    out.compose(body.inverse(), qubits=list(range(body.num_qubits)), inplace=True)
    for k, q in enumerate(phys):
        out.measure(q, k)
    pm0 = generate_preset_pass_manager(0, target=rc.target, initial_layout=list(range(body.num_qubits)),
                                       routing_method="none", seed_transpiler=0)
    return pm0.run(out)


def accept(cu, cf, cm, phys):
    """The 2D acceptance checks (as user2d.build_batch): user and mirror measure `phys` in order, the floor keeps
    every CZ and is diagonal, the mirror has twice the CZs. Returns the checks; Refused if one fails."""
    from psf_os import mirror2d as Mi
    from psf_os.compile import measured_qubits
    want = {k: p for k, p in enumerate(phys)}
    cz = Mi.cz_count(cu)
    checks = dict(user_measures=measured_qubits(cu) == want, floor_cz=Mi.cz_count(cf) == cz,
                  floor_diagonal=all(i.operation.name in ("cz", "rz", "barrier", "measure", "delay") for i in cf.data),
                  mirror_cz=Mi.cz_count(cm) == 2 * cz, mirror_measures=measured_qubits(cm) == want)
    if not all(checks.values()):
        raise Refused(f"acceptance failed: {checks}")
    return checks


def prepare(dev, L, M, N, t, theta, layout="auto", rc=None):
    """Place and compile one circuit with its floor and mirror (nothing written). Returns a dict."""
    from qiskit import QuantumCircuit
    from psf_os import mirror2d as Mi
    from psf_os import user2d as U
    from psf_os.compile import RuleCompiler, rzz_exact
    qc, placed, used, passes, where = physical(dev, L, M, N, t, theta, layout=layout, info=True)
    n = L * M * N
    phys = [placed[s] for s in range(n)]
    full = QuantumCircuit(dev.num_qubits)
    full.compose(qc, qubits=list(range(qc.num_qubits)), inplace=True)
    full = rzz_exact(full)  # RZZ(+-pi/2): one CZ, not two (as user2d)
    rc = rc or RuleCompiler(dev)
    cu = rc.pm.run(U.with_measure(full, phys))
    cf = U.floor_of(cu)
    cm = mirror_of_compiled(rc, cu, phys)
    checks = accept(cu, cf, cm, phys)
    return dict(cu=cu, cf=cf, cm=cm, cz=Mi.cz_count(cu), cz_layers=cu.depth(lambda i: len(i.qubits) == 2),
                phys=phys, used=used, passes=passes, where=where, checks=checks,
                grid3d=dict(L=L, M=M, N=N, t=t, theta=theta, passes=passes, **where))


def write_records(dev, runs, tag, prep, shots, L, M, N, extra=None):
    """The user, floor and mirror records of one prepared circuit (names p2d_<tag>, _floor, _mirror)."""
    from psf_os import user2d as U
    n = L * M * N
    labels = list(range(n))
    mapping = {s: prep["phys"][s] for s in labels}
    common = dict(labels=labels, mapping=mapping, shots=shots)
    ex = dict(tag=tag, edges=[list(e) for e in lattice_edges(L, M, N)], **(extra or {}))
    cz = prep["cz"]
    U._record(dev, runs, f"p2d_{tag}", "user", "user2d", [prep["cu"]], [dict(input="u", setting="z")],
              extra=dict(ex, cz=cz), valid=True, **common)
    U._record(dev, runs, f"p2d_{tag}_floor", "floor", "user2d_floor", [prep["cf"]], [dict(input="f", setting="z")],
              extra=dict(ex, cz=cz), valid=True, **common)
    U._record(dev, runs, f"p2d_{tag}_mirror", "mirror", "user2d_mirror", [prep["cm"]], [dict(input="m", setting="z")],
              extra=dict(ex, cz=2 * cz), valid=True, **common)


def build(dev, L, M, N, t, theta, out, shots=4000, max_seconds=60, tag="g3d", layout="auto"):
    """Place, compile, add floor and mirror, assemble one job. Returns a report dict."""
    from psf_os import job as J
    pr = prepare(dev, L, M, N, t, theta, layout=layout)
    n = L * M * N
    runs = os.path.join(out, "runs")
    os.makedirs(runs)
    write_records(dev, runs, tag, pr, shots, L, M, N, extra=dict(grid3d=pr["grid3d"]))
    man = J.assemble(dev, runs, os.path.join(out, "job"), max_seconds=max_seconds)
    rep = dict(device=dev.name, L=L, M=M, N=N, t=t, theta=theta, sites=n, qubits_used=len(pr["used"]), cz=pr["cz"],
               cz_layers=pr["cz_layers"], passes=pr["passes"], **pr["where"], circuits=len(man["pubs"]),
               qpu_seconds_model=man["qpu_seconds_model"],
               mapping={str(s): pr["phys"][s] for s in range(n)}, job=os.path.join(out, "job"), runs=runs)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    return rep
