"""The entrance's 3D family (2026-11-13; psf_os.entrance calls it): a logical circuit whose interaction graph is a
4 x M x N cubic lattice (the fold's rows and blocks: M <= 8 and N <= 4, or the
reverse; N = 1 is a 4 x M square grid) placed by psf_os.grid3d's fixed fold layout, for ANY
one- and two-qubit gates on the lattice bonds. grid3d.py is not changed; only its layout pieces are used.

Recognition (labelled, no search): the edges equal grid3d.lattice_edges of some (d0, d1, d2) with one side 4, in
itertools.product order of the user's own axes; the axes are permuted so that the side of 4 is grid3d's x. Site
index k of the user = its grid3d site under that permutation (identity when the user wrote grid3d's order).

Two paths:
  * "kicked Ising" (exact): every item is gate for gate grid3d.logical(4, M, N, t, theta) (grid3d order). Then
    grid3d.prepare itself places and compiles it (fold v3.1 by default, v3.2 "carry", or "line"), so the declared 3D
    jobs come out unchanged (the notebook's regress_entrance.py checks the gate hashes record by record).
  * generic: the fold layout of grid3d._Fold (row (y, z) on heavy-hex row r0 + y, block z, odd blocks mirrored; the
    first translation in grid3d.fold_offsets' fixed order that touches no failed element). The circuit is walked in
    its own order; a gate runs as soon as its qubits' earlier gates have run and its two sites sit where the layout
    can join them, otherwise the block schedule (grid3d.FOLD_SCHEDULES, the same fixed rounds of swaps applied to
    every row and block at once) advances one round:
      x-bond: native when the two columns are adjacent in the block;
      z-bond: native when the column sits at the block boundary;
      y-bond: through the block's bridge b (|0> between uses) when the column sits under it:
              a diagonal gate G (RZZ, CZ, CP, ...): CX(a, b) G(b, p) CX(a, b)  (b copies Z_a, as grid3d);
              any other gate (fSim, XX+YY, ...): move a -> b (CX(a, b) CX(b, a)), G(b, p), move back (CX(b, a)
              CX(a, b)): a SWAP through an empty qubit costs 2 CX instead of 3.
    No merging of bonds into swaps (grid3d's v3.1 / v3.2 merge only RZZ(-pi/2)). The swaps and bridge moves are
    reported as the inflation over the logical circuit.
The mirror is grid3d.mirror_of_compiled (level-0 translation: exactly twice the CZ), as every 3D job had; the floor and
the acceptance are grid3d's. Records are written as grid3d.write_records does (labels = site indices, mapping to the
device), so user2d's card reads them.
"""
from __future__ import annotations

import itertools

import numpy as np

from psf_os import grid3d as G


FOLDS = ("v3.1", "v3.2", "line")
MERGE = {"v3.1": True, "v3.2": "carry"}


class Refused3D(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ recognition

def recognise(nodes, edges):
    """dict(M, N, dims, perm, site_of) if the graph is a 4 x M x N lattice in the user's product order, else None.
    site_of[k] = the grid3d site index of the user's site k."""
    n = len(nodes)
    if set(nodes) != set(range(n)):
        return None
    want = {tuple(sorted(e)) for e in edges}
    for d0 in range(1, n + 1):
        if n % d0:
            continue
        for d1 in range(1, n // d0 + 1):
            if (n // d0) % d1:
                continue
            d2 = n // (d0 * d1)
            dims = (d0, d1, d2)
            if 4 not in dims or (1 in dims[:2] and dims != (1, 1, 1) and any(d > 1 for d in dims[dims.index(1):])):
                continue                  # a side of 1 only at the end (a 2D grid is d0 x d1 x 1)
            if {tuple(sorted(e)) for e in G.lattice_edges(*dims)} != want:
                continue
            ax = dims.index(4)
            rest = [i for i in range(3) if i != ax]
            M, N = dims[rest[0]], dims[rest[1]]
            if not ((M <= 8 and N <= 4) or (N <= 8 and M <= 4)):
                return dict(too_large=True, dims=dims)
            perm = (ax, rest[0], rest[1])
            gidx = {s: i for i, s in enumerate(G.sites(4, M, N))}
            site_of = {}
            for k, s in enumerate(itertools.product(*(range(d) for d in dims))):
                site_of[k] = gidx[tuple(s[p] for p in perm)]
            return dict(M=M, N=N, dims=dims, perm=perm, site_of=site_of)
    return None


def _gates(qc):
    return [(i.operation.name, tuple(qc.find_bit(q).index for q in i.qubits),
             tuple(round(float(p), 12) for p in i.operation.params)) for i in qc.data
            if i.operation.name not in ("barrier", "measure")]


def kicked_ising(qc, M, N, site_of):
    """(t, theta) if qc is gate for gate grid3d.logical(4, M, N, t, theta) in grid3d's order, else None."""
    if any(site_of[k] != k for k in site_of):
        return None
    if any(i.operation.name not in ("rzz", "rx", "barrier", "measure") for i in qc.data):
        return None
    g = _gates(qc)
    rx = [p for name, _, p in g if name == "rx"]
    n = 4 * M * N
    if not rx or len(rx) % n:
        return None
    t, theta = len(rx) // n, rx[0][0]
    return (t, theta) if _gates(G.logical(4, M, N, t, theta)) == g else None


def is_diagonal(op):
    if op.name in ("rzz", "cz", "cp", "crz", "cu1", "rz", "z", "s", "sdg", "t", "tdg", "p", "u1"):
        return True
    if op.name in ("rx", "ry", "sx", "x", "y", "h", "swap", "iswap", "xx_plus_yy", "rxx", "ryy", "cx", "ecr"):
        return False
    try:
        from qiskit.quantum_info import Operator
        m = Operator(op).data
        return bool(np.allclose(m, np.diag(np.diag(m)), atol=1e-12))
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------ generic walker

class _Walker:
    """The generic fold path on one translation (see the module docstring)."""

    def __init__(self, M, N, adj, r0, c0, swapped):
        from qiskit import QuantumCircuit
        self.fM, self.fN = (N, M) if swapped else (M, N)     # rows and blocks of the fold
        self.F = G._Fold(self.fM, self.fN, adj, r0, c0, merge=False)
        self.M, self.N, self.swapped = M, N, swapped
        self.qc = QuantumCircuit(self.F.qc.num_qubits)
        self.sched, self.r = None, 0
        self.swaps = self.moves = self.rounds = 0

    def rowblock(self, y, z):
        return (z, y) if self.swapped else (y, z)

    def pos(self, x):
        return self.F.col.index(x)

    def phys(self, site):
        x, y, z = site
        fy, fz = self.rowblock(y, z)
        return self.F.lab(fy, fz, self.pos(x))

    def try_two(self, op, s1, s2):
        """Emit the gate if the layout can join the two sites now; True if emitted."""
        F = self.F
        (x1, y1, z1), (x2, y2, z2) = s1, s2
        (fy1, fz1), (fy2, fz2) = self.rowblock(y1, z1), self.rowblock(y2, z2)
        if (fy1, fz1) == (fy2, fz2):                                     # x-bond
            p1, p2 = self.pos(x1), self.pos(x2)
            if abs(p1 - p2) != 1:
                return False
            self.qc.append(op, [F.lab(fy1, fz1, p1), F.lab(fy2, fz2, p2)])
            return True
        if x1 != x2:
            raise Refused3D(f"not a lattice bond: {s1} {s2}")
        if fy1 == fy2:                                                   # across blocks (native coupler)
            zl = min(fz1, fz2)
            jp = 3 if zl % 2 == 0 else 0
            if self.pos(x1) != jp:
                return False
            self.qc.append(op, [F.lab(fy1, fz1, jp), F.lab(fy2, fz2, jp)])
            return True
        yl = min(fy1, fy2)                                               # across rows: through the bridge
        b, bp = F.bridge[(yl, fz1)]
        if self.pos(x1) != bp:
            return False
        a, p = F.lab(yl, fz1, bp), F.lab(yl + 1, fz1, bp)
        first_low = fy1 == yl
        q = self.qc
        if is_diagonal(op):
            q.cx(a, b)
            q.append(op, [b, p] if first_low else [p, b])
            q.cx(a, b)
        else:
            q.cx(a, b)
            q.cx(b, a)
            q.append(op, [b, p] if first_low else [p, b])
            q.cx(b, a)
            q.cx(a, b)
            self.moves += 1
        return True

    def advance(self):
        """One round of the fixed block schedule on every row and block."""
        F = self.F
        if self.sched is None or self.r == len(self.sched):
            self.sched, self.r = list(F.schedule()), 0
        rnd = self.sched[self.r]
        for z in range(F.N):
            for i in rnd:
                for y in range(F.M):
                    self.qc.swap(F.lab(y, z, i), F.lab(y, z, i + 1))
                    self.swaps += 1
        for i in rnd:
            F.col[i], F.col[i + 1] = F.col[i + 1], F.col[i]
        self.r += 1
        self.rounds += 1

    def run(self, qc, site_of):
        gs = G.sites(4, self.M, self.N)
        ops = [(i.operation, [gs[site_of[qc.find_bit(q).index]] for q in i.qubits]) for i in qc.data
               if i.operation.name not in ("barrier", "measure")]
        diag = [is_diagonal(op) for op, _ in ops]
        done = [False] * len(ops)
        left, idle = len(ops), 0
        while left:
            # a pending gate blocks its qubits for every later gate; a pending diagonal gate only for later
            # non-diagonal ones (diagonal gates commute with each other, so their order among themselves is free)
            blocked, blocked_diag, emitted = set(), set(), False
            for k, (op, ss) in enumerate(ops):
                if done[k]:
                    continue
                if blocked & set(ss) or (not diag[k] and blocked_diag & set(ss)):
                    blocked |= set(ss)
                    continue
                ok = True
                if len(ss) == 1:
                    self.qc.append(op, [self.phys(ss[0])])
                elif len(ss) == 2:
                    ok = self.try_two(op, ss[0], ss[1])
                else:
                    raise Refused3D(f"'{op.name}' on {len(ss)} qubits")
                if ok:
                    done[k], emitted = True, True
                    left -= 1
                elif diag[k]:
                    blocked_diag |= set(ss)
                else:
                    blocked |= set(ss)
            if not emitted:
                self.advance()
                idle += 1
                if idle > 64:
                    raise RuntimeError("the fold schedule did not bring a bond into place")
            else:
                idle = 0

    def placed(self):
        gs = G.sites(4, self.M, self.N)
        return {i: self.phys(s) for i, s in enumerate(gs)}


def physical_generic(dev, qc, M, N, site_of, at=None):
    """(placed circuit on labels, {grid3d site: label}, used labels, info) on the first fold translation that touches
    no failed element; Refused3D with the first reason otherwise."""
    adj = {}
    for a, b in dev.couplers:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    bad, badc = set(dev.failed_qubits), {frozenset(c) for c in dev.failed_couplers}
    cpl = {frozenset(c) for c in dev.couplers}
    first = None
    for swapped, r0, c0 in G.fold_offsets(M, N):
        if at is not None and (r0, c0) not in {tuple(v) for v in at}:
            continue
        try:
            W = _Walker(M, N, adj, r0, c0, swapped)
        except G.Refused as e:
            first = first or str(e)
            continue
        W.run(qc, site_of)
        q = W.qc
        used = sorted({q.find_bit(x).index for ins in q.data for x in ins.qubits})
        pairs = {frozenset(q.find_bit(x).index for x in ins.qubits) for ins in q.data if len(ins.qubits) == 2}
        why = None
        if q.num_qubits > dev.num_qubits:
            why = "the layout is larger than the device"
        elif pairs - cpl:
            why = f"pairs that are not couplers on {dev.name}"
        elif bad & set(used) or badc & pairs:
            why = (f"the fold touches a failed element on {dev.name}: qubits {sorted(bad & set(used))}, couplers "
                   f"{[sorted(c) for c in badc & pairs]}")
        if why:
            first = first or why
            continue
        return q, W.placed(), used, dict(layout="fold-generic" + ("-swapped" if swapped else ""), r0=r0, c0=c0,
                                         swaps=W.swaps, bridge_moves=W.moves, rounds=W.rounds)
    raise Refused3D(f"no translation of the folded layout fits on {dev.name}; first reason: {first}")


def prepare_generic(dev, qc, M, N, site_of, rc=None, at=None):
    """As grid3d.prepare, for a generic circuit: compile, floor, mirror of the compiled circuit, acceptance."""
    from qiskit import QuantumCircuit
    from psf_os import mirror2d as Mi
    from psf_os import user2d as U
    from psf_os.compile import RuleCompiler, rzz_exact
    pq, placed, used, info = physical_generic(dev, qc, M, N, site_of, at=at)
    n = 4 * M * N
    inv = {g: k for k, g in site_of.items()}
    phys = [placed[site_of[k]] for k in range(n)]           # the user's site k -> label
    full = QuantumCircuit(dev.num_qubits)
    full.compose(pq, qubits=list(range(pq.num_qubits)), inplace=True)
    full = rzz_exact(full)
    rc = rc or RuleCompiler(dev)
    cu = rc.pm.run(U.with_measure(full, phys))
    cf = U.floor_of(cu)
    cm = G.mirror_of_compiled(rc, cu, phys)
    checks = G.accept(cu, cf, cm, phys)
    return dict(cu=cu, cf=cf, cm=cm, cz=Mi.cz_count(cu), cz_layers=cu.depth(lambda i: len(i.qubits) == 2),
                phys=phys, used=used, passes=info["rounds"], where=info, checks=checks, placed_circuit=pq,
                site_label={str(k): placed[site_of[k]] for k in range(n)}, inverse=inv,
                grid3d=dict(L=4, M=M, N=N, generic=True, **info))


def prepare(dev, qc, rec3, fold="v3.1", rc=None, at=None):
    """The 3D item: the kicked-Ising path through grid3d.prepare when it applies, else the generic fold path."""
    if fold not in FOLDS:
        raise Refused3D(f"fold {fold!r} is not one of {FOLDS}")
    M, N, site_of = rec3["M"], rec3["N"], rec3["site_of"]
    ki = kicked_ising(qc, M, N, site_of)
    if ki is not None:
        t, theta = ki
        layout = "line" if fold == "line" else "fold"
        try:
            pr = G.prepare(dev, 4, M, N, t, theta, layout=layout, rc=rc, merge=MERGE.get(fold, True), at=at)
        except G.Refused as e:
            raise Refused3D(str(e)) from None
        pr["path"] = "kicked_ising"
        return pr
    if fold == "line":
        raise Refused3D("the line layout (grid3d_line) has schedules for the kicked-Ising step only")
    pr = prepare_generic(dev, qc, M, N, site_of, rc=rc, at=at)
    pr["path"] = "generic"
    return pr


def write_records(dev, runs, tag, pr, shots, M, N, extra=None):
    """The records of one 3D item, as grid3d.write_records (labels = the user's site indices; edges in the user's
    indices when the axes were permuted)."""
    inv = pr.get("inverse")
    if not inv or all(g == k for g, k in inv.items()):
        G.write_records(dev, runs, tag, pr, shots, 4, M, N, extra=dict(grid3d=pr["grid3d"], **(extra or {})))
        return
    from psf_os import user2d as U
    n = 4 * M * N
    labels = list(range(n))
    common = dict(labels=labels, mapping={s: pr["phys"][s] for s in labels}, shots=shots)
    ex = dict(tag=tag, edges=[sorted((inv[a], inv[b])) for a, b in G.lattice_edges(4, M, N)], grid3d=pr["grid3d"],
              **(extra or {}))
    cz = pr["cz"]
    U._record(dev, runs, f"p2d_{tag}", "user", "user2d", [pr["cu"]], [dict(input="u", setting="z")],
              extra=dict(ex, cz=cz), valid=True, **common)
    U._record(dev, runs, f"p2d_{tag}_floor", "floor", "user2d_floor", [pr["cf"]], [dict(input="f", setting="z")],
              extra=dict(ex, cz=cz), valid=True, **common)
    U._record(dev, runs, f"p2d_{tag}_mirror", "mirror", "user2d_mirror", [pr["cm"]], [dict(input="m", setting="z")],
              extra=dict(ex, cz=2 * cz), valid=True, **common)
