"""Prototype 3D layer (not declared; `user2d` and `compile` are not touched): a kicked Ising circuit on an
L x M x N cubic lattice placed on the Heron r2 heavy-hex by fixed rules, with its floor and mirror in the same job,
read back with the 2D answer card.

Problem: each step RZZ(-pi/2) on every lattice edge, then RX(theta) on every site; Z readout of every site.

Placement (no search, no calibration number; only the coupling graph and the failed marks):
  * lattice column x -> heavy-hex column x (L a multiple of 4, L <= 16); lattice row (y, z) -> one heavy-hex row,
    rows in a snake order of z so that half of the y-bonds start on adjacent heavy-hex rows;
  * x-bonds are native;
  * a bond between lattice rows on adjacent heavy-hex rows goes through the bridge qubit (SWAP, RZZ, SWAP) during a
    "pass": block-local odd-even transposition rounds (blocks of 4 columns) of the rows of that pass only, until
    every column has met its bridge column, ending on the identity (a multiple of 8 rounds);
  * lattice rows that must bond but are apart are brought together by an odd-even sort of the rows' contents
    ("exchange" passes: SWAP, SWAP, SWAP through the bridge, column by column).
The whole layout must avoid failed qubits and couplers; the prototype refuses otherwise (no relocation yet).

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


def bridges(adj, rows):
    """{h: [(bridge label, column)]} for heavy-hex row pairs (h, h+1), read from the coupling graph."""
    out = {}
    for r in range(rows - 1):
        row, nxt = set(range(20 * r, 20 * r + 16)), set(range(20 * (r + 1), 20 * (r + 1) + 16))
        for b in range(20 * r + 16, 20 * r + 20):
            up = [q for q in adj.get(b, ()) if q in row]
            dn = [q for q in adj.get(b, ()) if q in nxt]
            if up and dn and up[0] - 20 * r == dn[0] - 20 * (r + 1):
                out.setdefault(r, []).append((b, up[0] - 20 * r))
    return out


class _Builder:
    def __init__(self, L, M, N, adj):
        from qiskit import QuantumCircuit
        if L % 4 or not 4 <= L <= 16:
            raise Refused("L must be 4, 8, 12 or 16")
        self.L, self.M, self.N = L, M, N
        R = M * N
        if R > 8:
            raise Refused("M x N must be at most 8 (heavy-hex rows of Heron r2)")
        br = bridges(adj, R)
        if len(br) < R - 1:
            raise Refused("the coupling graph lacks a bridge between two rows")
        self.bcols = {h: {c: b for b, c in br[h] if c < L} for h in br}
        self.qc = QuantumCircuit(20 * R + 16)
        self.rowlab = [(y, z if y % 2 == 0 else N - 1 - z) for y in range(M) for z in range(N)]
        self.col = list(range(L))
        self.passes = 0

    def lab(self, h, i):
        return 20 * h + i

    def run_pass(self, jobs):
        L = self.L
        todo = {h: set(range(L)) for h in jobs}
        moving = sorted({h for h in jobs} | {h + 1 for h in jobs})
        r = 0
        while True:
            for h, kind in jobs.items():
                for c, b in self.bcols[h].items():
                    x = self.col[c]
                    if x in todo[h]:
                        a, p = self.lab(h, c), self.lab(h + 1, c)
                        self.qc.swap(a, b)
                        if kind == "bond":
                            self.qc.rzz(-np.pi / 2, b, p)
                        else:
                            self.qc.swap(b, p)
                        self.qc.swap(a, b)
                        todo[h].discard(x)
            if all(not v for v in todo.values()) and r % 8 == 0:
                break
            for blk in range(0, L, 4):
                for i in range(blk + r % 2, blk + 3, 2):
                    for h in moving:
                        self.qc.swap(self.lab(h, i), self.lab(h, i + 1))
                    self.col[i], self.col[i + 1] = self.col[i + 1], self.col[i]
            r += 1
            if r > 48:
                raise RuntimeError("a pass did not close")
        for h, kind in jobs.items():
            if kind == "exchange":
                self.rowlab[h], self.rowlab[h + 1] = self.rowlab[h + 1], self.rowlab[h]
        self.passes += 1

    def step(self, theta):
        L, M, N = self.L, self.M, self.N
        R = M * N
        for h in range(R):
            for i in range(L - 1):
                self.qc.rzz(-np.pi / 2, self.lab(h, i), self.lab(h, i + 1))
        need = set()
        for y in range(M):
            for z in range(N):
                if z + 1 < N:
                    need.add(frozenset(((y, z), (y, z + 1))))
                if y + 1 < M:
                    need.add(frozenset(((y, z), (y + 1, z))))
        done, k = set(), 0
        while True:
            bonds = {h: "bond" for h in range(R - 1) if frozenset((self.rowlab[h], self.rowlab[h + 1])) in need - done}
            if bonds:
                self.run_pass(bonds)
                done |= {frozenset((self.rowlab[h], self.rowlab[h + 1])) for h in bonds}
            if done == need:
                break
            self.run_pass({h: "exchange" for h in range(k % 2, R - 1, 2)})
            k += 1
            if k > 4 * R:
                raise RuntimeError("the row sort did not close")
        for h in range(R):
            for i in range(L):
                self.qc.rx(theta, self.lab(h, i))

    def placed(self):
        """{site index: physical label} at the end."""
        idx = {s: i for i, s in enumerate(sites(self.L, self.M, self.N))}
        return {idx[(self.col[i], y, z)]: self.lab(h, i) for h, (y, z) in enumerate(self.rowlab) for i in range(self.L)}


def physical(dev, L, M, N, t, theta):
    """The placed circuit on device labels (no measurement), the site -> label map and the labels it touches."""
    adj = {}
    for a, b in dev.couplers:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    B = _Builder(L, M, N, adj)
    for _ in range(t):
        B.step(theta)
    qc = B.qc
    used = sorted({qc.find_bit(q).index for ins in qc.data for q in ins.qubits})
    bad = set(dev.failed_qubits)
    badc = {frozenset(c) for c in dev.failed_couplers}
    pairs = {frozenset(qc.find_bit(q).index for q in ins.qubits) for ins in qc.data if len(ins.qubits) == 2}
    if bad & set(used) or badc & pairs:
        raise Refused(f"the fixed layout touches a failed element on {dev.name}: "
                      f"qubits {sorted(bad & set(used))}, couplers {[sorted(c) for c in badc & pairs]}")
    if qc.num_qubits > dev.num_qubits:
        raise Refused("the layout is larger than the device")
    return qc, B.placed(), used, B.passes


def build(dev, L, M, N, t, theta, out, shots=4000, max_seconds=60, tag="g3d"):
    """Place, compile, add floor and mirror, assemble one job. Returns a report dict."""
    from qiskit import QuantumCircuit
    from psf_os import job as J
    from psf_os import mirror2d as Mi
    from psf_os import user2d as U
    from psf_os.compile import RuleCompiler
    qc, placed, used, passes = physical(dev, L, M, N, t, theta)
    n = L * M * N
    phys = [placed[s] for s in range(n)]
    full = QuantumCircuit(dev.num_qubits)
    full.compose(qc, qubits=list(range(qc.num_qubits)), inplace=True)
    rc = RuleCompiler(dev)
    cu = rc.pm.run(U.with_measure(full, phys))
    cf = U.floor_of(cu)
    cm = rc.pm.run(Mi.mirror(full, phys))
    cz = Mi.cz_count(cu)
    if Mi.cz_count(cf) != cz or Mi.cz_count(cm) != 2 * cz:
        raise Refused("acceptance failed: floor or mirror CZ count")
    runs = os.path.join(out, "runs")
    os.makedirs(runs)
    labels = list(range(n))
    mapping = {s: placed[s] for s in labels}
    edges = [list(e) for e in lattice_edges(L, M, N)]
    common = dict(labels=labels, mapping=mapping, shots=shots)
    extra = dict(tag=tag, edges=edges, grid3d=dict(L=L, M=M, N=N, t=t, theta=theta, passes=passes))
    U._record(dev, runs, f"p2d_{tag}", "user", "user2d", [cu], [dict(input="u", setting="z")],
              extra=dict(extra, cz=cz), valid=True, **common)
    U._record(dev, runs, f"p2d_{tag}_floor", "floor", "user2d_floor", [cf], [dict(input="f", setting="z")],
              extra=dict(extra, cz=cz), valid=True, **common)
    U._record(dev, runs, f"p2d_{tag}_mirror", "mirror", "user2d_mirror", [cm], [dict(input="m", setting="z")],
              extra=dict(extra, cz=2 * cz), valid=True, **common)
    man = J.assemble(dev, runs, os.path.join(out, "job"), max_seconds=max_seconds)
    rep = dict(device=dev.name, L=L, M=M, N=N, t=t, theta=theta, sites=n, qubits_used=len(used), cz=cz,
               passes=passes, circuits=len(man["pubs"]), qpu_seconds_model=man["qpu_seconds_model"],
               mapping={str(s): placed[s] for s in labels}, job=os.path.join(out, "job"), runs=runs)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    return rep
