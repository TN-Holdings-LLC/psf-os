"""Honeycomb strips of Heron r2 (2026-11-13; ported from psf-os-lab-notebook notes/ohm_honeycomb_2026-11-13): rows
0 .. R-1 (16 qubits each, labels 20 r + c), columns 0 .. c, and every bridge whose two ends are inside; labels sorted
by (column, level), level = 2 x row for a row qubit and 2 x row + 1 for a bridge. Reads the coupling graph only.

    from psf_os import strip
    phys, bonds = strip.strip(couplers, R, c)       # physical labels in strip order, bonds as index pairs
                                                     # (c is the last column: the note's "2 x 11" strip is c = 10)
    layers = strip.colour(bonds)                     # greedy colours in the sorted bond order
"""
from __future__ import annotations

STRIDE, ROW = 20, 16


def strip(couplers, R, c, bad_qubits=()):
    E = {tuple(sorted(e)) for e in couplers}
    bad = set(bad_qubits)
    qs = [(col, 2 * r, STRIDE * r + col) for r in range(R) for col in range(c + 1) if STRIDE * r + col not in bad]
    inside = {q for _, _, q in qs}
    for r in range(R - 1):
        for b in range(STRIDE * r + ROW, STRIDE * r + STRIDE):
            ends = {y for e in E for y in e if b in e and y != b}
            up = [y for y in ends if STRIDE * r <= y < STRIDE * r + ROW]
            dn = [y for y in ends if STRIDE * (r + 1) <= y < STRIDE * (r + 1) + ROW]
            if up and dn and up[0] in inside and dn[0] in inside and b not in bad:
                qs.append((up[0] - STRIDE * r, 2 * r + 1, b))
    phys = [q for _, _, q in sorted(qs)]
    idx = {q: k for k, q in enumerate(phys)}
    bonds = sorted(tuple(sorted((idx[a], idx[b]))) for a, b in E if a in idx and b in idx)
    return phys, bonds


def colour(bonds):
    col = {}
    for e in bonds:
        used = {col[f] for f in col if set(f) & set(e)}
        col[e] = min(k for k in range(len(bonds)) if k not in used)
    return [[e for e in bonds if col[e] == k] for k in range(max(col.values()) + 1)] if col else []


def find(couplers, edges, n, rows=(2, 3, 4)):
    """(R, c, phys) of the strip whose bonds (as index pairs) equal `edges` with n sites, or None."""
    want = {tuple(sorted(e)) for e in edges}
    for R in rows:
        for c in range(16):
            phys, bonds = strip(couplers, R, c)
            if len(phys) == n and set(bonds) == want:
                return R, c, phys
            if len(phys) > n:
                break
    return None
