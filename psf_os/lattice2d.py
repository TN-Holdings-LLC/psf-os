"""Placement of any edge set written on Heron r2 qubit labels, by translation only (map-free, no search).

A Heron r2 lattice (156 qubits) is rows of 16 qubits (index 20 r + c, c = 0..15) joined by 4 bridge qubits per row gap
(index 20 r + 16 + k, k = 0..3, joining row r at column 3 + 4k (r even) or 1 + 4k (r odd) to row r + 1). The pattern
repeats every two rows (+40 in the index) and every four columns (+4 for a row qubit, +1 for a bridge qubit). So an
edge set written on these labels has exact copies at every (drow, dcol) with drow even and dcol a multiple of 4 that
keeps every qubit inside the device.

`copies(dev, edges)`: every such copy, in a fixed order (drow ascending, then dcol ascending; negative shifts included
when the pattern allows them). A copy whose shifted edges are not all couplers of the device's graph is not a copy
(none on a Heron r2 graph; the check guards against another lattice).
`place(dev, edges)`: the first copy, in that order, with every qubit working, every edge a working coupler, and no
qubit at an end of a failed or masked coupler (P5's suspect-qubit rule). The reasons each earlier copy was passed over
are returned. No copy qualifying raises; the caller's rule (P6's declaration: the next device by queue) applies.
"""
from __future__ import annotations

ROW = 20       # index step from one row to the next (16 row qubits + 4 bridge qubits)
NROW = 16      # row qubits per row
NBRIDGE = 4    # bridge qubits per row gap


def kind(q):
    """("row", r, c) or ("bridge", r, k) of a Heron r2 label."""
    r, c = divmod(q, ROW)
    return ("bridge", r, c - NROW) if c >= NROW else ("row", r, c)


def shift(q, drow, dcol):
    """The label q moved by drow rows (even) and dcol columns (a multiple of 4), or None if it leaves the row pattern
    (a row column outside 0..15 or a bridge index outside 0..3). The device size is checked by the caller."""
    if drow % 2 or dcol % 4:
        raise ValueError("drow must be even and dcol a multiple of 4")
    k, r, c = kind(q)
    if k == "row":
        c2 = c + dcol
        return None if not 0 <= c2 < NROW else ROW * (r + drow) + c2
    k2 = c + dcol // 4
    return None if not 0 <= k2 < NBRIDGE else ROW * (r + drow) + NROW + k2


def copies(dev, edges, qubits=None):
    """[(drow, dcol, {label: physical})] in the fixed order, every copy inside `dev` whose edges are couplers."""
    edges = [tuple(e) for e in edges]
    qubits = sorted(set(qubits or ()) | {q for e in edges for q in e})
    graph = {tuple(sorted(e)) for e in dev.all_couplers()}
    rows = [kind(q)[1] for q in qubits]
    nrows = (dev.num_qubits + ROW - 1) // ROW + 1
    out = []
    for dr in range(-2 * (max(rows) // 2) - 2, 2 * nrows + 2, 2):
        for dc in range(-NROW, NROW + 1, 4):
            m = {}
            for q in qubits:
                p = shift(q, dr, dc)
                if p is None or p < 0 or p >= dev.num_qubits:
                    m = None
                    break
                m[q] = p
            if m is None:
                continue
            if all(tuple(sorted((m[a], m[b]))) in graph for a, b in edges):
                out.append((dr, dc, m))
    return out


def check(dev, edges, m):
    """Why the copy m does not qualify ([] if it does): failed qubits, suspect qubits, couplers not working."""
    working = {tuple(sorted(e)) for e in dev.working_couplers()}
    suspect = {q for c in dev.failed_couplers for q in c}
    why = []
    for q in sorted(m):
        p = m[q]
        if not dev.qubit_ok(p):
            why.append(f"qubit {p} failed")
        elif p in suspect:
            why.append(f"qubit {p} suspect")
    for a, b in edges:
        if tuple(sorted((m[a], m[b]))) not in working:
            why.append(f"coupler ({m[a]}, {m[b]}) not working")
    return why


def place(dev, edges, qubits=None, offsets=None):
    """(drow, dcol, mapping, reasons) of the first qualifying copy. `offsets` (a list of (drow, dcol)) restricts the
    copies to those shifts, in the given order. Raises RuntimeError if none qualifies."""
    edges = [tuple(e) for e in edges]
    cs = copies(dev, edges, qubits)
    if offsets is not None:
        by = {(dr, dc): m for dr, dc, m in cs}
        cs = [(dr, dc, by[(dr, dc)]) for dr, dc in offsets if (dr, dc) in by]
    reasons = []
    for dr, dc, m in cs:
        why = check(dev, edges, m)
        if not why:
            return dr, dc, m, reasons
        reasons.append(dict(drow=dr, dcol=dc, why=why))
    raise RuntimeError(f"no copy of the pattern qualifies: {reasons}")
