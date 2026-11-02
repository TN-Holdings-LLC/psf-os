"""L3: the allocator. The rule's placement on the day's device record with the registry's paths masked.

The mask is TOQB's (ported from toqb/defect_list.py middle_edge, mask_target, place_avoiding at 8d44f05): the middle
coupler of every avoided path is marked failed, so the unchanged rule places around it. `place_avoiding` reports
which outlines the rule passed over because they contain an avoided path (TOQB's `place`); `mask` is stricter (it
also skips outlines that use the middle coupler without the whole path), and the build uses `mask`.
"""
from __future__ import annotations

from psf_os import place as P
from psf_os import scan as S


def middle_edge(path):
    k = (len(path) - 1) // 2
    return tuple(sorted((path[k], path[k + 1])))


def mask(dev, paths, reason=""):
    """(masked Device, masked couplers) for avoid entries [{"path": [...], ...}] or plain paths."""
    plain = [p["path"] if isinstance(p, dict) else p for p in paths]
    edges = sorted({middle_edge(p) for p in plain if len(p) >= 2})
    return dev.with_mask(edges, reason=reason), edges


def place_avoiding(dev, n, paths):
    """The rule's outline order with one more condition: the outline contains no avoided path. Returns (m, name,
    cycle, skipped) or None; `skipped` lists the outlines passed over because of the paths."""
    plain = [tuple(p["path"] if isinstance(p, dict) else p) for p in paths]
    edges = dev.working_couplers()
    cands = P.shapes_for(P.lattice(edges, dev.num_qubits), n, P._all_pairs(dev))
    skipped = []
    for m in sorted({c[0] for c in cands}):
        if m < n:
            continue
        for length, name, cyc in cands:
            if length != m:
                continue
            if any(not dev.qubit_ok(q) for q in cyc):
                continue
            if not all(tuple(sorted((cyc[i], cyc[(i + 1) % m]))) in edges for i in range(m)):
                continue
            hit = [p for p in plain if S.contains_path(cyc, p)]
            if hit:
                skipped.append(dict(m=m, name=str(name), hits=[list(p) for p in hit]))
                continue
            return m, name, cyc, skipped
    return None
