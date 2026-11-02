"""L4: repair by a local frame flip (psf_os.circuits.frame_flip: X around the CZ ring of alternate layers on chosen
ring positions, virtual Z on their neighbours; the ideal circuit is unchanged, test T5).

Rules (OS plan): avoid first; a repair entry is applied only after it helped on at least two days; repairs are for
mirror-type circuits only. The flip cancels a static ZZ error exactly only in a mirror circuit (forward then inverse;
home's simulation, 1 November), so a forward computation gets no repair.
"""
from __future__ import annotations

MIRROR_KINDS = ("echo",)


def plan(kind, ring, ready):
    """(ring positions to flip, or None; reason). `ready`: repair entries that helped on enough days."""
    if kind not in MIRROR_KINDS:
        return None, f"no repair: '{kind}' is not a mirror circuit"
    if not ready:
        return None, "no repair entry has helped on two days"
    pos = {q: j for j, q in enumerate(ring)}
    sites, used = [], []
    for r in ready:
        if all(q in pos for q in r["path"]) and all(q in pos for q in r["flip"]):
            sites.extend(pos[q] for q in r["flip"])
            used.append(r)
    if not sites:
        return None, "no ready repair entry lies on this ring"
    sites = sorted(set(sites))
    n = len(ring)
    if any((s + 1) % n in sites for s in sites):
        return None, "the entries' flipped positions are ring neighbours; not applied"
    return sites, f"frame flip at ring positions {sites} from {len(used)} entr{'y' if len(used) == 1 else 'ies'}"
