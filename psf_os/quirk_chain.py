"""Chain candidates for the quirk ledger's special condition (OS item 4, workplace VS Claude, 2026-11-13).

The chain rule as the declared builders used it (psf_os.chain.chain_of): the fixed du91 shapes first, each placed by
translation (psf_os.lattice2d, copies in its fixed order), else rule C from the starts of chain.START_ORDER. Here every
chain that rule can place is listed, one per start, in the rule's own order:

  1. for each shape in chain.SHAPES (in order) that fits n: every qualifying translation copy, in lattice2d's order
     (name "shape S1 dr2 dc0");
  2. then for each start of chain.START_ORDER: rule C's chain from that start, if it reaches n (name
     "rule C start (v, row, col)").

A candidate whose qubit set equals an earlier one's is left out (the earlier name is kept). The first candidate is the
chain chain_of chooses, on physical qubits: for a shape, chain_of returns the shape's own snake labels and the
placement translates them to the first qualifying copy, which is candidate 1; for rule C, chain_of's labels are
physical already. Reads only the device record (coupling graph and failed marks); no quality number.

    from psf_os import quirk, quirk_chain
    cands = quirk_chain.candidates(dev, 40)                  # [{"name", "qubits", "rule", ...}, ...]
    ordered, report = quirk.rank(dev.name, cands, "ledger.json", before="2026-11-14")
    chain = ordered[0]["qubits"]                             # physical labels, in chain order

Same rule C budget and health judgement (suspect=True) as chain.place.
"""
from __future__ import annotations

from psf_os import chain as CH


def _shape_copies(dev, n):
    from psf_os import lattice2d as L
    out = []
    for sh in CH.SHAPES:
        try:
            ch = CH.snake(n, sh)
        except ValueError:
            continue
        edges = [tuple(sorted(e)) for e in zip(ch, ch[1:])]
        for dr, dc, m in L.copies(dev, edges, ch):
            if not L.check(dev, edges, m):
                out.append(dict(name=f"shape {sh[0]} dr{dr} dc{dc}", qubits=[int(m[q]) for q in ch], rule="shape",
                                shape=sh[0], shift=[dr, dc]))
    return out


def _rule_c(dev, n, budget=CH.BUDGET):
    out = []
    for st in CH.START_ORDER:
        ch, steps, why = CH.walk(dev, n, *st, budget=budget, suspect=True)
        if why is None:
            out.append(dict(name=f"rule C start {tuple(st)}", qubits=[int(q) for q in ch], rule="C", start=list(st),
                            steps=steps))
    return out


def candidates(dev, n):
    """Every chain of n sites the chain rule (rule C, and the du91 shapes when they apply) can place on dev, one per
    start, in the rule's own fixed order: list of {"name": str, "qubits": [physical...]}. The first is the default."""
    if dev.num_qubits != 156:
        raise CH.Refused(f"the chain rule is written for Heron r2 labels (156 qubits); {dev.name} has {dev.num_qubits}")
    out, seen = [], set()
    for c in _shape_copies(dev, n) + _rule_c(dev, n):
        k = frozenset(c["qubits"])
        if k not in seen:
            seen.add(k)
            out.append(c)
    if not out:
        raise CH.Refused(f"no shape copy and no rule C start reaches {n} sites")
    return out


def default_physical(dev, n):
    """chain_of's chain on physical qubits (shape labels translated by the first qualifying copy)."""
    labels, how = CH.chain_of(dev, n)
    if how["rule"] == "shape":
        pl = CH.shape_place(dev, n)
        return [int(q) for q in pl["chain"]]
    return [int(q) for q in labels]
