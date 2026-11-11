"""Problem cards (2026-11-10, not declared): one JSON format for the time evolution of a spin system, its checks at
the entrance, the logical circuit made from it and its exact answers. The first step towards one entrance for the
models that so far each have their own script (kicked Ising: grid3d; Heisenberg: TOQB gpu/heis3d.py; Schwinger: TOQB
gpu/schwinger.py). Placement is not done here (that is psf-os's existing entrance); `user2d.py` and `compile.py` are
not touched. The format is described in docs/card.md.

A card (JSON object):
    {"card": 1, "name": "...",
     "lattice":     {"kind": "chain" | "grid2d" | "grid3d", "size": [L] | [L, M] | [L, M, N], "periodic": false}
                    or {"kind": "graph", "sites": n, "edges": [[a, b], ...]},
     "hamiltonian": {"terms": [{"pauli": "ZZ", "sites": [a, b], "coef": c, "group": "g"},
                               {"pauli": "X", "on": "sites", "coef": c, "group": "g"},
                               {"pauli": "ZZ", "on": "edges", "axis": 0, "coef": c, "group": "g"}, ...]},
     "initial":     {"bits": "0101..."} | {"product": "++++..."} | {"product": "+"},
     "time":        {"dt": 0.05, "steps": 4, "order": ["g1", "g2", ...]}   (or "orders": [[...], [...]], cycled),
     "measure":     [{"op": "Z", "sites": "all"}, {"op": "ZZ", "sites": [i, j]}, {"op": "X", "sites": [i]}, ...],
     "conserved":   [{"name": "...", "terms": [{"pauli": "Z", "on": "sites", "coef": 1}, ...]}]   (optional)}

Sites are numbered 0 .. n-1; on a grid, site (x, y, z) is number (x * M + y) * N + z (itertools.product order, the
order of psf_os.grid3d.sites). A step applies, for each group in the step's order, exp(-i dt c P) for every term
(c, P) of the group; the terms of a group must commute, so their order inside the group does not matter. A two-site
term lies on an edge of the lattice. Qubit i of the logical circuit is site i.

    from psf_os import card as C
    card = C.load("card.json")
    rep = C.check(card)          # CardRefused(reasons) before anything runs
    qc = C.circuit(card)         # logical Trotter circuit, no measurement
    ans = C.exact(card)          # {"Z_0": ..., "Z_0 Z_1": ...}, exact expectation values of that circuit
    ans = C.exact(card, method="tn")   # light-cone tensor network (psf_os.exact_tn), no site limit, memory guard
    python -m psf_os.card check CARD.json [--tn]     # the entrance; exit 2 with every reason when refused
    python -m psf_os.card exact CARD.json [--out ANS.json] [--method auto|statevector|aer|tn]
"""
from __future__ import annotations

import itertools
import json
import math
import numbers
import sys

import numpy as np

CARD_VERSION = 1
MAX_EXACT_SITES = 28          # exact answers by statevector up to this many sites (method "tn": the memory guard)
COMMUTE_ATOL = 1e-10          # a commutator counts as zero below this (largest Pauli coefficient)
PAULI = "XYZ"
PRODUCT = {"0": "|0>", "1": "|1>", "+": "|+>", "-": "|->", "r": "|+i>", "l": "|-i>"}


class CardRefused(Exception):
    """The card is refused at the entrance; .reasons lists every reason found."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------
# lattice
# ---------------------------------------------------------------------------------------------------------------

def lattice(lat):
    """(n, edges) with edges [(a, b, axis)], a < b; axis is the grid direction (None for a graph)."""
    kind = lat.get("kind")
    if kind == "graph":
        n = lat.get("sites")
        if not _is_int(n) or n < 1:
            raise CardRefused(["lattice: a graph needs 'sites', a positive integer"])
        edges, seen = [], set()
        for e in lat.get("edges", []):
            if not (isinstance(e, (list, tuple)) and len(e) == 2 and all(_is_int(v) for v in e)):
                raise CardRefused([f"lattice: edge {e!r} is not a pair of site numbers"])
            a, b = sorted(e)
            if a == b or a < 0 or b >= n:
                raise CardRefused([f"lattice: edge {e!r} is not between two sites 0 .. {n - 1}"])
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b, None))
        return n, edges
    dims = {"chain": 1, "grid2d": 2, "grid3d": 3}.get(kind)
    if dims is None:
        raise CardRefused([f"lattice: unknown kind {kind!r} (chain, grid2d, grid3d, graph)"])
    size = lat.get("size")
    if not (isinstance(size, (list, tuple)) and len(size) == dims and all(_is_int(v) and v >= 1 for v in size)):
        raise CardRefused([f"lattice: {kind} needs 'size', a list of {dims} positive integers"])
    periodic = bool(lat.get("periodic", False))
    coords = list(itertools.product(*[range(v) for v in size]))
    idx = {c: i for i, c in enumerate(coords)}
    edges, seen = [], set()
    for c, i in idx.items():
        for ax in range(dims):
            t = list(c)
            t[ax] += 1
            if t[ax] == size[ax]:
                if not (periodic and size[ax] > 2):
                    continue
                t[ax] = 0
            j = idx[tuple(t)]
            a, b = min(i, j), max(i, j)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b, ax))
    return len(coords), edges


def _is_int(v):
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _real(v):
    """The coefficient as a float, or None if it is not a real number (complex, string, bool, non-finite)."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------------------------------------------
# terms
# ---------------------------------------------------------------------------------------------------------------

def _expand(specs, n, edges, where, reasons, need_group, on_edges=True):
    """Term specs -> [dict(pauli, sites, coef, group)]; reasons collected. on_edges: two-site terms must be edges."""
    eset = {(a, b) for a, b, _ in edges}
    out = []
    for k, t in enumerate(specs):
        tag = f"{where} term {k}"
        if not isinstance(t, dict):
            reasons.append(f"{tag}: not an object")
            continue
        p = t.get("pauli")
        if not (isinstance(p, str) and 1 <= len(p) <= 2 and all(ch in PAULI for ch in p)):
            reasons.append(f"{tag}: pauli {p!r} is not one or two letters of X, Y, Z")
            continue
        c = _real(t.get("coef"))
        if c is None:
            reasons.append(f"{tag}: coefficient {t.get('coef')!r} is not a real number (H must be Hermitian)")
            continue
        g = t.get("group")
        if need_group and not (isinstance(g, str) and g):
            reasons.append(f"{tag}: needs a 'group' (a name used in time.order)")
            continue
        if "on" in t:
            on = t["on"]
            if on == "sites" and len(p) == 1:
                site_lists = [(i,) for i in range(n)]
            elif on == "edges" and len(p) == 2:
                ax = t.get("axis")
                site_lists = [(a, b) for a, b, eax in edges if ax is None or eax == ax]
                if not site_lists:
                    reasons.append(f"{tag}: no edge on axis {ax!r}")
                    continue
            else:
                reasons.append(f"{tag}: 'on' must be 'sites' for a one-letter or 'edges' for a two-letter pauli")
                continue
        else:
            s = t.get("sites")
            if not (isinstance(s, (list, tuple)) and len(s) == len(p) and all(_is_int(v) for v in s)):
                reasons.append(f"{tag}: sites {s!r} must be {len(p)} site number(s) for pauli {p}")
                continue
            site_lists = [tuple(s)]
        for s in site_lists:
            if any(v < 0 or v >= n for v in s):
                reasons.append(f"{tag}: site {list(s)} outside 0 .. {n - 1}")
                continue
            if len(s) == 2 and s[0] == s[1]:
                reasons.append(f"{tag}: the two sites are the same ({s[0]})")
                continue
            if on_edges and len(s) == 2 and (min(s), max(s)) not in eset:
                reasons.append(f"{tag}: {p} on sites {list(s)} is not on an edge of the lattice")
                continue
            out.append(dict(pauli=p, sites=tuple(int(v) for v in s), coef=c, group=g))
    return out


def _op(terms, n):
    from qiskit.quantum_info import SparsePauliOp
    if not terms:
        return SparsePauliOp.from_sparse_list([("", [], 0.0)], num_qubits=n)
    return SparsePauliOp.from_sparse_list([(t["pauli"], list(t["sites"]), t["coef"]) for t in terms], num_qubits=n)


def _comm_norm(A, B):
    """Largest |coefficient| of [A, B] (SparsePauliOp), after simplification."""
    c = (A.compose(B) - B.compose(A)).simplify(atol=1e-14)
    return float(np.max(np.abs(c.coeffs))) if len(c.coeffs) else 0.0


def _pair_commute(t1, t2):
    """Two Pauli terms commute iff they anticommute on an even number of shared sites."""
    a = dict(zip(t1["sites"], t1["pauli"]))
    b = dict(zip(t2["sites"], t2["pauli"]))
    return sum(1 for s in set(a) & set(b) if a[s] != b[s]) % 2 == 0


# ---------------------------------------------------------------------------------------------------------------
# the check at the entrance
# ---------------------------------------------------------------------------------------------------------------

def parse(card, exact=True):
    """Everything checked; returns the parsed problem (dict) or raises CardRefused with every reason found.
    exact: refuse more than MAX_EXACT_SITES sites (the exact answer is a statevector)."""
    reasons = []
    if not isinstance(card, dict):
        raise CardRefused(["the card is not a JSON object"])
    if card.get("card") != CARD_VERSION:
        reasons.append(f"'card' must be {CARD_VERSION} (the format version)")
    for key in ("lattice", "hamiltonian", "initial", "time", "measure"):
        if key not in card:
            reasons.append(f"missing '{key}'")
    if reasons:
        raise CardRefused(reasons)
    n, edges = lattice(card["lattice"])
    if exact and n > MAX_EXACT_SITES:
        reasons.append(f"size: {n} sites; exact answers by statevector only up to {MAX_EXACT_SITES}")
    terms = _expand(card["hamiltonian"].get("terms", []), n, edges, "hamiltonian", reasons, need_group=True)
    if not terms and not reasons:
        reasons.append("hamiltonian: no terms")
    groups = {}
    for t in terms:
        groups.setdefault(t["group"], []).append(t)
    for g, ts in groups.items():   # first order inside a group would be ambiguous unless its terms commute
        bad = [(i, j) for i, j in itertools.combinations(range(len(ts)), 2) if not _pair_commute(ts[i], ts[j])]
        if bad:
            i, j = bad[0]
            reasons.append(f"group {g!r}: terms {ts[i]['pauli']}{list(ts[i]['sites'])} and "
                           f"{ts[j]['pauli']}{list(ts[j]['sites'])} do not commute (a group's terms must commute)")
    tm = card["time"]
    dt = _real(tm.get("dt"))
    if dt is None:
        reasons.append(f"time: dt {tm.get('dt')!r} is not a real number")
    steps = tm.get("steps")
    if not (_is_int(steps) and steps >= 0):
        reasons.append(f"time: steps {steps!r} is not a non-negative integer")
    if ("order" in tm) == ("orders" in tm):
        reasons.append("time: give exactly one of 'order' (every step) or 'orders' (a cycle of per-step orders)")
        orders = []
    else:
        orders = [tm["order"]] if "order" in tm else tm["orders"]
        if not (isinstance(orders, list) and orders and all(isinstance(o, list) for o in orders)):
            reasons.append("time: an order is a list of group names ('orders': a non-empty list of such lists)")
            orders = []
    for k, o in enumerate(orders):
        where = "time.order" if "order" in tm else f"time.orders[{k}]"
        unknown = [g for g in o if g not in groups]
        if unknown:
            reasons.append(f"{where}: unknown group(s) {unknown}")
        if sorted(map(str, o)) != sorted(groups):
            missing = sorted(set(groups) - set(o))
            twice = sorted({g for g in o if o.count(g) > 1})
            if missing or twice:
                reasons.append(f"{where}: every group exactly once per step (missing {missing}, more than once {twice})")
    init = card["initial"]
    state = None
    if isinstance(init, dict) and "bits" in init and "product" not in init:
        b = init["bits"]
        if isinstance(b, str) and len(b) == n and set(b) <= {"0", "1"}:
            state = b
        else:
            reasons.append(f"initial: bits must be a string of {n} characters 0/1 (character i = site i)")
    elif isinstance(init, dict) and "product" in init and "bits" not in init:
        p = init["product"]
        if isinstance(p, str) and len(p) == 1:
            p = p * n
        if isinstance(p, str) and len(p) == n and set(p) <= set(PRODUCT):
            state = p
        else:
            reasons.append(f"initial: product must be one character or {n} characters of {''.join(PRODUCT)}")
    else:
        reasons.append("initial: give exactly one of 'bits' or 'product'")
    meas = []
    for k, m in enumerate(card["measure"] if isinstance(card["measure"], list) else [None]):
        if not (isinstance(m, dict) and isinstance(m.get("op"), str)):
            reasons.append(f"measure {k}: needs 'op' (Z, X, Y, ZZ, ...) and 'sites'")
            continue
        p, s = m["op"], m.get("sites")
        if not (1 <= len(p) <= 2 and all(ch in PAULI for ch in p)):
            reasons.append(f"measure {k}: op {p!r} is not one or two letters of X, Y, Z")
            continue
        if s == "all" and len(p) == 1:
            lists = [(i,) for i in range(n)]
        elif s == "edges" and len(p) == 2:
            lists = [(a, b) for a, b, _ in edges]
        elif isinstance(s, (list, tuple)) and len(s) == len(p) and all(_is_int(v) for v in s):
            lists = [tuple(s)]
        else:
            reasons.append(f"measure {k}: sites {s!r} do not fit op {p}")
            continue
        for s in lists:
            if any(v < 0 or v >= n for v in s) or len(set(s)) != len(s):
                reasons.append(f"measure {k}: sites {list(s)} are not distinct sites 0 .. {n - 1}")
                continue
            meas.append((p, tuple(int(v) for v in s)))
    meas = list(dict.fromkeys(meas))
    if not meas and not reasons:
        reasons.append("measure: nothing to measure")
    conserved = []
    for k, q in enumerate(card.get("conserved", [])):
        nm = q.get("name", f"conserved {k}") if isinstance(q, dict) else f"conserved {k}"
        qt = _expand(q.get("terms", []) if isinstance(q, dict) else [], n, edges, f"conserved {nm!r}", reasons,
                     need_group=False, on_edges=False)
        conserved.append((nm, qt))
    if reasons:
        raise CardRefused(reasons)
    H = _op(terms, n)
    cons = []
    for nm, qt in conserved:   # the declared charge must commute with H; whether each group keeps it is reported
        C = _op(qt, n)
        c_h = _comm_norm(H, C)
        if c_h > COMMUTE_ATOL:
            reasons.append(f"conserved {nm!r}: does not commute with H (largest coefficient of [H, Q] {c_h:.3g})")
        per_group = {g: _comm_norm(_op(ts, n), C) for g, ts in groups.items()}
        cons.append(dict(name=nm, terms=len(qt), comm_H=c_h,
                         kept_by_every_group=all(v <= COMMUTE_ATOL for v in per_group.values()),
                         groups_not_keeping=sorted(g for g, v in per_group.items() if v > COMMUTE_ATOL)))
    if reasons:
        raise CardRefused(reasons)
    return dict(name=card.get("name", ""), n=n, edges=edges, terms=terms, groups=groups, dt=dt, steps=steps,
                orders=orders, state=state, initial_kind="bits" if "bits" in init else "product", measure=meas,
                conserved=cons)


def check(card, exact=True):
    """The entrance: a short report of what the card asks for, or CardRefused (nothing runs)."""
    P = parse(card, exact=exact)
    return dict(name=P["name"], sites=P["n"], edges=len(P["edges"]), terms=len(P["terms"]),
                groups={g: len(ts) for g, ts in P["groups"].items()}, dt=P["dt"], steps=P["steps"],
                orders=len(P["orders"]), initial=P["initial_kind"], observables=len(P["measure"]),
                settings=settings(P), conserved=P["conserved"])


def hamiltonian(card):
    """H as a SparsePauliOp on n qubits (qubit i = site i)."""
    P = card if "groups" in card and "terms" in card and "n" in card else parse(card, exact=False)
    return _op(P["terms"], P["n"])


# ---------------------------------------------------------------------------------------------------------------
# the logical circuit
# ---------------------------------------------------------------------------------------------------------------

def _to_z(qc, q, p):
    """Rotate qubit q so that Pauli p reads as Z (V p V^dagger = Z)."""
    if p == "X":
        qc.h(q)
    elif p == "Y":
        qc.sdg(q)
        qc.h(q)


def _from_z(qc, q, p):
    if p == "X":
        qc.h(q)
    elif p == "Y":
        qc.h(q)
        qc.s(q)


def zz_rotation(qc, a, q0, q1):
    """exp(-i a Z Z) on (q0, q1), up to a global phase: one CZ when a = +-pi/4 mod pi, else CZ RX CZ (two CZ)."""
    r = a % np.pi
    if abs(r - np.pi / 4) < 1e-12:
        qc.rz(np.pi / 2, q0)
        qc.rz(np.pi / 2, q1)
        qc.cz(q0, q1)
    elif abs(r - 3 * np.pi / 4) < 1e-12:
        qc.rz(-np.pi / 2, q0)
        qc.rz(-np.pi / 2, q1)
        qc.cz(q0, q1)
    elif min(r, np.pi - r) < 1e-15:
        return
    else:
        qc.h(q1)
        qc.cz(q0, q1)
        qc.rx(2 * a, q1)
        qc.cz(q0, q1)
        qc.h(q1)


def term_gate(qc, term, dt):
    """exp(-i dt c P) for one term (fixed decomposition, no search): one site RX/RY/RZ(2 dt c); two sites: each
    qubit turned to Z (X: H; Y: S^dagger H), exp(-i dt c Z Z), turned back."""
    a = dt * term["coef"]
    p, s = term["pauli"], term["sites"]
    if len(p) == 1:
        {"X": qc.rx, "Y": qc.ry, "Z": qc.rz}[p](2 * a, s[0])
        return
    for q, ch in zip(s, p):
        _to_z(qc, q, ch)
    zz_rotation(qc, a, s[0], s[1])
    for q, ch in zip(s, p):
        _from_z(qc, q, ch)


def prepare(qc, P):
    for i, ch in enumerate(P["state"]):
        if ch == "1":
            qc.x(i)
        elif ch == "+":
            qc.h(i)
        elif ch == "-":
            qc.x(i)
            qc.h(i)
        elif ch == "r":
            qc.h(i)
            qc.s(i)
        elif ch == "l":
            qc.h(i)
            qc.sdg(i)


_KAK = None


def pair_gate(qc, terms, dt):
    """exp(-i dt sum c P) for commuting terms on one pair of sites, as one two-qubit unitary decomposed analytically
    by the KAK (Weyl) decomposition into at most 3 CZ (qiskit TwoQubitBasisDecomposer, CZ basis, exact; nothing
    searched)."""
    global _KAK
    from qiskit.circuit.library import CZGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import TwoQubitBasisDecomposer
    from scipy.linalg import expm
    if _KAK is None:
        _KAK = TwoQubitBasisDecomposer(CZGate(), euler_basis="ZSX")
    lo, hi = sorted(terms[0]["sites"])
    loc = {lo: 0, hi: 1}
    h = SparsePauliOp.from_sparse_list([(t["pauli"], [loc[v] for v in t["sites"]], t["coef"]) for t in terms],
                                       num_qubits=2).to_matrix()
    sub = _KAK(expm(-1j * dt * h), approximate=False)
    qc.compose(sub, qubits=[lo, hi], inplace=True)


def _group_gates(qc, terms, dt, merge):
    """The gates of one group: term by term, or (merge) the two-site terms on one pair as one gate, placed where the
    pair's first term stands (the group's terms commute, so the rearrangement is exact); a pair with one term and the
    one-site terms keep the fixed decomposition."""
    if not merge:
        for t in terms:
            term_gate(qc, t, dt)
        return
    pairs = {}
    for t in terms:
        if len(t["sites"]) == 2:
            pairs.setdefault(frozenset(t["sites"]), []).append(t)
    done = set()
    for t in terms:
        if len(t["sites"]) == 1:
            term_gate(qc, t, dt)
            continue
        key = frozenset(t["sites"])
        if key in done:
            continue
        done.add(key)
        if len(pairs[key]) == 1:
            term_gate(qc, t, dt)
        else:
            pair_gate(qc, pairs[key], dt)


def circuit(card, steps=None, merge=False):
    """The logical first-order Trotter circuit (no measurement): the initial state, then per step the groups in the
    step's order (step s uses orders[s mod len(orders)]), each term of a group once, in the card's order.
    merge (option, 2026-11-10): commuting terms of a group on the same pair of sites become one two-qubit gate
    (`pair_gate`, at most 3 CZ); the default keeps the term-by-term decomposition."""
    from qiskit import QuantumCircuit
    P = card if "groups" in card and "orders" in card else parse(card)
    qc = QuantumCircuit(P["n"])
    prepare(qc, P)
    for s in range(P["steps"] if steps is None else steps):
        for g in P["orders"][s % len(P["orders"])]:
            _group_gates(qc, P["groups"][g], P["dt"], merge)
    return qc


def settings(P):
    """The bases to read in: a list of strings (one letter per site, '-' = not read), each observable in the first
    setting whose letters agree with it (greedy, in the card's order). Z-only cards read in one setting."""
    out, where = [], []
    for p, s in P["measure"]:
        for k, b in enumerate(out):
            if all(b[i] in ("-", ch) for i, ch in zip(s, p)):
                for i, ch in zip(s, p):
                    b[i] = ch
                where.append(k)
                break
        else:
            b = ["-"] * P["n"]
            for i, ch in zip(s, p):
                b[i] = ch
            out.append(b)
            where.append(len(out) - 1)
    return ["".join(b) for b in out]


def measured_circuits(card, merge=False):
    """[(basis, circuit)]: the Trotter circuit, every read site turned to its basis, measured into clbit = site."""
    from qiskit import ClassicalRegister
    P = card if "groups" in card and "orders" in card else parse(card)
    base = circuit(P, merge=merge)
    out = []
    for b in settings(P):
        qc = base.copy()
        qc.add_register(ClassicalRegister(P["n"], "c"))
        for i, ch in enumerate(b):
            if ch != "-":
                _to_z(qc, i, ch)
        for i, ch in enumerate(b):
            if ch != "-":
                qc.measure(i, i)
        out.append((b, qc))
    return out


def label(p, s):
    return " ".join(f"{ch}_{i}" for ch, i in zip(p, s))


# ---------------------------------------------------------------------------------------------------------------
# exact answers
# ---------------------------------------------------------------------------------------------------------------

def exact(card, method="auto", merge=False, info=None, **tn_options):
    """{label: exact expectation value} of the card's Trotter circuit, one per observable. method "statevector"
    (qiskit.quantum_info, up to about 20 sites), "aer" (qiskit-aer save_expectation_value, up to MAX_EXACT_SITES),
    "tn" (2026-11-11: psf_os.exact_tn, each observable's light cone as a tensor network; no site limit, refused by
    the memory guard instead: largest intermediate over tn_options["max_size"], default 2^27 elements), "auto"
    (statevector up to 20 sites, "tn" above). info: a list that receives exact_tn's report per observable."""
    from qiskit.quantum_info import SparsePauliOp
    if method not in ("auto", "statevector", "aer", "tn"):
        raise ValueError(f"method {method!r}")
    if "groups" in card and "orders" in card:
        P = card
    else:
        P = parse(card, exact=method in ("statevector", "aer"))
    n = P["n"]
    if method == "auto":
        method = "statevector" if n <= 20 else "tn"
    if method in ("statevector", "aer") and n > MAX_EXACT_SITES:
        raise CardRefused([f"size: {n} sites; exact answers by {method} only up to {MAX_EXACT_SITES}"])
    qc = circuit(P, merge=merge)
    if method == "tn":
        from psf_os import exact_tn
        try:
            vals, reps = exact_tn.expectations(qc, [(p, tuple(s)) for p, s in P["measure"]], info=True, **tn_options)
        except exact_tn.TooLarge as err:
            raise CardRefused([f"memory guard (tn): {err.reason}"])
        if info is not None:
            info.extend(reps)
        return {label(p, s): v for (p, s), v in zip(P["measure"], vals)}
    ops = [(label(p, s), SparsePauliOp.from_sparse_list([(p, list(s), 1.0)], num_qubits=n)) for p, s in P["measure"]]
    if method == "statevector":
        from qiskit.quantum_info import Statevector
        sv = Statevector(qc)
        return {lab: float(np.real(sv.expectation_value(op))) for lab, op in ops}
    from qiskit_aer import AerSimulator
    q2 = qc.copy()
    for k, (_, op) in enumerate(ops):
        q2.save_expectation_value(op, list(range(n)), label=f"e{k}")
    data = AerSimulator(method="statevector").run(q2).result().data()
    return {lab: float(np.real(data[f"e{k}"])) for k, (lab, _) in enumerate(ops)}


# ---------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser(prog="psf_os.card", description="problem cards: check, exact answers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="the entrance checks; exit 2 and the reasons when refused")
    c.add_argument("card")
    c.add_argument("--tn", action="store_true", help="no site limit (answers by the tensor network, method tn)")
    e = sub.add_parser("exact", help="exact expectation values of the card's Trotter circuit")
    e.add_argument("card")
    e.add_argument("--out", default=None, help="write the answers here (never overwritten)")
    e.add_argument("--merge", action="store_true", help="commuting terms on one pair as one gate (at most 3 CZ)")
    e.add_argument("--method", default="auto", choices=["auto", "statevector", "aer", "tn"],
                   help="auto: statevector up to 20 sites, tn above")
    a = ap.parse_args(argv)
    try:
        card = load(a.card)
        if a.cmd == "check":
            print(json.dumps(check(card, exact=not a.tn), indent=1))
            return 0
        ans = exact(card, method=a.method, merge=a.merge)
    except CardRefused as err:
        print("REFUSED")
        for r in err.reasons:
            print("  - " + r)
        return 2
    if a.out:
        if os.path.exists(a.out):
            raise SystemExit(f"STOP: {a.out} exists")
        with open(a.out, "w") as f:
            json.dump(ans, f, indent=1)
    for k, v in ans.items():
        print(f"{k}\t{v:.12f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
