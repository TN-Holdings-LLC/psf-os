"""One entrance for a logical circuit (2026-11-13; design: psf-os-lab-notebook notes/os_entrance_2026-11-13/DESIGN.md,
item 2 of the owner's OS gaps). The steps every builder of 2026-11-07 to 2026-11-12 repeated, done from the circuits
alone by fixed rules: recognise the family of the circuits' interaction graph, place it, hand the circuits to
psf-os's user entrance (user2d.build_batch: translation copy, compile, floor, mirror, ring sentinel), report the
cost, and score the answer card. Placement reads the coupling graph and the failed marks only (no error numbers, no
search such as SABRE); no mitigation; answers are written to their own file, outside the job.

    from psf_os import entrance as E
    rep = E.check(dev, items)                                   # family, placement, cost; nothing written
    rep = E.build(dev, items, out, answers=..., answers_out=..., options...)
    card = E.score(out, answers=..., twin_answers=...)

items = [(tag, QuantumCircuit), ...]: logical circuits, qubit k = logical site k, one- and two-qubit gates, barriers,
final measurements only (user2d.analyse refuses anything else). Items with one interaction graph G (their two-qubit pairs)
share one placement, as every builder did (arms x times x strings);
items with different graphs (e.g. L 20 and L 30 in one job) are placed group by group, as hub_build placed each L.

Families, tested in this order (the first that fits is used; `family=` forces one; a family that does not fit on the
day's record refuses with its reason and the entrance does not fall through to another family on its own):
  chain   G is a path. Logical order: from the path's smaller-index end. Placement: psf_os.chain.chain_of (the fixed
          shapes S1-S4 of the 11-09 dual-unitary rule first, then rule C), position i on labels[i]. For circuits
          written on 0..n-1 along the path this is exactly gpu/hub_build.py's on_chain(chain_of(dev, n)).
  ring    G is a cycle. Placement: psf_os.place.rule_choice(dev, n) when its outline is a plain cycle of exactly n
          (no pendant, no hole: those schedules are written for the ring classifier only); cycle position i on the
          logical cycle's i-th site from its smallest index towards its smaller neighbour. Otherwise refused.
  labels  every edge of G is a coupler of the Heron r2 graph: the circuit is already written on device labels and goes
          to user2d unchanged (its translation copy is user2d's rule).
  strip   G equals a honeycomb strip's bonds in the strip's own label order (psf_os.strip: rows 0..R-1, columns 0..c
          and the bridges inside; index k = the strip's k-th label). Other labellings of the same graph are not searched.
Refused now, with the reason: a 3D grid or a 2D square grid (G has 4-cycles; heavy-hex has none: 3D runs through
grid3d's kicked-Ising builder only, a generic 3D circuit is not built), degree above 3, and anything else.

Options (all recorded in build.json):
  mirror="compiled" (default)  the mirror is the compiled circuit, a barrier, its inverse (as hub_build's
                       _mirror_from_compiled, which every declared builder since 11-11 used), set for this build only
                       and restored after; "logical" keeps user2d's own mirror.
  sentinel_fallback=None | "same_qubits"   psf_os.sentinel's fallback (off by default).
  twin=False | True    user2d's Clifford twin per circuit; build then writes the twins' exact answers (stabilizer).
  special=None | "quirk_ledger"   the placement candidates (the qualifying translation copies of the family's pattern,
                       in lattice2d's fixed order) are ranked by psf_os.quirk.rank(dev.name, candidates, ledger, before)
                       and the first is pinned (user2d then uses that copy only). A special condition: it is written
                       to build.json as such; None (the default) is "special": null and changes nothing.
"""
from __future__ import annotations

import contextlib
import json
import os

import numpy as np

FAMILIES = ("chain", "ring", "labels", "strip")
SPECIALS = (None, "quirk_ledger")
MIRRORS = ("compiled", "logical")
READ_LINE = 1 - 1 / np.sqrt(2)   # r at which a pure shrink reaches MSE = mean square / 2


class Refused(ValueError):
    """The circuits cannot go through the entrance as given; the message says why."""


# ------------------------------------------------------------------------------------------------ the graph

def interaction(items):
    """(nodes, edges) of the union of the items' two-qubit pairs; nodes = every qubit an operation touches."""
    nodes, edges = set(), set()
    for tag, qc in items:
        for ins in qc.data:
            name = ins.operation.name
            qs = [qc.find_bit(q).index for q in ins.qubits]
            if name in ("barrier", "measure"):
                continue
            if len(qs) > 2:
                raise Refused(f"{tag}: '{name}' acts on {len(qs)} qubits (one- and two-qubit gates only)")
            nodes.update(qs)
            if len(qs) == 2:
                edges.add(tuple(sorted(qs)))
    return nodes, edges


def _adj(nodes, edges):
    a = {q: set() for q in nodes}
    for x, y in edges:
        a[x].add(y)
        a[y].add(x)
    return a


def _connected(nodes, a):
    if not nodes:
        return False
    seen, todo = set(), [min(nodes)]
    while todo:
        q = todo.pop()
        if q not in seen:
            seen.add(q)
            todo.extend(a[q] - seen)
    return seen == set(nodes)


def path_order(nodes, edges):
    """The path's sites from its smaller-index end, or None if G is not a path over every node."""
    a = _adj(nodes, edges)
    if len(nodes) < 2 or len(edges) != len(nodes) - 1 or max(len(v) for v in a.values()) > 2 \
            or not _connected(nodes, a):
        return None
    start = min(q for q in nodes if len(a[q]) == 1)
    order, prev = [start], None
    while len(order) < len(nodes):
        nxt = [y for y in a[order[-1]] if y != prev]
        prev = order[-1]
        order.append(nxt[0])
    return order


def cycle_order(nodes, edges):
    """The cycle's sites from its smallest index towards its smaller neighbour, or None if G is not one cycle."""
    a = _adj(nodes, edges)
    if len(nodes) < 3 or len(edges) != len(nodes) or any(len(v) != 2 for v in a.values()) or not _connected(nodes, a):
        return None
    order = [min(nodes), min(a[min(nodes)])]
    while len(order) < len(nodes):
        order.append(next(y for y in a[order[-1]] if y != order[-2]))
    return order


def _has_4cycle(a):
    for x in a:
        for y in a:
            if y > x and len(a[x] & a[y]) >= 2:
                return True
    return False


def recognise(dev, items, family="auto"):
    """dict(family, order | phys, tests): which family the items' graph G belongs to, by the fixed tests in order."""
    from psf_os import strip as S
    if family != "auto" and family not in FAMILIES:
        raise Refused(f"family {family!r} is not one of {FAMILIES} (or 'auto')")
    nodes, edges = interaction(items)
    if not edges:
        raise Refused("no two-qubit gate: nothing to place by the graph (use user2d directly on device labels)")
    a = _adj(nodes, edges)
    tests = {}
    want = FAMILIES if family == "auto" else (family,)
    for fam in want:
        if fam == "chain":
            o = path_order(nodes, edges)
            tests["chain"] = o is not None
            if o is not None:
                return dict(family="chain", order=o, n=len(o), tests=tests)
        elif fam == "ring":
            o = cycle_order(nodes, edges)
            tests["ring"] = o is not None
            if o is not None:
                return dict(family="ring", order=o, n=len(o), tests=tests)
        elif fam == "labels":
            ok = all(e in dev.all_couplers() for e in edges) and max(nodes) < dev.num_qubits
            tests["labels"] = ok
            if ok:
                return dict(family="labels", n=len(nodes), tests=tests)
        elif fam == "strip":
            got = S.find(dev.all_couplers(), edges, len(nodes)) if set(nodes) == set(range(len(nodes))) else None
            tests["strip"] = got is not None
            if got is not None:
                R, c, phys = got
                return dict(family="strip", R=R, c=c, phys=phys, n=len(phys), tests=tests)
    deg = max(len(v) for v in a.values())
    if _has_4cycle(a):
        why = ("G has 4-cycles (a 2D square or 3D grid): heavy-hex has none; not native now (3D runs through grid3d's "
               "kicked-Ising builder only; a generic 3D circuit through the fold schedule is not built)")
    elif deg > 3:
        why = f"G has a site of degree {deg}: heavy-hex has degree 3 at most"
    elif not _connected(nodes, a):
        why = "G is not connected (place each part as its own job)"
    else:
        why = "G is none of path, cycle, Heron-label subgraph, honeycomb strip (in its label order)"
    raise Refused(f"refused: {why}; tests {tests}")


# ------------------------------------------------------------------------------------------------ placement

def placement(dev, rec):
    """{logical site: device label} for the recognised family, and how it was chosen."""
    from psf_os import chain as CH
    fam = rec["family"]
    if fam == "chain":
        try:
            labels, how = CH.chain_of(dev, rec["n"])
        except Exception as e:  # rule C's own refusal (e.g. no usable chain of n)
            raise Refused(f"chain of {rec['n']}: {e}") from None
        return {q: labels[i] for i, q in enumerate(rec["order"])}, dict(how, labels=labels)
    if fam == "ring":
        from psf_os import place as P
        try:
            rc = P.rule_choice(dev, rec["n"])
        except Exception as e:
            raise Refused(f"ring of {rec['n']}: {e}") from None
        if rc["pend"] or rc["m"] != rec["n"]:
            raise Refused(f"ring of {rec['n']}: the rule's outline is {rc['kind']} (m {rc['m']}, pendants "
                          f"{len(rc['pend'])}); pendant and hole schedules are written for the ring classifier only")
        cyc = rc["cycle"]
        return {q: cyc[i] for i, q in enumerate(rec["order"])}, dict(rule="ring", shape=rc["shape"], cycle=cyc)
    if fam == "labels":
        return None, dict(rule="labels")
    if fam == "strip":
        return {k: p for k, p in enumerate(rec["phys"])}, dict(rule="strip", R=rec["R"], c=rec["c"], phys=rec["phys"])
    raise Refused(fam)


def on_labels(qc, mapping, width):
    """The circuit with logical site k on device label mapping[k] (every instruction kept, barriers included), as
    gpu/hub_build.on_chain."""
    from qiskit import QuantumCircuit
    if mapping is None:
        if qc.num_qubits == width and not qc.clbits:
            return qc
        out = QuantumCircuit(width)
        for ins in qc.data:
            if ins.operation.name != "measure":
                out.append(ins.operation, [qc.find_bit(q).index for q in ins.qubits])
        return out
    out = QuantumCircuit(width)
    for ins in qc.data:
        if ins.operation.name == "measure":
            continue
        out.append(ins.operation, [mapping[qc.find_bit(q).index] for q in ins.qubits])
    return out


@contextlib.contextmanager
def compiled_mirror(dev, on=True):
    """For this build only: user2d's mirror = the compiled circuit, a barrier, its inverse, Z readout (as
    gpu/hub_build._mirror_from_compiled). Restored on exit."""
    from psf_os import mirror2d as M
    if not on:
        yield
        return
    from qiskit import QuantumCircuit
    from psf_os.compile import RuleCompiler
    rc = RuleCompiler(dev)

    def mirror(u, phys):
        body = rc.pm.run(u)
        out = QuantumCircuit(body.num_qubits, len(phys))
        out.compose(body, qubits=list(range(body.num_qubits)), inplace=True)
        out.barrier()
        out.compose(body.inverse(), qubits=list(range(body.num_qubits)), inplace=True)
        for k, q in enumerate(phys):
            out.measure(q, k)
        return out
    old = M.mirror
    M.mirror = mirror
    try:
        yield
    finally:
        M.mirror = old


@contextlib.contextmanager
def pinned_copy(on=True):
    """For this build only: user2d's translation uses the copy at shift (0, 0) only (the special condition's choice).
    Restored on exit."""
    from psf_os import lattice2d as L
    if not on:
        yield
        return
    old = L.place

    def place(dev, edges, qubits=None, offsets=None):
        return old(dev, edges, qubits, offsets=[(0, 0)])
    L.place = place
    try:
        yield
    finally:
        L.place = old


def candidates(dev, mapping, edges):
    """The qualifying translation copies of the placed pattern, in lattice2d's fixed order, as quirk candidates."""
    from psf_os import lattice2d as L
    e2 = [(mapping[a], mapping[b]) for a, b in edges] if mapping else list(edges)
    qs = sorted({q for e in e2 for q in e}) if mapping is None else sorted(mapping.values())
    out = []
    for dr, dc, m in L.copies(dev, e2, qs):
        if not L.check(dev, e2, m):
            out.append(dict(name=f"dr{dr}_dc{dc}", qubits=[m[q] for q in qs], shift=(dr, dc), map=m))
    return out


def apply_special(dev, special, mapping, edges, ledger=None, before=None, quirk=None):
    """(mapping, record): the special condition's choice. quirk: the module (default psf_os.quirk, the VS's)."""
    if special is None:
        return mapping, None
    if special not in SPECIALS:
        raise Refused(f"special {special!r} is not one of {SPECIALS}")
    if quirk is None:
        try:
            from psf_os import quirk
        except ImportError:
            raise Refused("special 'quirk_ledger': psf_os.quirk is not present (the VS's piece)") from None
    cs = candidates(dev, mapping, edges)
    if not cs:
        raise Refused("special 'quirk_ledger': no qualifying copy to rank")
    ordered, report = quirk.rank(dev.name, [dict(name=c["name"], qubits=c["qubits"]) for c in cs], ledger, before)
    first = next(c for c in cs if c["name"] == ordered[0]["name"])
    m = first["map"]
    new = {k: m[v] for k, v in mapping.items()} if mapping else {q: m[q] for q in m}
    return new, dict(name=special, chosen=first["name"], first_by_rule=cs[0]["name"],
                     candidates=[c["name"] for c in cs], report=report)


# ------------------------------------------------------------------------------------------------ cost

def two_qubit_layers(qc):
    return qc.depth(filter_function=lambda i: len(i.qubits) == 2 and i.operation.name not in ("barrier", "measure"))


def cz_layers(qc):
    return qc.depth(filter_function=lambda i: i.operation.name == "cz")


def boundary_hint(path, family_prefix=None):
    """From a boundary table (notebook notes/boundary_*/boundary.json): the most CZ layers that read and the fewest
    that did not, over the rows with CZ layers (and, if given, results starting with family_prefix). Not a verdict."""
    if not path or not os.path.exists(path):
        return None
    rows = [r for r in json.load(open(path))["rows"] if r.get("cz_layers") is not None
            and (family_prefix is None or r["result"].startswith(family_prefix))]
    read = [r["cz_layers"] for r in rows if "DOES NOT READ" not in r["verdict"] and "READ" in r["verdict"]]
    not_read = [r["cz_layers"] for r in rows if "DOES NOT READ" in r["verdict"]]
    return dict(source=os.path.basename(os.path.dirname(path)), prefix=family_prefix, rows=len(rows),
                read_up_to_layers=max(read) if read else None, first_not_read_layers=min(not_read) if not_read else None)


def cost(dev, items, placed, shots, max_seconds, twin=False, boundary=None, boundary_prefix=None, pinned=False):
    """Per item: CZ, CZ layers, logical two-qubit layers, inflation, CZ per qubit, qubits; the job's seconds by the
    time model (the ring sentinel's circuits included once the job is built: see build's qpu_seconds_model)."""
    from psf_os import job as J
    from psf_os import user2d as U
    from psf_os.compile import RuleCompiler, rzz_exact
    from psf_os import lattice2d as L
    from psf_os import mirror2d as M
    rc = RuleCompiler(dev)
    per = {}
    for tag, qc in items:
        u2 = placed[tag]
        labels, edges, logical = U.analyse(u2, dev)
        dr, dc, m, _ = L.place(dev, edges, labels, offsets=[(0, 0)] if pinned else None)
        phys = [m[q] for q in labels]
        cu = rc.pm.run(U.with_measure(rzz_exact(U.relabel(logical, m, dev.num_qubits)), phys))
        cz, lay, l2 = M.cz_count(cu), cz_layers(cu), two_qubit_layers(qc)
        per[tag] = dict(cz=cz, cz_layers=lay, logical_2q_layers=l2, inflation=(lay / l2) if l2 else None,
                        qubits=len(labels), cz_per_qubit=cz / len(labels))
    n_c = len(items) * (4 if twin else 3)
    sec = J.qpu_seconds(n_c, shots)
    return dict(items=per, circuits_without_sentinel=n_c, seconds_without_sentinel=round(sec, 2),
                fits=sec <= max_seconds, max_seconds=max_seconds,
                boundary_hint=boundary_hint(boundary, boundary_prefix))


# ------------------------------------------------------------------------------------------------ check / build

def groups(items):
    """Items grouped by their own interaction graph, in the order of first appearance: [(key, [tags])]."""
    out = {}
    for tag, qc in items:
        nodes, edges = interaction([(tag, qc)])
        out.setdefault((tuple(sorted(nodes)), tuple(sorted(edges))), []).append(tag)
    return list(out.items())


def _prepare(dev, items, family, special, ledger, before, quirk):
    """Each group of items with one interaction graph is recognised and placed on its own (as hub_build placed each L
    on its own chain). Returns (groups report, {tag: mapping}, {tag: circuit on device labels})."""
    by = dict(items)
    reps, maps, placed = [], {}, {}
    for (nodes, edges), tags in groups(items):
        sub = [(t, by[t]) for t in tags]
        rec = recognise(dev, sub, family)
        mapping, how = placement(dev, rec)
        mapping, sp = apply_special(dev, special, mapping, list(edges), ledger, before, quirk)
        reps.append(dict(tags=tags, n=len(nodes), family=rec["family"], tests=rec["tests"], placement=how, special=sp,
                         mapping=None if mapping is None else {str(k): v for k, v in mapping.items()}))
        for t in tags:
            maps[t] = mapping
            placed[t] = on_labels(by[t], mapping, dev.num_qubits)
    return reps, maps, placed


def check(dev, items, family="auto", shots=4000, max_seconds=160, twin=False, special=None, ledger=None, before=None,
          quirk=None, boundary=None, boundary_prefix=None):
    """Family, placement and cost before anything is written."""
    reps, maps, placed = _prepare(dev, items, family, special, ledger, before, quirk)
    return dict(device=dev.name, groups=reps, special=special,
                cost=cost(dev, items, placed, shots, max_seconds, twin, boundary, boundary_prefix,
                          pinned=special is not None))


def build(dev, items, out, answers=None, answers_out=None, shots=4000, max_seconds=160, family="auto",
          mirror="compiled", sentinel_fallback=None, twin=False, special=None, ledger=None, before=None, quirk=None,
          boundary=None, boundary_prefix=None):
    """Recognise, place, build the job (user2d.build_batch) and write out/build.json. answers: {tag: [exact <Z> per
    logical site]} (optional), written by device label to answers_out (outside the job); twin answers (if twin) to
    answers_out's name + '_twin.json' or out/twin_answers.json."""
    from psf_os import sentinel
    from psf_os import user2d as U
    if mirror not in MIRRORS:
        raise Refused(f"mirror {mirror!r} is not one of {MIRRORS}")
    if sentinel_fallback not in sentinel.FALLBACKS:
        raise Refused(f"sentinel_fallback {sentinel_fallback!r} is not one of {sentinel.FALLBACKS}")
    for p in (out, answers_out):
        if p and os.path.exists(p):
            raise Refused(f"{p} exists (nothing is overwritten)")
    reps, maps, placed = _prepare(dev, items, family, special, ledger, before, quirk)
    spec = dict(sentinel.SENTINEL_JOB, fallback=sentinel_fallback) if sentinel_fallback else None
    with compiled_mirror(dev, mirror == "compiled"), pinned_copy(special is not None):
        rep = U.build_batch(dev, [(t, placed[t]) for t, _ in items], out, shots=shots, max_seconds=max_seconds,
                            sentinel_spec=spec, **(dict(twin=True) if twin else {}))
    runs = os.path.join(out, "runs")
    from qiskit import qpy
    per = {}
    for tag, qc in items:
        cu = qpy.load(open(os.path.join(runs, f"p2d_{tag}.qpy"), "rb"))[0]
        l2, lay = two_qubit_layers(qc), cz_layers(cu)
        it = rep["items"][tag]
        per[tag] = dict(cz=it["cz"], cz_layers=lay, logical_2q_layers=l2, inflation=(lay / l2) if l2 else None,
                        qubits=len(it["labels"]), cz_per_qubit=it["cz"] / len(it["labels"]), copy=it["copy"])
    fb = {}
    if sentinel_fallback:
        import glob
        used = any("fallback" in json.load(open(f)) for f in glob.glob(os.path.join(runs, "ksent_*.json")))
        fb = dict(asked=sentinel_fallback, used=bool(used))
    if answers is not None and answers_out:
        by_label = {t: {str(maps[t][k] if maps[t] is not None else k): float(v) for k, v in enumerate(vals)}
                    for t, vals in answers.items()}
        json.dump(by_label, open(answers_out, "w"), indent=1, sort_keys=True)
    tw_path = None
    if twin:
        base = answers_out[:-5] if answers_out and answers_out.endswith(".json") else os.path.join(out, "answers")
        tw_path = base + "_twin.json"
        twin_answers(out, [t for t, _ in items], tw_path)
    b = dict(device=dev.name, entrance="psf_os.entrance 2026-11-13", groups=reps, special=special, mirror=mirror, sentinel_fallback=fb or None, twin=bool(twin), twin_answers=tw_path,
             cost=dict(items=per, circuits=rep["circuits"], qpu_seconds_model=rep["qpu_seconds_model"],
                       fits=rep["qpu_seconds_model"] <= max_seconds, max_seconds=max_seconds,
                       boundary_hint=boundary_hint(boundary, boundary_prefix)),
             shots=shots, circuits=rep["circuits"], qpu_seconds_model=rep["qpu_seconds_model"])
    json.dump(b, open(os.path.join(out, "build.json"), "w"), indent=1)
    return b


def twin_answers(out, tags, path):
    """The twins' exact per-qubit <Z> by label (psf_os.twin.z_exact), as the notebook's gpu/twin_util."""
    import ast
    from qiskit import qpy
    from psf_os import twin as TW
    runs = os.path.join(out, "runs")
    res = {}
    for tag in tags:
        rec = json.load(open(os.path.join(runs, f"p2d_{tag}_twin.json")))
        m = rec["mapping"] if isinstance(rec["mapping"], dict) else ast.literal_eval(rec["mapping"])
        ct = qpy.load(open(os.path.join(runs, f"p2d_{tag}_twin.qpy"), "rb"))[0]
        z = TW.z_exact(ct, [int(m[str(q)] if str(q) in m else m[q]) for q in rec["labels"]])
        res[tag] = {str(q): float(v) for q, v in zip(rec["labels"], z)}
    if os.path.exists(path):
        raise Refused(f"{path} exists")
    json.dump(res, open(path, "w"), indent=1, sort_keys=True)
    return res


# ------------------------------------------------------------------------------------------------ score

def verdict(z, e):
    """READS (MSE <= mean square / 2), AGREES count (|z - e| <= 0.05), r = sum z e / sum e^2 and the split
    MSE / mean square = (1 - r)^2 + rest (notebook notes/paper9/r_thread_2026-11-13)."""
    z, e = np.asarray(z, float), np.asarray(e, float)
    mse, ms = float(np.mean((z - e) ** 2)), float(np.mean(e ** 2))
    r = float(z @ e / (e @ e)) if e @ e > 0 else None
    ratio = mse / ms if ms > 0 else None
    shrink = (1 - r) ** 2 if r is not None else None
    return dict(mse=mse, mean_square=ms, ratio=ratio, reads="READS" if ms > 0 and mse <= ms / 2 else "DOES NOT READ",
                agrees=int(np.sum(np.abs(z - e) <= 0.05)), of=len(z), r=r, shrink=shrink,
                rest=(ratio - shrink) if ratio is not None and shrink is not None else None)


def _twin_z(read, tag):
    from psf_os import user2d as U
    from psf_os.kim import z_values
    for name, (rec, cc) in read.items():
        if rec.get("psfos_job", {}).get("kind") == "user2d_twin" and U._tag_of(name, rec) == tag:
            z, _ = z_values(cc[(rec["points"][0]["input"], "z")], len(rec["labels"]))
            return rec["labels"], np.asarray(z)
    return None, None


CAVEAT = ("r_twin >= 0.293 says the device would read if its error on this circuit were a pure shrink; where the "
          "error also bends the pattern (Hubbard U = 4 at 6 steps: r 0.38-0.42 and did not read) it over-promises")


def score(out, answers=None, twin_answers=None):
    """{tag: card}: user2d's card plus, with answers ({tag: {label: e}}), READS / AGREES / r / split; with the twins'
    answers, r_twin and whether it reaches 1 - 1/sqrt(2), with the caveat."""
    from psf_os import job as J
    from psf_os import user2d as U
    job, runs = os.path.join(out, "job"), os.path.join(out, "runs")
    answers = answers or {}
    cards = U.cards(job, runs, answers)
    read = J.read(job, runs) if twin_answers else None
    res = {}
    for tag, c in cards.items():
        q = {x["q"]: x for x in c["qubits"]}
        row = dict(card=c)
        if tag in answers:
            labs = [lab for lab in c["labels"] if "exact" in q[lab]]
            row["verdict"] = verdict([q[lab]["z"] for lab in labs], [q[lab]["exact"] for lab in labs])
        if twin_answers and tag in twin_answers:
            labs, zt = _twin_z(read, tag)
            et = np.array([twin_answers[tag][str(lab)] for lab in labs])
            v = verdict(zt, et)
            row["twin"] = dict(v, r_twin=v["r"], reaches_line=(v["r"] is not None and v["r"] >= READ_LINE),
                               line=READ_LINE, caveat=CAVEAT)
        res[tag] = row
    return res
