"""L6 for 2D: the user's own circuit in, a job with its controls out, and the answer card back.

The user writes a logical circuit on Heron r2 qubit labels (QPY or OpenQASM 2/3): circuit qubit index = label, every
two-qubit gate on an edge of the heavy-hex lattice, measurements (if any) only at the end. psf-os then
  1. places it by translation (`lattice2d.place`: the first copy, in the fixed order, with no failed, masked or suspect
     element; no search), or refuses with the reasons;
  2. compiles it with the rule's map-free pass manager (RZZ(+-pi/2) first written exactly as one CZ and two RZ,
     compile.rzz_exact) and adds, in the same job:
     - the floor: the compiled circuit with every non-diagonal single-qubit gate removed (sx, x; what is left is CZ and
       RZ, so the ideal <Z_q> is 1 and the CZ count is the user's): readout and idle loss, for any circuit;
     - the mirror sentinel (`mirror2d.mirror`: U, barrier, U^dagger; acceptance: exactly twice U's CZ count);
     - the ring sentinel (echo D = 24 on the rule's n = 40 ring);
  3. scores the result as an answer card: the magnetisation and each <Z_q> with 2 SE of shot noise, the floor, the
     mirror flags, the sentinel's verdict, and, if the user gives answers, the difference and the difference in SE.
Every qubit the circuit touches is measured in Z, in label order (clbit k = k-th label).
"""
from __future__ import annotations

import json
import os

import numpy as np

from psf_os import lattice2d as L
from psf_os import mirror2d as M

NON_DIAGONAL_1Q = ("sx", "sxdg", "x", "y", "h", "rx", "ry", "u", "u1", "u2", "u3")  # after compilation only sx, x occur
REFUSED = ("reset", "initialize", "delay_until", "if_else", "while_loop", "for_loop", "switch_case")


class Refused(ValueError):
    """The circuit cannot be run as given; the message says why."""


def load(path):
    """A QuantumCircuit from a .qpy (its first circuit) or an OpenQASM file (2 or 3)."""
    if path.endswith(".qpy"):
        from qiskit import qpy
        with open(path, "rb") as f:
            return qpy.load(f)[0]
    text = open(path).read()
    if "OPENQASM 3" in text:
        from qiskit import qasm3
        return qasm3.loads(text)
    from qiskit import qasm2
    return qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def analyse(qc, dev):
    """(labels, edges, logical circuit without measurements) or Refused. Edges are checked on the device's graph."""
    graph = {tuple(sorted(e)) for e in dev.all_couplers()}
    used, edges, seen_measure = set(), [], set()
    for ins in qc.data:
        name = ins.operation.name
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if name in REFUSED or getattr(ins.operation, "condition", None):
            raise Refused(f"'{name}' is not read (final measurements only, no resets or classical control)")
        if name == "barrier":
            continue
        if name == "measure":
            seen_measure.update(qs)
            continue
        if seen_measure & set(qs):
            raise Refused(f"'{name}' on qubit {qs} after its measurement (only final measurements are read)")
        if len(qs) > 2:
            raise Refused(f"'{name}' acts on {len(qs)} qubits; write it with one- and two-qubit gates")
        used.update(qs)
        if len(qs) == 2:
            e = tuple(sorted(qs))
            if e not in graph:
                raise Refused(f"'{name}' on ({e[0]}, {e[1]}): not an edge of the heavy-hex lattice")
            if e not in edges:
                edges.append(e)
    if not used:
        raise Refused("the circuit has no gates")
    logical = qc.remove_final_measurements(inplace=False)
    return sorted(used), edges, logical


def relabel(logical, mapping, width):
    from qiskit import QuantumCircuit
    out = QuantumCircuit(width)
    for ins in logical.data:
        if ins.operation.name == "measure":
            continue
        out.append(ins.operation, [out.qubits[mapping[logical.find_bit(q).index]] for q in ins.qubits])
    return out


def with_measure(c, phys):
    from qiskit import QuantumCircuit
    out = QuantumCircuit(c.num_qubits, len(phys))
    out.compose(c, inplace=True)
    for k, p in enumerate(phys):
        out.measure(p, k)
    return out


def floor_of(compiled):
    """The compiled circuit without its non-diagonal single-qubit gates (CZ, RZ, barriers, measurements stay)."""
    out = compiled.copy_empty_like()
    for ins in compiled.data:
        if len(ins.qubits) == 1 and ins.operation.name in NON_DIAGONAL_1Q:
            continue
        out.append(ins.operation, ins.qubits, ins.clbits)
    return out


def _record(dev, runs, name, role, kind, circuits, points, labels, mapping, extra, shots, valid):
    from qiskit import qpy
    from psf_os.compile import VERSION
    from psf_os.registry import sha256_file
    for k, c in enumerate(circuits):
        c.name = f"{name}-{k}"
    jf, qf = os.path.join(runs, name + ".json"), os.path.join(runs, name + ".qpy")
    for f in (jf, qf):
        if os.path.exists(f):
            raise FileExistsError(f"{f} exists: a run never overwrites another")
    with open(qf, "wb") as f:
        qpy.dump(circuits, f)
    rec = dict(kind="psfos", psfos_role=role, psfos_job=dict(kind=kind), arm="patch:copy", version=VERSION,
               device=dev.name, masked=dev.masked, labels=labels, mapping={str(q): mapping[q] for q in labels},
               points=points, valid=bool(valid), shots=shots,
               export=dict(qpy=os.path.basename(qf), circuits=len(circuits), sha256=sha256_file(qf)), **extra)
    json.dump(rec, open(jf, "w"), indent=1)
    return rec


def _tag_ok(tag):
    return bool(tag) and all(ch.isalnum() or ch in "_.-" for ch in tag)


def build_batch(dev, items, out, shots=4000, ring=40, sentinel_spec=None, max_seconds=60):
    """Several user circuits in one job: items = [(tag, QuantumCircuit), ...]. Each circuit is placed by translation on
    its own (its own copy), compiled, and given its own floor and mirror; one ring sentinel serves the whole job. If any
    circuit is refused, nothing is written for any of them. Returns {"items": {tag: report}, "circuits", ...}."""
    import numpy as _np
    from psf_os import circuits as C
    from psf_os import job as J
    from psf_os import sentinel
    from psf_os.compile import RuleCompiler, measured_qubits, rzz_exact
    tags = [t for t, _ in items]
    if not items:
        raise Refused("no circuit given")
    if len(set(tags)) != len(tags) or not all(_tag_ok(t) for t in tags):
        raise Refused(f"tags must be distinct and use letters, digits, '_', '.', '-' only: {tags}")
    rc = RuleCompiler(dev)
    want_all, prepared = {}, []
    for tag, qc in items:
        try:
            labels, edges, logical = analyse(qc, dev)
        except Refused as e:
            raise Refused(f"{tag}: {e}") from None
        try:
            dr, dc, m, reasons = L.place(dev, edges, labels)
        except RuntimeError as e:
            raise Refused(f"{tag}: no copy of the circuit's pattern qualifies on {dev.name}: {e}") from None
        phys = [m[q] for q in labels]
        u = rzz_exact(relabel(logical, m, dev.num_qubits))  # RZZ(+-pi/2): one CZ, not two
        cu = rc.pm.run(with_measure(u, phys))
        cz_u = M.cz_count(cu)
        cf = floor_of(cu)
        cm = rc.pm.run(M.mirror(u, phys))
        want = {k: p for k, p in enumerate(phys)}
        checks = dict(user_measures=measured_qubits(cu) == want, floor_cz=M.cz_count(cf) == cz_u,
                      floor_diagonal=all(i.operation.name in ("cz", "rz", "barrier", "measure", "delay")
                                         for i in cf.data),
                      mirror_cz=M.cz_count(cm) == 2 * cz_u, mirror_measures=measured_qubits(cm) == want)
        if not all(checks.values()):
            raise Refused(f"{tag}: acceptance failed: {checks}")
        prepared.append((tag, labels, edges, dr, dc, m, reasons, cu, cf, cm, cz_u, checks))
    runs = os.path.join(out, "runs")
    os.makedirs(runs, exist_ok=True)
    reps = {}
    for tag, labels, edges, dr, dc, m, reasons, cu, cf, cm, cz_u, checks in prepared:
        common = dict(labels=labels, mapping=m, shots=shots)
        copy = dict(copy=dict(drow=dr, dcol=dc, passed_over=reasons), edges=[list(e) for e in edges])
        _record(dev, runs, f"p2d_{tag}", "user", "user2d", [cu], [dict(input="u", setting="z")],
                extra=dict(copy, cz=cz_u, tag=tag), valid=True, **common)
        _record(dev, runs, f"p2d_{tag}_floor", "floor", "user2d_floor", [cf], [dict(input="f", setting="z")],
                extra=dict(cz=M.cz_count(cf), tag=tag), valid=True, **common)
        _record(dev, runs, f"p2d_{tag}_mirror", "mirror", "user2d_mirror", [cm], [dict(input="m", setting="z")],
                extra=dict(cz=M.cz_count(cm), cz_expected=2 * cz_u, tag=tag), valid=True, **common)
        reps[tag] = dict(labels=labels, edges=[list(e) for e in edges], copy=dict(drow=dr, dcol=dc),
                         passed_over=reasons, mapping={str(q): m[q] for q in labels}, cz=cz_u, checks=checks)
    mq = measured_qubits(rc.compile(C.model_circuit(list(_np.zeros(C.n_model_angles(ring, 1))), ring, 1)))
    ring_q = [int(mq[j]) for j in range(ring)]
    sentinel.build(dev, runs, ring_q, shots, rc, spec=sentinel_spec)
    man = J.assemble(dev, runs, os.path.join(out, "job"), max_seconds=max_seconds)
    rep = dict(device=dev.name, items=reps, tags=tags, circuits=len(man["pubs"]),
               qpu_seconds_model=man["qpu_seconds_model"], job=os.path.join(out, "job"), runs=runs)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    return rep


def build(dev, qc, out, shots=4000, ring=40, sentinel_spec=None, max_seconds=60, tag="user"):
    """Place, compile and assemble the user's job with floor, mirror and ring sentinel. Returns a report dict."""
    b = build_batch(dev, [(tag, qc)], out, shots=shots, ring=ring, sentinel_spec=sentinel_spec,
                    max_seconds=max_seconds)
    rep = dict(device=b["device"], **b["items"][tag], circuits=b["circuits"],
               qpu_seconds_model=b["qpu_seconds_model"], job=b["job"], runs=b["runs"])
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    return rep


def _tag_of(name, rec):
    if "tag" in rec:
        return rec["tag"]
    base = name[len("p2d_"):]
    for suf in ("_floor", "_mirror"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def tags_of(job_dir, runs):
    """The tags of the user circuits in a finished job, in the job's order."""
    from psf_os import job as J
    return [_tag_of(n, r) for n, (r, _) in J.read(job_dir, runs).items()
            if r.get("psfos_job", {}).get("kind") == "user2d"]


def cards(job_dir, runs, answers_by_tag=None):
    """{tag: answer card} for every user circuit of the job. answers_by_tag: {tag: {label: exact <Z>}} (any subset)."""
    answers_by_tag = answers_by_tag or {}
    return {t: card(job_dir, runs, answers_by_tag.get(t), tag=t) for t in tags_of(job_dir, runs)}


def card(job_dir, runs, answers=None, tag=None):
    """The answer card of one user circuit. answers: {label: exact <Z>} (any subset) or None. tag: which circuit of a
    batch job (None: the job must hold exactly one)."""
    from psf_os import job as J
    from psf_os import sentinel
    from psf_os.kim import z_values
    got = {}
    labels = edges = None
    read = J.read(job_dir, runs)
    if tag is None:
        ts = {_tag_of(n, r) for n, (r, _) in read.items() if r.get("psfos_job", {}).get("kind") == "user2d"}
        if len(ts) != 1:
            raise ValueError(f"the job holds {len(ts)} user circuits {sorted(ts)}: name one with tag=")
        tag = ts.pop()
    for name, (rec, cc) in read.items():
        kind = rec.get("psfos_job", {}).get("kind")
        if kind in ("user2d", "user2d_floor", "user2d_mirror") and _tag_of(name, rec) == tag:
            p = rec["points"][0]
            z, se = z_values(cc[(p["input"], "z")], len(rec["labels"]))
            got[kind] = (z, se)
            labels = rec["labels"]
            if kind == "user2d":
                edges = [tuple(e) for e in rec["edges"]]
    (z, se), (z0, _), (zm, _) = got["user2d"], got["user2d_floor"], got["user2d_mirror"]
    n = len(labels)
    fl = M.flags(z0, zm, labels=labels, edges=edges)
    out = dict(tag=tag, sentinel=sentinel.verdict(job_dir, runs)["status"], labels=labels,
               M=float(z.mean()), M_two_se=float(2 * np.sqrt((se ** 2).sum()) / n),
               floor_mean=float(z0.mean()), mirror_mean=float(zm.mean()), mirror_flags=fl,
               qubits=[dict(q=q, z=float(z[k]), two_se=float(2 * se[k]), floor=float(z0[k]), mirror=float(zm[k]))
                       for k, q in enumerate(labels)])
    if answers:
        idx = [k for k, q in enumerate(labels) if str(q) in answers or q in answers]
        a = np.array([answers.get(str(labels[k]), answers.get(labels[k])) for k in idx], dtype=float)
        mz = z[idx]
        two_se = float(2 * np.sqrt((se[idx] ** 2).sum()) / len(idx))
        diff = float(mz.mean() - a.mean())
        out["answered"] = dict(qubits=len(idx), M=float(mz.mean()), two_se=two_se, M_exact=float(a.mean()),
                               difference=diff, difference_in_se=diff / (two_se / 2) if two_se else None,
                               mse=float(np.mean((mz - a) ** 2)))
        for k, v in zip(idx, a):
            out["qubits"][k]["exact"] = float(v)
    return out
