"""Exact expectation values by a light-cone tensor network (2026-11-11, not declared; home's instruction of 11-11).

<0| U^dag P U |0> for a Pauli product P, built only from the gates in P's backward light cone: walking the circuit
from the end, a gate is kept when it touches a qubit already in the support, and then its qubits join the support;
every other gate cancels exactly between ket and bra. The network:
  ket: |0> on every qubit of the cone; each kept gate as a tensor; a diagonal gate (RZ, RZZ, CZ, ...) adds no new
       index (a weight on the qubits' current indices: hyper-edges); any other gate opens new indices;
  P:   one 2 x 2 tensor per site joining the final bra and ket indices (never a 2^k matrix); off P the two are one index;
  bra: the conjugate.
quimb's exact simplifications shrink the network first; the contraction order is searched with cotengra.
Two forms of each network are tried: gate by gate with hyper-edges (suits the kicked Ising / Eagle circuits), and
runs of gates on one pair fused into two-qubit blocks (suits circuits of many small gates, such as the heisE line
card); the one whose quick (greedy) order is smaller is searched further.

Memory guard: before any contraction, the largest intermediate tensor of the chosen order (in elements) is compared
with `max_size` (default 2^27, 2 GiB of complex128); with `slicing` the order is searched again with slices to fit,
otherwise (or if it still does not fit) TooLarge is raised with the reason and nothing is computed. A pairwise
contraction holds its two inputs and its output at once, so the peak of live intermediates is also checked
(`max_peak`, default 4 x max_size; on the workplace's 8 GB machine an order of width 2^27 was killed by the
out-of-memory handler). `max_cost` (optional) refuses an order whose estimated cost (flops) is larger.

Used as the "tn" method of card.exact and as an option beyond the statevector limits of kim.answer_cone,
ising2d.answer_cone and grid3d.exact. Placement, compilation, controls and scoring do not use this module.

    from psf_os import exact_tn
    exact_tn.expectations(qc, [("ZZ", (3, 4)), {0: "X", 7: "Y"}])      # -> [float, float]
    exact_tn.expectations(qc, [...], info=True)                          # -> ([...], [per-observable report])
"""
import math
import time

import numpy as np

DEFAULT_MAX_SIZE = 2 ** 27

_P = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
      "Y": np.array([[0, -1j], [1j, 0]], dtype=complex), "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
_SKIP = {"barrier", "delay", "measure"}


class TooLarge(Exception):
    """The contraction would exceed the memory guard (or the cost limit); nothing was computed."""

    def __init__(self, reason, report=None):
        super().__init__(reason)
        self.reason = reason
        self.report = report or {}


def _observable(o):
    """{qubit: letter} from ("XZ", (i, j)) or {i: "X"}; identities dropped."""
    if isinstance(o, dict):
        d = {int(q): str(c).upper() for q, c in o.items()}
    else:
        p, s = o
        if len(p) != len(s):
            raise ValueError(f"observable {o!r}: letters and sites differ in length")
        d = {int(q): c.upper() for c, q in zip(p, s)}
    for q, c in d.items():
        if c not in _P:
            raise ValueError(f"observable: letter {c!r} on {q}")
    return {q: c for q, c in d.items() if c != "I"}


def _gates(qc):
    """[(qubit indices, matrix, diagonal?)] of the circuit in order; final measurements and barriers skipped."""
    from qiskit.quantum_info import Operator
    out, cache = [], {}
    for ins in qc.data:
        op = ins.operation
        if op.name in _SKIP:
            continue
        if op.name == "reset" or getattr(op, "condition", None) is not None or ins.clbits:
            raise ValueError(f"exact_tn: operation {op.name!r} is not a unitary gate")
        qs = tuple(qc.find_bit(q).index for q in ins.qubits)
        key = (op.name, tuple(float(p) for p in op.params if isinstance(p, (int, float)))) \
            if all(isinstance(p, (int, float)) for p in op.params) else None
        if key is not None and key in cache and cache[key][0].shape[0] == 2 ** len(qs):
            m, diag = cache[key]
        else:
            m = np.asarray(Operator(op).data, dtype=complex)
            diag = bool(np.allclose(m, np.diag(np.diag(m)), atol=0.0))
            if key is not None:
                cache[key] = (m, diag)
        if np.allclose(m, np.eye(m.shape[0]) * m[0, 0], atol=1e-15) and abs(abs(m[0, 0]) - 1) < 1e-15:
            continue                                          # identity up to a global phase (cancels with the bra)
        out.append((qs, m, diag))
    return out


def cone_gates(gates, support):
    """The gates of the backward light cone of `support` (in circuit order) and the cone's qubits."""
    sup, keep = set(support), []
    for qs, m, diag in reversed(gates):
        if sup.intersection(qs):
            keep.append((qs, m, diag))
            sup.update(qs)
    keep.reverse()
    return keep, sorted(sup)


def network(gates, obs):
    """(arrays, inputs, size_dict, n_cone_qubits, n_cone_gates) of <0|U^dag P U|0> for P = obs ({qubit: letter})."""
    keep, qubits = cone_gates(gates, obs)
    arrays, inputs = [], []
    final = {}
    for side in ("k", "b"):
        cur = {q: f"{side}{q}_0" for q in qubits}
        cnt = dict.fromkeys(qubits, 0)
        for q in qubits:
            arrays.append(np.array([1.0 + 0j, 0.0]))
            inputs.append((cur[q],))
        for qs, m, diag in keep:
            k = len(qs)
            if side == "b":
                m = m.conj()
            # qiskit's matrix: row/col index = sum_j bit(qs[j]) 2^j, so the reshaped axes run qs[k-1] ... qs[0]
            if diag:
                arrays.append(np.diag(m).reshape([2] * k))
                inputs.append(tuple(cur[q] for q in reversed(qs)))
            else:
                old = [cur[q] for q in reversed(qs)]
                for q in qs:
                    cnt[q] += 1
                    cur[q] = f"{side}{q}_{cnt[q]}"
                new = [cur[q] for q in reversed(qs)]
                arrays.append(m.reshape([2] * (2 * k)))
                inputs.append(tuple(new + old))
        final[side] = cur
    ren = {}
    for q in qubits:
        if q in obs:
            arrays.append(_P[obs[q]])
            inputs.append((final["b"][q], final["k"][q]))
        else:
            ren[final["b"][q]] = final["k"][q]
    inputs = [tuple(ren.get(i, i) for i in ix) for ix in inputs]
    size = {i: 2 for ix in inputs for i in ix}
    return arrays, inputs, size, len(qubits), len(keep)


def _simplify(arrays, inputs):
    """quimb's exact simplifications (rank, diagonal, column, reduction, split; atol 1e-14): a light cone of 72 sites
    goes from about 1,400 tensors to about 230 and the best width from 25 to 21 (kicked Ising 4x6x3, t 4)."""
    import quimb.tensor as qtn
    tn = qtn.TensorNetwork([qtn.Tensor(a, inds=ix) for a, ix in zip(arrays, inputs)])
    tn = tn.full_simplify("ADCRS", output_inds=(), atol=1e-14, equalize_norms=False, progbar=False)
    arr = [t.data for t in tn]
    inp = [tuple(t.inds) for t in tn]
    return arr, inp, {i: 2 for ix in inp for i in ix}


def _search(inputs, size, max_size, slicing, search_time, quick=False):
    """Contraction order (cotengra). quick: greedy, a few repeats (to compare the two forms of a network); otherwise
    greedy for small networks, KaHyPar (if installed) for large ones (greedy alone found widths up to 35 where KaHyPar
    found 18 on the heisE line card's cones)."""
    import cotengra as ctg
    try:
        import kahypar  # noqa: F401
        big_methods = ["kahypar"]
    except ImportError:
        big_methods = ["greedy", "random-greedy"]

    def opt(target):
        small = quick or len(inputs) <= 80
        kw = dict(methods=["greedy"] if small else big_methods, max_repeats=8 if small else 32,
                  max_time=min(search_time, 3.0) if quick else search_time, minimize="combo", parallel=False,
                  progbar=False)
        if target:
            kw["slicing_opts"] = dict(target_size=max_size)
        return ctg.HyperOptimizer(**kw).search(inputs, (), size)
    tree = opt(False)
    # slicing a far too wide order (greedy at times returned width 2^59 or 2^114 on the heisE cones) made cotengra's
    # slice search itself run out of memory: slices are searched only within 2^12 of the guard
    if tree.max_size() > max_size and slicing and math.log2(tree.max_size()) <= math.log2(max_size) + 12:
        tree = opt(True)
    return tree


def _fuse(qc):
    """The circuit with every run of gates on one qubit pair merged into one two-qubit unitary (qiskit Collect2qBlocks
    + ConsolidateBlocks): for circuits of many small gates (the heisE line card: 3,032 gates -> 100 blocks)."""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    return PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(qc)


def prepare(qc, forms=("gates", "fused")):
    """{form: gate list}: "gates" (gate by gate, diagonal gates as hyper-edges) and "fused" (two-qubit blocks). Each
    observable uses the form whose quick contraction order is smaller."""
    out = {}
    if "gates" in forms:
        out["gates"] = _gates(qc)
    if "fused" in forms:
        out["fused"] = _gates(_fuse(qc.remove_final_measurements(inplace=False)))
    return out


def _contract(arrays, inputs, size, tree):
    """Contract with the tree's path, slice by slice (cotengra's tree.contract fails on these hyper-edge networks:
    'tuple index out of range', as home found for utility_tn; the path itself is fine)."""
    import itertools
    import cotengra as ctg
    path = tree.get_path()
    sl = list(tree.sliced_inds)
    if not sl:
        return complex(ctg.array_contract(arrays, inputs, (), size_dict=size, optimize=path))
    total = 0j
    for vals in itertools.product(range(2), repeat=len(sl)):
        fix = dict(zip(sl, vals))
        arr, inp = [], []
        for a, ix in zip(arrays, inputs):
            sel = tuple(fix[i] if i in fix else slice(None) for i in ix)
            arr.append(a[sel])
            inp.append(tuple(i for i in ix if i not in fix))
        total += complex(ctg.array_contract(arr, inp, (), size_dict=size, optimize=path))
    return total


def expectation(circuit, obs, max_size=DEFAULT_MAX_SIZE, slicing=True, max_cost=None, search_time=10.0,
                simplify=True, max_peak=None, forms=("gates", "fused")):
    """(value, report) for one observable. `circuit`: a QuantumCircuit, or prepare()'s dict (reused across
    observables). report: the form used, cone qubits / gates, tensors (before / after simplification), slices,
    largest intermediate (elements, log2), peak memory of the order (elements, log2), estimated cost (log10 flops),
    search and contraction seconds (search_s includes the simplification). max_peak (default 4 x max_size) also
    refuses an order whose live intermediates together exceed it. simplify=False keeps the raw network (checks)."""
    import cotengra as ctg  # noqa: F401
    pre = circuit if isinstance(circuit, dict) else prepare(circuit, forms)
    max_peak = 4 * max_size if max_peak is None else max_peak
    o = _observable(obs)
    rep = dict(observable={str(q): c for q, c in sorted(o.items())})
    if not o:
        return 1.0, dict(rep, cone_qubits=0, method="identity")
    t0 = time.time()
    cands = []
    for form, gates in pre.items():
        arrays, inputs, size, nq, ng = network(gates, o)
        r = dict(form=form, cone_qubits=nq, cone_gates=ng, tensors=len(arrays))
        if simplify:
            arrays, inputs, size = _simplify(arrays, inputs)
        r["tensors_simplified"] = len(arrays)
        if len(arrays) == 1 and not inputs[0]:
            v = complex(np.asarray(arrays[0]).reshape(()))
            rep.update(r, search_s=round(time.time() - t0, 2), slices=1, max_intermediate=1, peak=1,
                       log2_max_intermediate=0.0, log2_peak=0.0, log10_cost=0.0, contract_s=0.0,
                       imag=float(abs(v.imag)), method="tn")
            return float(v.real), rep
        tree = _search(inputs, size, max_size, False, search_time, quick=True)
        key = (tree.max_size(), tree.total_flops())
        cands.append((key, tree, arrays, inputs, size, r))
    cands.sort(key=lambda c: c[0])
    # the quick greedy order can misjudge the forms (weight-17, 5 steps: the fused form was taken at width 2^27 where
    # the gate form reaches 2^16), so a form whose searched order stays wider than 2^20 lets the next form be searched
    best = None
    for key, tree, arrays, inputs, size, r in cands:
        if len(arrays) > 80 or (tree.max_size() > max_size and slicing):
            full = _search(inputs, size, max_size, slicing, search_time)
            if (full.max_size(), full.total_flops()) < (tree.max_size(), tree.total_flops()):
                tree = full
        if best is None or (tree.max_size(), tree.total_flops()) < (best[0].max_size(), best[0].total_flops()):
            best = (tree, arrays, inputs, size, r)
        if best[0].max_size() <= 2 ** 20:
            break
    tree, arrays, inputs, size, r = best
    rep.update(r)
    big, peak = int(tree.max_size()), int(tree.peak_size())
    rep.update(search_s=round(time.time() - t0, 2), slices=int(tree.nslices), max_intermediate=big,
               log2_max_intermediate=round(math.log2(max(big, 1)), 2), peak=peak,
               log2_peak=round(math.log2(max(peak, 1)), 2),
               log10_cost=round(math.log10(max(float(tree.total_flops()), 1.0)), 2))
    if big > max_size:
        raise TooLarge(f"largest intermediate 2^{rep['log2_max_intermediate']} elements > guard "
                       f"2^{math.log2(max_size):.1f} (cone {rep['cone_qubits']} qubits, {rep['cone_gates']} gates)",
                       rep)
    if peak > max_peak:
        raise TooLarge(f"peak of live intermediates 2^{rep['log2_peak']} elements > guard 2^{math.log2(max_peak):.1f}",
                       rep)
    if max_cost is not None and float(tree.total_flops()) > max_cost:
        raise TooLarge(f"estimated cost 10^{rep['log10_cost']} flops > limit 10^{math.log10(max_cost):.1f}", rep)
    t1 = time.time()
    v = _contract(arrays, inputs, size, tree)
    rep.update(contract_s=round(time.time() - t1, 2), imag=float(abs(v.imag)), method="tn")
    return float(v.real), rep


def expectations(qc, observables, info=False, **kw):
    """[value per observable] of the circuit's final state (qc without mid-circuit measurements); with info=True
    also the per-observable reports. Raises TooLarge at the first observable over the guard."""
    pre = prepare(qc, kw.pop("forms", ("gates", "fused")))
    vals, reps = [], []
    for o in observables:
        v, r = expectation(pre, o, **kw)
        vals.append(v)
        reps.append(r)
    return (vals, reps) if info else vals


def z_sites(qc, sites, info=False, **kw):
    """<Z_q> for q in sites."""
    return expectations(qc, [{q: "Z"} for q in sites], info=info, **kw)
