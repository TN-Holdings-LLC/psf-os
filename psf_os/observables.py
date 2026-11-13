"""Observables beyond per-qubit Z (workplace VS Claude, 2026-11-13 g; new file, not yet used by the entrance).

The card so far reads only <Z_q> per qubit. This module handles Pauli strings: products of Z (ZZ correlations, the
weight-10 / weight-17 Z products of IBM 2023) and strings with X and Y. It gives the measured circuits (one per basis
setting), the expectation values and shot errors from counts, the exact values, and the verdict on a set of
observables with the card's definitions.

Observable: {"name": str, "pauli": P} where P is either
  - a string with one letter per circuit qubit, "I" / "X" / "Y" / "Z"; letter k is circuit qubit k (written left to
    right from qubit 0, NOT Qiskit's reversed label order), e.g. "ZIZ" on 3 qubits is Z_0 Z_2; or
  - a dict {qubit: "X" | "Y" | "Z"}; qubit is a circuit qubit index, or a label when `labels` ({label: index}) is
    given (e.g. physical labels of a compiled circuit).
An observable with no letter (all "I") is refused.

Settings: each observable goes into the first setting whose letters agree with it on every qubit it reads (greedy,
in the given order; as psf_os.card.settings). Z-only observables share one setting.

Measuring: build() copies the circuit and, per setting, adds only single-qubit rotations before the measurements
(X: H; Y: S^dagger then H; Z: nothing) and measures every read qubit into the clbit of the same index. No
two-qubit gate is added (checked). The circuit must not contain measurements already.

Values from counts: <P> = the mean over shots of the product of (1 - 2 b_q) over the observable's qubits, in its
setting's counts; shot error se = sqrt((1 - <P>^2) / shots) (each shot gives +1 or -1). For a single Z this is
exactly psf_os.kim.z_values, as the card reads it.

Exact values (exact()): "statevector" (qiskit.quantum_info, up to MAX_STATEVECTOR qubits), "stabilizer" (Clifford
circuits, any size), "tn" (psf_os.exact_tn, light cone, memory guard; needs quimb and cotengra). "auto": statevector
up to MAX_STATEVECTOR qubits, else stabilizer if the circuit is Clifford, else tn. The method used is returned.

Verdict (verdict()): psf_os.entrance.verdict, the card's definitions: READS (MSE <= mean square / 2), AGREES
(|value - exact| <= 0.05, counted), r = sum v e / sum e^2 and the split (1 - r)^2 + rest.

    from psf_os import observables as O
    obs = [{"name": "Z0Z1", "pauli": "ZZII"}, {"name": "X2", "pauli": {2: "X"}}, {"name": "Y0Y3", "pauli": "YIIY"}]
    plan = O.build(qc, obs)                    # {"settings": [...], "where": [...], "circuits": [(basis, circuit)]}
    counts = [run(c) for _, c in plan["circuits"]]          # one counts dict per setting, in that order
    vals = O.values(plan, counts)              # [{"name", "value", "se", "shots", "setting"}]
    ex, how = O.exact(qc, obs)                 # [exact values], "statevector" | "stabilizer" | "tn"
    v = O.verdict([x["value"] for x in vals], ex)   # {"reads", "agrees", "of", "r", "mse", "mean_square", ...}
"""
from __future__ import annotations

import numpy as np

MAX_STATEVECTOR = 20
LETTERS = "XYZ"


class Refused(ValueError):
    pass


def normalize(obs, n, labels=None):
    """{circuit qubit index: letter} of one observable (letters X, Y, Z only)."""
    p = obs["pauli"]
    if isinstance(p, str):
        if len(p) != n:
            raise Refused(f"{obs.get('name')}: string of {len(p)} letters for {n} qubits")
        if any(ch not in "IXYZ" for ch in p):
            raise Refused(f"{obs.get('name')}: letters must be I, X, Y, Z")
        out = {k: ch for k, ch in enumerate(p) if ch != "I"}
    elif isinstance(p, dict):
        out = {}
        for q, ch in p.items():
            if labels is not None:
                if q not in labels and str(q) not in labels:
                    raise Refused(f"{obs.get('name')}: label {q!r} is not in the circuit")
                k = labels[q] if q in labels else labels[str(q)]
            else:
                k = int(q)
            if not 0 <= k < n:
                raise Refused(f"{obs.get('name')}: qubit {k} outside 0..{n - 1}")
            if ch not in LETTERS:
                raise Refused(f"{obs.get('name')}: letter {ch!r} (X, Y or Z)")
            if k in out:
                raise Refused(f"{obs.get('name')}: qubit {k} given twice")
            out[k] = ch
    else:
        raise Refused(f"{obs.get('name')}: pauli must be a string or a dict")
    if not out:
        raise Refused(f"{obs.get('name')}: no letter other than I")
    return out


def settings(observables, n, labels=None):
    """(settings, where): settings as strings of n letters ('-' = not read); where[i] = the setting of observable i."""
    out, where = [], []
    for o in observables:
        s = normalize(o, n, labels)
        for k, b in enumerate(out):
            if all(b[q] in ("-", ch) for q, ch in s.items()):
                for q, ch in s.items():
                    b[q] = ch
                where.append(k)
                break
        else:
            b = ["-"] * n
            for q, ch in s.items():
                b[q] = ch
            out.append(b)
            where.append(len(out) - 1)
    return ["".join(b) for b in out], where


def _rotate(qc, q, ch):
    if ch == "X":
        qc.h(q)
    elif ch == "Y":
        qc.sdg(q)
        qc.h(q)


def build(qc, observables, labels=None):
    """{"settings", "where", "circuits": [(basis, circuit)], "observables"}: per setting a copy of qc with the basis
    rotations and the measurements of the read qubits (clbit k = circuit qubit k)."""
    from qiskit import ClassicalRegister
    if any(i.operation.name == "measure" for i in qc.data):
        raise Refused("the circuit already measures; give it without measurements")
    n = qc.num_qubits
    sets, where = settings(observables, n, labels)
    circs = []
    for b in sets:
        c = qc.copy()
        c.add_register(ClassicalRegister(n, "obs"))
        before = sum(1 for i in c.data if len(i.qubits) >= 2)
        for q, ch in enumerate(b):
            if ch != "-":
                _rotate(c, q, ch)
        for q, ch in enumerate(b):
            if ch != "-":
                c.measure(q, c.clbits[-n + q])
        assert sum(1 for i in c.data if len(i.qubits) >= 2) == before, "a two-qubit gate was added"
        circs.append((b, c))
    return dict(settings=sets, where=where, circuits=circs,
                observables=[dict(name=o.get("name"), support=normalize(o, n, labels)) for o in observables], n=n)


def _signs(counts, n):
    """(+-1 matrix [outcomes x n] for build()'s register, shot counts). build() adds its register last, so its clbit q
    is clbit (total - n + q) of the circuit: character -1 - (total - n + q) of the key without spaces."""
    keys = list(counts)
    rows = []
    for k in keys:
        b = k.replace(" ", "")
        off = len(b) - n
        if off < 0:
            raise Refused(f"count key {k!r} has fewer than {n} bits")
        rows.append([int(b[-1 - (off + q)]) for q in range(n)])
    bits = np.array(rows)
    return 1 - 2 * bits, np.array([counts[k] for k in keys], float)


def values(plan, counts_per_setting):
    """[{"name", "value", "se", "shots", "setting"}] from one counts dict per setting (in plan["settings"] order).
    Count keys as Qiskit writes them (clbit 0 rightmost; spaces between registers ignored); reads build()'s register."""
    if len(counts_per_setting) != len(plan["settings"]):
        raise Refused(f"{len(counts_per_setting)} counts for {len(plan['settings'])} settings")
    n = plan["n"]
    sg = [_signs(c, n) for c in counts_per_setting]
    out = []
    for o, k in zip(plan["observables"], plan["where"]):
        s, w = sg[k]
        prod = np.prod(s[:, sorted(o["support"])], axis=1)
        shots = float(w.sum())
        v = float(prod @ w / shots)
        out.append(dict(name=o["name"], value=v, se=float(np.sqrt(max(1 - v * v, 0.0) / shots)), shots=int(shots),
                        setting=k))
    return out


def _qiskit_label(support, n):
    lab = ["I"] * n
    for q, ch in support.items():
        lab[n - 1 - q] = ch
    return "".join(lab)


def _is_clifford(qc):
    from qiskit.quantum_info import Clifford
    try:
        Clifford(qc)
        return True
    except Exception:  # noqa: BLE001
        return False


def exact(qc, observables, method="auto", labels=None, **tn_options):
    """([exact value per observable], method used) for qc's final state (qc without measurements)."""
    from qiskit.quantum_info import Pauli
    if method not in ("auto", "statevector", "stabilizer", "tn"):
        raise ValueError(f"method {method!r}")
    n = qc.num_qubits
    sup = [normalize(o, n, labels) for o in observables]
    c = qc.remove_final_measurements(inplace=False)
    if method == "auto":
        method = "statevector" if n <= MAX_STATEVECTOR else ("stabilizer" if _is_clifford(c) else "tn")
    if method == "statevector":
        if n > MAX_STATEVECTOR:
            raise Refused(f"statevector only up to {MAX_STATEVECTOR} qubits; {n} given")
        from qiskit.quantum_info import Statevector
        sv = Statevector(c)
        return [float(np.real(sv.expectation_value(Pauli(_qiskit_label(s, n))))) for s in sup], method
    if method == "stabilizer":
        from qiskit.quantum_info import StabilizerState
        if not _is_clifford(c):
            raise Refused("stabilizer: the circuit is not Clifford")
        st = StabilizerState(c)
        return [float(np.real(st.expectation_value(Pauli(_qiskit_label(s, n))))) for s in sup], method
    from psf_os import exact_tn
    try:
        vals = exact_tn.expectations(c, [dict(s) for s in sup], **tn_options)
    except exact_tn.TooLarge as err:
        raise Refused(f"memory guard (tn): {err.reason}") from None
    return [float(v) for v in vals], method


def verdict(vals, exact_vals):
    """The card's verdict over a set of observables (psf_os.entrance.verdict)."""
    from psf_os.entrance import verdict as V
    return V(vals, exact_vals)
