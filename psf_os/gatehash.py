"""A circuit hash that does not depend on Qiskit's QPY format (W4).

`circuits_sha256` in a job manifest is the SHA-256 of the QPY bytes; QPY's encoding changes between Qiskit versions
(checked 2026-11-05: rebuilt circuits had the same gates, other bytes). This hash is taken over the gate list itself:
for every instruction, (name, qubit indices, clbit indices, parameters rounded to 1e-10), in order; circuits in file
order; written as canonical JSON (sorted keys, no spaces) and hashed with SHA-256. Barriers are kept (they fix which
gates run together). The circuit's name and metadata are not part of it.
"""
from __future__ import annotations

import hashlib
import json

ROUND = 10


def _param(p):
    try:
        v = float(p)
    except (TypeError, ValueError):
        return str(p)
    v = round(v, ROUND)
    return 0.0 if v == 0 else v  # -0.0 and 0.0 alike


def gate_list(circuit):
    out = []
    for ins in circuit.data:
        out.append([ins.operation.name,
                    [circuit.find_bit(q).index for q in ins.qubits],
                    [circuit.find_bit(c).index for c in ins.clbits],
                    [_param(p) for p in ins.operation.params]])
    return dict(num_qubits=circuit.num_qubits, num_clbits=circuit.num_clbits, gates=out)


def circuits_hash(circuits):
    """SHA-256 (hex) of the canonical JSON of [gate_list(c) for c in circuits]."""
    doc = [gate_list(c) for c in circuits]
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def qpy_hash(path):
    """dict(circuits=n, gate_sha256=...) of every circuit in a QPY file."""
    from qiskit import qpy
    with open(path, "rb") as f:
        cs = qpy.load(f)
    return dict(circuits=len(cs), gate_sha256=circuits_hash(cs))


# 2026-11-19 (psf-os v0.2, the loop's verify across machines): a digest that opens the conditional blocks (gate_list
# writes a block as str(), a text drawing that changes with the qiskit version) and writes every RZ angle in
# (-pi, pi] (RZ(t) and RZ(t + 2 pi) are the same pulse; two machines' compilers were seen to split a free angle
# differently on 2026-11-18). circuits_hash above is unchanged.

def _wrap(t):
    import math
    t = math.remainder(t, 2 * math.pi)
    return math.pi if abs(t + math.pi) < 1e-9 else t


def _gates_open(circuit, qmap=None, cmap=None):
    qmap = qmap or {q: circuit.find_bit(q).index for q in circuit.qubits}
    cmap = cmap or {c: circuit.find_bit(c).index for c in circuit.clbits}
    out = []
    for ins in circuit.data:
        op = ins.operation
        qs = [qmap[q] for q in ins.qubits]
        cs = [cmap[c] for c in ins.clbits]
        if op.name == "if_else":
            tgt, val = op.condition
            bits = [cmap[b] for b in tgt] if type(tgt).__name__ == "ClassicalRegister" else [cmap[tgt]]
            blocks = []
            for blk in op.blocks:
                if blk is None:
                    blocks.append(None)
                    continue
                blocks.append(_gates_open(blk, {bq: qs[i] for i, bq in enumerate(blk.qubits)},
                                          {bc: cs[i] for i, bc in enumerate(blk.clbits)}))
            out.append(["if_else", qs, cs, [bits, int(val)], blocks])
        elif op.name == "rz":
            out.append(["rz", qs, cs, [_param(_wrap(float(op.params[0])))]])
        else:
            out.append([op.name, qs, cs, [_param(p) for p in op.params]])
    return dict(num_qubits=circuit.num_qubits, num_clbits=circuit.num_clbits, gates=out)


def circuits_digest(circuits):
    """SHA-256 (hex) of the canonical JSON of every circuit's gates, conditional blocks opened, RZ in (-pi, pi]:
    comparable across machines and qiskit / numpy versions (parameters to 1e-10)."""
    doc = [_gates_open(c) for c in circuits]
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def qpy_digest(path):
    from qiskit import qpy
    with open(path, "rb") as f:
        return circuits_digest(qpy.load(f))
