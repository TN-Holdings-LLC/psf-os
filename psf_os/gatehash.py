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
