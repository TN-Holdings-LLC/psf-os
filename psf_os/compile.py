"""The rule's compiler on a Device: placement (psf_os.place.rule_choice), the hole or pendant schedule, then Qiskit's
level-1 pass manager with the layout fixed and no routing, on the map-free Target (no error, no duration).

Ported from TOQB toqb/fold.py FoldLoop (modes rule; template: the angles become parameters, the schedule is built
and transpiled once per structure, the values are bound per call) at 8d44f05. TOQB transpiles on the fetched Target;
here the Target carries the working elements only, so nothing in the pass manager can read a quality number.
"""
from __future__ import annotations

import numpy as np

from psf_os import circuits as C
from psf_os import place as P
from psf_os.device import map_free_target

VERSION = "psf-os rule compiler (TOQB fold:rule at 8d44f05, map-free Target)"


class RuleCompiler:
    def __init__(self, dev):
        from qiskit.transpiler import generate_preset_pass_manager
        self.dev = dev
        self.target = map_free_target(dev)
        self.pm = generate_preset_pass_manager(1, target=self.target, initial_layout=list(range(dev.num_qubits)),
                                               routing_method="none", seed_transpiler=0)
        self._cache = {}
        self.last = {}

    @staticmethod
    def _sig(qc):
        return tuple((i.operation.name, tuple(qc.find_bit(q).index for q in i.qubits),
                      tuple(qc.find_bit(c).index for c in i.clbits)) for i in qc.data)

    def _template(self, qc):
        from qiskit.circuit import ParameterVector
        values = [float(p) for i in qc.data for p in i.operation.params]
        pv = ParameterVector("v", len(values))
        par = qc.copy_empty_like()
        k = 0
        for ins in qc.data:
            op = ins.operation
            if op.params:
                op = op.to_mutable()
                op.params = [pv[k + j] for j in range(len(op.params))]
                k += len(op.params)
            par.append(op, ins.qubits, ins.clbits)
        choice = P.rule_choice(self.dev, qc.num_qubits)
        if choice["kind"] == "pend":
            hc, init, fin = P.pend_circuit(par, choice["cycle"], choice["pend"], self.dev.num_qubits)
        else:
            hc, init, fin = P.hole_circuit(par, choice["cycle"], self.dev.num_qubits)
        tq = self.pm.run(hc)
        return tq, [pv[j] for j in range(len(values))], init, fin, choice

    def compile(self, qc):
        sig = self._sig(qc)
        if sig not in self._cache:
            self._cache[sig] = self._template(qc)
        tq, order, init, fin, choice = self._cache[sig]
        values = [float(p) for i in qc.data for p in i.operation.params]
        present = set(tq.parameters)
        out = tq.assign_parameters({p: v for p, v in zip(order, values) if p in present})
        out.metadata = dict(psfos_initial=list(init), psfos_final=list(fin))
        self.last = dict(kind=choice["kind"], m=choice["m"], shape=str(choice["shape"]))
        return out

    def per_layer_q2(self, n, seed=0):
        """Two-qubit gates of one ring layer (a one-layer model circuit with generic angles, compiled)."""
        rng = np.random.default_rng(seed)
        ang = list(rng.uniform(-1, 1, C.n_model_angles(n, 1)))
        return C.q2_count(self.compile(C.model_circuit(ang, n, 1)))


def rzz_exact(qc, tol=1e-9):
    """The circuit with every RZZ(theta), theta = +-pi/2 (mod 2 pi), written as RZ(theta) x RZ(theta) then one CZ.

    e^{-i theta/2 ZZ} = CZ (RZ(theta) x RZ(theta)) up to a global phase for theta = +-pi/2 (both are diagonal; compare
    the four phases). The level-1 pass manager otherwise writes every RZZ with two CZs. Any other angle is left as it
    is. An identity, not a search: the same circuit comes out for the same input, and no device number is read."""
    out = qc.copy_empty_like()
    for ins in qc.data:
        op = ins.operation
        if op.name == "rzz" and not getattr(op, "condition", None):
            try:
                th = float(op.params[0])
            except TypeError:  # a free parameter: leave it
                th = None
            if th is not None:
                r = (th + np.pi) % (2 * np.pi) - np.pi  # in [-pi, pi)
                for s in (np.pi / 2, -np.pi / 2):
                    if abs(r - s) < tol:
                        a, b = ins.qubits
                        out.rz(s, a)
                        out.rz(s, b)
                        out.cz(a, b)
                        out.global_phase += s / 2 + np.pi * round((th - s) / (2 * np.pi))
                        break
                else:
                    out.append(op, ins.qubits, ins.clbits)
                continue
        out.append(op, ins.qubits, ins.clbits)
    return out


def measured_qubits(compiled):
    """{clbit: physical qubit} of a compiled circuit."""
    return {compiled.find_bit(i.clbits[0]).index: compiled.find_bit(i.qubits[0]).index
            for i in compiled.data if i.operation.name == "measure"}
