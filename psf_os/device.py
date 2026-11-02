"""L0: the device as psf-os sees it. Its coupling graph, its basis and the elements marked as failed, nothing else.

No quality number (T1, T2, gate error, readout error, gate length) is read into psf-os or stored. The one place that
touches a Target's error field is `failed_elements`, and only to see the failed mark: error >= 1 or no entry, which
is how IBM marks an element that must not be used (ported from TOQB toqb/devices.py `_failed` and toqb/fold.py
`bad` at 8d44f05). Everything after that uses the Device record below, and `map_free_target` rebuilds a Qiskit Target
from it with no error and no duration, so that the compiler cannot read a map either (test T6).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field

FAILED_ERROR = 1.0
BASIS_1Q = ("sx", "x", "rz")


@dataclass
class Device:
    name: str
    num_qubits: int
    couplers: list            # every CZ coupler of the device, [a, b] with a < b (the geometry)
    failed_couplers: list     # couplers that must not be used (failed on the device, or masked by the registry)
    failed_qubits: list       # qubits whose sx or measurement is failed
    fake: bool = False
    fetched: str = ""
    masked: list = field(default_factory=list)   # couplers masked by the registry (also in failed_couplers)
    source: dict = field(default_factory=dict)

    # -- views used by the placement
    def working_couplers(self):
        bad = {tuple(e) for e in self.failed_couplers}
        return {tuple(e) for e in self.couplers if tuple(e) not in bad}

    def all_couplers(self):
        return {tuple(e) for e in self.couplers}

    def qubit_ok(self, q):
        return q not in set(self.failed_qubits)

    # -- files
    def to_json(self):
        return asdict(self)

    def save(self, folder):
        os.makedirs(folder, exist_ok=False)
        json.dump(self.to_json(), open(os.path.join(folder, "device.json"), "w"), indent=1)
        return folder

    @staticmethod
    def load(folder):
        return Device(**json.load(open(os.path.join(folder, "device.json"))))

    def with_mask(self, couplers, reason=""):
        """A copy with these couplers marked failed (the registry's mask)."""
        m = sorted({tuple(sorted(e)) for e in couplers})
        d = Device(**self.to_json())
        d.failed_couplers = sorted({tuple(e) for e in self.failed_couplers} | set(m))
        d.failed_couplers = [list(e) for e in d.failed_couplers]
        d.masked = [list(e) for e in sorted({tuple(e) for e in self.masked} | set(m))]
        d.source = dict(self.source, mask_reason=reason)
        return d


def failed_elements(target):
    """(failed couplers, failed qubits) of a Qiskit Target by IBM's failed mark only (error >= 1, or no entry)."""
    def bad(name, qa):
        p = target[name].get(qa) if name in target.operation_names else None
        return p is None or (p.error is not None and p.error >= FAILED_ERROR)
    all_c = sorted({tuple(sorted(qa)) for qa in target["cz"] if qa is not None})
    failed_c = sorted({tuple(sorted(qa)) for qa in target["cz"] if qa is not None and bad("cz", qa)})
    failed_q = sorted(q for q in range(target.num_qubits) if bad("sx", (q,)) or bad("measure", (q,)))
    return all_c, failed_c, failed_q


def from_target(target, name, fake=False, source=None):
    all_c, failed_c, failed_q = failed_elements(target)
    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    return Device(name=name, num_qubits=target.num_qubits, couplers=[list(e) for e in all_c],
                  failed_couplers=[list(e) for e in failed_c], failed_qubits=list(failed_q), fake=fake, fetched=now,
                  source=dict(source or {}))


def fetch_fake(fake_name):
    """A fake device of qiskit-ibm-runtime (FakeFez, FakeMarrakesh, FakeKingston, ...)."""
    from qiskit_ibm_runtime import fake_provider
    b = getattr(fake_provider, fake_name)()
    return from_target(b.target, b.name, fake=True, source=dict(kind="fake", cls=fake_name))


def map_free_target(dev):
    """A Qiskit Target with the device's working elements only and no error or duration anywhere."""
    from qiskit.circuit import Delay, Measure, Parameter
    from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
    from qiskit.transpiler import Target
    t = Target(num_qubits=dev.num_qubits)
    ok = [q for q in range(dev.num_qubits) if dev.qubit_ok(q)]
    for gate in (SXGate(), XGate()):
        t.add_instruction(gate, {(q,): None for q in ok})
    t.add_instruction(RZGate(Parameter("theta")), {(q,): None for q in ok})
    t.add_instruction(Measure(), {(q,): None for q in ok})
    t.add_instruction(Delay(Parameter("t")), {(q,): None for q in ok})
    cz = {}
    for a, b in sorted(dev.working_couplers()):
        if dev.qubit_ok(a) and dev.qubit_ok(b):
            cz[(a, b)] = None
            cz[(b, a)] = None
    t.add_instruction(CZGate(), cz)
    return t
