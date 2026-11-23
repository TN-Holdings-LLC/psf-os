"""L0: the device as psf-os sees it. Its coupling graph, its basis and the elements marked as failed, nothing else.

No quality number (T1, T2, gate error, readout error, gate length) is read into psf-os or stored. The one place that
touches a Target's error field is `failed_elements`, and only to see the failed mark: error >= 1 or no entry, which
is how IBM marks an element that must not be used (ported from TOQB toqb/devices.py `_failed` and toqb/fold.py
`bad` at 8d44f05). Everything after that uses the Device record below, and `map_free_target` rebuilds a Qiskit Target
from it with no error and no duration, so that the compiler cannot read a map either (test T6).

0.3.2 (candidate): the record also holds `dt`, the backend configuration's clock step in seconds (backend.dt, passed
to from_target by the fetch: from_target itself still reads failed marks only, test T6). It is a unit of time, the
same for every qubit, not a calibration or quality number: it converts delays between dt and seconds and prices them
in the time model (job.assemble). A record without it (every record before 0.3.2) loads and
builds as before; its JSON is written without the key, so old records stay byte for byte.

0.3.3rc1 (candidate; the owner's decision of 2026-11-23): the record may also hold `timing`, read by the fetch with
`timing_of` from the backend's Target: the duration of x, sx, cz, measure and reset per qubit or coupler, in dt, and
the timing constraints (granularity, min_length, pulse_alignment, acquire_alignment). These are units and lengths of
time, which the owner allowed to be read; no error rate, no T1, T2 or frequency is read (test T6 still guards
from_target, which reads failed marks only; test_checks_durations guards timing_of). The compiler never sees them
(map_free_target has no duration): they are used to refuse a delay that does not fit the device's time grid and to
estimate, beside the job, how long a circuit and its waits last. A record without it builds as before and its JSON
is written without the key.
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
    dt: float | None = None   # the clock step in seconds (backend configuration), a unit of time, not a quality number
    timing: dict | None = None  # 0.3.3rc1: durations (dt) and timing constraints, time values only (timing_of)

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
        d = asdict(self)
        if d.get("dt") is None:
            d.pop("dt", None)     # records without a dt keep the JSON they had before 0.3.2
        if d.get("timing") is None:
            d.pop("timing", None)  # records without timing keep the JSON they had before 0.3.3rc1
        return d

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


def from_target(target, name, fake=False, source=None, step=None, timing=None):
    """The device record from a Target: failed marks only. step, the clock step dt (0.3.2), is passed in by the
    caller, read from the backend (backend.dt); timing (0.3.3rc1), timing_of's record, likewise. This function reads
    nothing else of the Target (test T6 guards it)."""
    all_c, failed_c, failed_q = failed_elements(target)
    from zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("Europe/Zurich")).isoformat(timespec="seconds")  # records are in CET (owner)
    return Device(name=name, num_qubits=target.num_qubits, couplers=[list(e) for e in all_c],
                  failed_couplers=[list(e) for e in failed_c], failed_qubits=list(failed_q), fake=fake, fetched=now,
                  source=dict(source or {}), dt=float(step) if step else None, timing=timing or None)


TIMING_GATES = ("x", "sx", "cz", "measure", "reset")
TIMING_CONSTRAINTS = ("granularity", "min_length", "pulse_alignment", "acquire_alignment")


def timing_of(target, step):
    """The time values of a Target (0.3.3rc1; the owner allowed lengths of time to be read, never an error number):
    dict(durations={gate: {"q" or "a,b": length in dt}}, constraints={granularity, min_length, pulse_alignment,
    acquire_alignment} in dt). Only an instruction property's `duration` is read, and only for TIMING_GATES; a duration
    that is not a whole number of dt (within 1e-6) is left out and named in `not_whole`. None without a dt."""
    if not step:
        return None
    durations, not_whole = {}, []
    names = set(target.operation_names)
    for g in TIMING_GATES:
        if g not in names:
            continue
        per = {}
        for qa, props in target[g].items():
            if qa is None or props is None:
                continue
            d = props.duration                       # a length of time (seconds); the error field is never read
            if d is None:
                continue
            k = float(d) / float(step)
            key = ",".join(str(q) for q in qa)
            if abs(k - round(k)) > 1e-6:
                not_whole.append(f"{g}[{key}]")
                continue
            per[key] = int(round(k))
        if per:
            durations[g] = per
    constraints = {}
    for c in TIMING_CONSTRAINTS:
        v = getattr(target, c, None)
        if v is not None:
            constraints[c] = int(v)
    out = dict(durations=durations, constraints=constraints, unit="dt")
    if not_whole:
        out["not_whole"] = sorted(not_whole)
    return out


def fetch_fake(fake_name):
    """A fake device of qiskit-ibm-runtime (FakeFez, FakeMarrakesh, FakeKingston, ...)."""
    from qiskit_ibm_runtime import fake_provider
    b = getattr(fake_provider, fake_name)()
    return from_target(b.target, b.name, fake=True, source=dict(kind="fake", cls=fake_name),
                       step=getattr(b, "dt", None), timing=timing_of(b.target, getattr(b, "dt", None)))


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
