"""0.3.3rc1 (candidate; the owner's decision of 2026-11-23): the candidate's name, the parking checked in the compiled
circuit, the device's lengths of time read without any error number, delays on the device's time grid, the fences'
cost and the circuits' summaries in the manifest, and the paired loop the default on the device."""
import json
import os

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

import psf_os
from psf_os import compiled as CP
from psf_os import device as Dv
from psf_os import feedforward as FF
from psf_os import job as J
from psf_os import loop as LP
from psf_os.footprint import fence_waits, park_likely, park_likely_plan, park_plan, park_waits

SPEC = dict(kind="echo_clifford", D=24)


def _ipe_like():
    """q0 the ancilla, q1 the target in |1>, read at the end (as tests/test_park_default.py)."""
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(3, "c"))
    qc.x(1)
    for r in range(2):
        qc.h(0)
        qc.cz(0, 1)
        qc.h(0)
        qc.measure(0, r)
        with qc.if_test((qc.clbits[r], 1)):
            qc.x(0)
    qc.measure(1, 2)
    return qc


def _likely():
    """q1 in |1> with probability sin^2(1.2), about 0.87, entangled with nothing; q0 read and reset by a
    conditional."""
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
    qc.ry(2.4, 1)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    qc.measure(1, 1)
    return qc


# ------------------------------------------------------------------------------------------------ the name

def test_the_candidate_is_named_rc_and_compared_by_pep440():
    assert psf_os.__version__ == "0.3.3" and not FF.is_candidate(psf_os.__version__) and FF.is_candidate("0.3.3rc1")
    assert FF.version_key("0.3.3rc1") < FF.version_key("0.3.3") < FF.version_key("0.3.4rc1")
    assert FF.same_version("v0.3.3rc1", "0.3.3rc1") and not FF.same_version("0.3.3rc1", "0.3.3")
    assert not FF.same_version("0.3.3rc1", "0.3.3rc2")
    assert FF.parks_by_default("0.3.3rc1") and FF.parks_likely_by_default("0.3.3rc1")
    # a candidate comes before its release: 0.3.1rc1 would not have parked the likely value by default
    assert not FF.parks_likely_by_default("0.3.1rc1") and FF.parks_likely_by_default("0.3.1")
    assert not FF.parks_by_default("a164fc5") and not FF.parks_by_default("0.3.0rc2")
    assert FF.version_key("0.3.1 (candidate)") == FF.version_key("0.3.1")


# ------------------------------------------------------------------------------------------------ lengths of time

class NoErrors:
    """A Target whose instruction properties give their duration and nothing else (an error read raises)."""

    def __init__(self, t):
        self._t = t
        self.operation_names = t.operation_names
        for c in Dv.TIMING_CONSTRAINTS:
            setattr(self, c, getattr(t, c))

    def __getitem__(self, name):
        inner = self._t[name]

        class P:
            def __init__(self, p):
                self._p = p

            @property
            def duration(self):
                return self._p.duration

            def __getattr__(self, k):
                raise AssertionError(f"instruction property {k} read")

        class M:
            def items(self_):
                return [(qa, None if p is None else P(p)) for qa, p in inner.items()]
        return M()

    def __getattr__(self, k):
        raise AssertionError(f"Target.{k} read")


def test_timing_reads_lengths_of_time_only():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime import fake_provider
    b = fake_provider.FakeFez()
    t = Dv.timing_of(NoErrors(b.target), b.dt)
    assert t["constraints"] == dict(granularity=1, min_length=2, pulse_alignment=1, acquire_alignment=1)
    assert set(t["durations"]) == {"x", "sx", "cz", "measure", "reset"}
    assert t["durations"]["x"]["0"] == 6 and t["durations"]["measure"]["0"] == 390
    assert Dv.timing_of(b.target, None) is None
    dev = Dv.fetch_fake("FakeFez")
    assert dev.timing == t
    # The record's keys name what was read; a calibration field would show as a key. (Searching the JSON text matched
    # the fetch's timestamp: "...t11:20..." contains "t1".)
    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield str(k).lower()
                yield from keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from keys(v)
    names = list(keys(dev.to_json()))
    for word in ("error", "t1", "t2", "frequency", "readout"):
        assert not any(word in k for k in names if not k.isdigit() and "," not in k), word
    old = Dv.Device(**{k: v for k, v in dev.to_json().items() if k != "timing"})
    assert "timing" not in old.to_json()                     # a record without timing keeps its JSON


def test_a_delay_off_the_time_grid_is_refused(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    dev.timing = dict(dev.timing, constraints=dict(granularity=16, min_length=64, pulse_alignment=16,
                                                   acquire_alignment=16))
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    qc.delay(4500, 1, unit="dt")
    qc.measure(1, 1)
    with pytest.raises(RuntimeError, match="time grid"):
        FF.build(dev, [("d", qc, dict(success={1: 0}))], str(tmp_path / "x"), shots=100, sentinel_spec=SPEC)
    assert J.delay_grid_problems([qc], dev.timing) == [dict(pub=0, dt=4500)]
    ok = dict(dev.timing, constraints=dict(granularity=1, min_length=2, pulse_alignment=1, acquire_alignment=1))
    assert J.delay_grid_problems([qc], ok) == [] and J.delay_grid_problems([qc], None) == []


# ------------------------------------------------------------------------------------------------ the compiled check

def test_a_parked_build_is_checked_in_its_compiled_circuit(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    out = str(tmp_path / "p")
    FF.build(dev, [("ip", _ipe_like(), dict(success={2: 1}))], out, shots=500, sentinel_spec=SPEC)
    man = json.load(open(os.path.join(out, "job", "manifest.json")))
    rec = man["compiled"]["records"]["ff_ip"]
    assert rec["park_checked"]["waits"] >= 2 and rec["park_checked"]["qubits"] >= 2
    target = json.load(open(os.path.join(out, "runs", "ff_ip.json")))["chain"][1]
    for w in rec["waits"]["waits"]:
        for q in w["qubits"]:
            if q["q"] == target:
                assert q["p1_max"] == 0.0 and q["entangled"] is False       # parked in |0> during every wait
    assert rec["fences"]["depth_after"] >= rec["fences"]["depth_before"]
    assert "lengths" in rec and rec["lengths"]["length_dt"] > 0                # FakeFez durations in the record
    conds = rec["lengths"]["conditionals"]
    assert conds["n"] == 2 and all(m == 390 for m, _ in conds["wait_dt_lower"])
    assert len(man["compiled"]["summary"]) == len(man["pubs"])
    assert all(set(s) == {"cz", "depth", "gates_sha256"} for s in man["compiled"]["summary"])


def test_the_check_refuses_a_parking_the_compiler_did_not_keep():
    qc = _ipe_like()
    plan = park_plan(qc)
    assert plan and all(w["kind"] == "always" and w["qubits"] == [1] for w in plan)
    parked, _ = park_waits(qc)
    assert CP.check_parked(parked, plan)["qubits"] == len(plan)
    with pytest.raises(ValueError, match="did not keep the parking"):
        CP.check_parked(fence_waits(qc, conditionals=True), plan)            # the X pairs gone: q1 waits in |1>
    # an X pair moved inside the wait: the qubit is in |1> again while the read is decided
    moved = parked.copy_empty_like()
    data = list(parked.data)
    for i, ins in enumerate(data):
        moved.append(ins.operation, ins.qubits, ins.clbits)
        if ins.operation.name == "measure" and i == next(k for k, x in enumerate(data) if x.operation.name == "measure"):
            moved.x(1)
            moved.x(1)
            moved.x(1)
    with pytest.raises(ValueError):
        CP.check_parked(moved, plan)


def test_the_likely_value_check():
    qc = _likely()
    plan = park_likely_plan(qc)
    assert plan and plan[0]["kind"] == "likely" and plan[0]["qubits"] == [1]
    p, _ = park_likely(qc)
    assert CP.check_parked(p, plan)["qubits"] == 1
    with pytest.raises(ValueError):
        CP.check_parked(fence_waits(qc, conditionals=True), plan)
    w = CP.waits(p)["waits"][0]
    p_start, p_max, ent = w["qubits"][1]
    assert p_start < 0.5 and p_max < 0.5 and not ent


def test_a_bundle_and_pack_fixed_check_every_sub_item(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("a", _ipe_like(), dict(success={2: 1})), ("b", FF.teleport_test(1), dict(success={1: 0}))]
    FF.build_bundle(dev, items, str(tmp_path / "b"), shots=200, sentinel_spec=SPEC, allow_conditional_neighbours=True)
    man = json.load(open(os.path.join(str(tmp_path / "b"), "job", "manifest.json")))
    assert man["compiled"]["records"]["ff_pack"]["a"]["park_checked"]["waits"] >= 2
    FF.build_pack_fixed(dev, items, str(tmp_path / "f"), shots=200, sentinel_spec=SPEC)
    man = json.load(open(os.path.join(str(tmp_path / "f"), "job", "manifest.json")))
    assert man["compiled"]["records"]["ff_a"]["park_checked"]["waits"] >= 2
    # the off arm (no conditionals): its second read is part of the final measurements, so one wait
    assert man["compiled"]["records"][f"ff_{FF.FIXED_PACK_TAG}_off"]["a"]["park_checked"]["waits"] == 1


def test_the_checks_add_records_only(tmp_path):
    """The circuits are the same with and without a device record holding timing (the compiler never sees it)."""
    dev = Dv.fetch_fake("FakeFez")
    bare = Dv.Device(**{k: v for k, v in dev.to_json().items() if k != "timing"})
    items = [("ip", _ipe_like(), dict(success={2: 1}))]
    FF.build(dev, items, str(tmp_path / "a"), shots=300, sentinel_spec=SPEC)
    FF.build(bare, items, str(tmp_path / "b"), shots=300, sentinel_spec=SPEC)
    for f in sorted(os.listdir(str(tmp_path / "a" / "runs"))):
        if f.endswith(".qpy"):
            assert open(str(tmp_path / "a" / "runs" / f), "rb").read() == \
                open(str(tmp_path / "b" / "runs" / f), "rb").read()
    mb = json.load(open(str(tmp_path / "b" / "job" / "manifest.json")))
    assert "lengths" not in mb["compiled"]["records"]["ff_ip"]                 # no durations: no lengths


# ------------------------------------------------------------------------------------------------ verify

def test_same_job_leaves_out_the_new_record_and_says_what_changed(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("ip", _ipe_like(), dict(success={2: 1}))]
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    FF.build(dev, items, a, shots=300, sentinel_spec=SPEC)
    FF.build(dev, items, b, shots=300, sentinel_spec=SPEC)
    m = json.load(open(os.path.join(a, "job", "manifest.json")))
    del m["compiled"]                                            # a manifest recorded before 0.3.3rc1
    json.dump(m, open(os.path.join(a, "job", "manifest.json"), "w"))
    diff, how, summ = LP.same_job_summary(os.path.join(a, "job"), os.path.join(b, "job"))
    assert diff is None and how == "bytes" and summ.startswith("summary: the same for all")
    c = str(tmp_path / "c")
    FF.build(dev, items, c, shots=300, sentinel_spec=SPEC, park=False)
    diff, how, summ = LP.same_job_summary(os.path.join(b, "job"), os.path.join(c, "job"))
    assert diff is not None and "circuits differ" in summ


def test_summary_difference_line():
    s = [dict(cz=2, depth=10, gates_sha256="a"), dict(cz=4, depth=12, gates_sha256="b")]
    t = [dict(cz=2, depth=10, gates_sha256="a"), dict(cz=4, depth=14, gates_sha256="c")]
    assert CP.summary_difference(s, s) == "summary: the same for all 2 circuits"
    assert CP.summary_difference(s, t, ["x", "y"]) == ("summary: 1 of 2 circuits differ; the first, circuit 1 (y): "
                                                       "depth 12 -> 14")


# ------------------------------------------------------------------------------------------------ the loop

class Exp:
    name = "fake"
    k_max = 2
    budget_s = 100.0


def test_the_loop_is_paired_on_the_device_by_default(tmp_path):
    z = str(tmp_path / "z.zip")
    open(z, "w").close()
    LP.start(str(tmp_path / "a"), Exp(), zip_path=z)
    cfg = json.load(open(str(tmp_path / "a" / "run.json")))
    assert cfg["paired"] is True and "cross_job" not in cfg
    with pytest.raises(SystemExit, match="cross_job"):
        LP.start(str(tmp_path / "b"), Exp(), zip_path=z, paired=False)
    LP.start(str(tmp_path / "c"), Exp(), zip_path=z, paired=False, cross_job=True)
    cfg = json.load(open(str(tmp_path / "c" / "run.json")))
    assert "paired" not in cfg and cfg["cross_job"] is True


def test_dry_runs_keep_their_default(tmp_path):
    d = str(tmp_path / "dev")
    Dv.fetch_fake("FakeFez").save(d)
    LP.start(str(tmp_path / "n"), Exp(), dry_devdir=d)
    assert "paired" not in json.load(open(str(tmp_path / "n" / "run.json")))


def test_version_order_dev_before_pre_releases():
    # PEP 440: a bare .devN comes before aN of the same release; .post is not accepted (None).
    order = ["0.3.0.dev1", "0.3.0a1", "0.3.0b2", "0.3.0rc1.dev2", "0.3.0rc1", "0.3.0", "0.3.1rc1", "0.3.1"]
    assert sorted(reversed(order), key=FF.version_key) == order
    assert FF.is_candidate("0.3.0.dev1") and not FF.parks_by_default("0.3.0.dev1")
    assert FF.version_key("0.3.0.post1") is None

