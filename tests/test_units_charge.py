"""0.3.2 (candidate, 2026-11-23): dt in the device record, delays in dt and microseconds, the limit on one circuit's
delay, and the delays priced in the time model (results/ipe-branch-cost-night-2026-11-22: 112 s charged against a
model of 59.6 s, the delays unpriced). A job without a delay keeps the seconds and the manifest it had."""
import dataclasses
import json
import os

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from psf_os import device as Dv
from psf_os import feedforward as FF
from psf_os import job as J

DT = 4e-9


def _dev(dt=DT):
    d = dataclasses.replace(Dv.fetch_fake("FakeFez"), failed_qubits=[], failed_couplers=[])
    return dataclasses.replace(d, dt=dt)


def _item(tag, delay=None, unit="dt"):
    """A two-qubit item with one conditional (the teleport-like test) and, when asked, a delay on both qubits."""
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    if delay is not None:
        qc.delay(delay, [0, 1], unit=unit)
    qc.measure(1, 1)
    off = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
    off.h(0)
    off.measure(0, 0)
    off.measure(1, 1)
    return (tag, qc, dict(success={1: 0}, bound=None, off_circuit=off, form="blocks")) if False else \
        (tag, qc, dict(success={0: 0, 1: 0}, off_circuit=off, form="blocks"))


def test_dt_in_the_record_and_old_records_unchanged(tmp_path):
    d = Dv.fetch_fake("FakeFez")
    assert d.dt and 1e-10 < d.dt < 1e-8                   # the clock step, a unit of time
    old = dataclasses.replace(d, dt=None)
    assert "dt" not in old.to_json()                       # a record without dt keeps its JSON
    old.save(str(tmp_path / "old"))
    assert Dv.Device.load(str(tmp_path / "old")).dt is None
    new = dataclasses.replace(d, dt=DT)
    new.save(str(tmp_path / "new"))
    assert Dv.Device.load(str(tmp_path / "new")).dt == DT
    assert "dt" not in {k for k in Dv.map_free_target(new).operation_names}


def test_the_model_without_delays_is_the_model_it_was():
    assert J.qpu_seconds(68, 2950) == 68 * 0.036 + 68 * 2950 * 0.00027 + 3.0
    assert round(J.qpu_seconds(68, 2950), 1) == 59.6      # the branch-cost night's model
    assert J.qpu_seconds(68, 2950, 0.0) == J.qpu_seconds(68, 2950)


def test_the_branch_cost_night_priced():
    """Plan d: 8 phases x (270 + 540 us at n = 6, 504 + 1008 us at n = 8), 2950 shots: 54.8 s of delays; the model
    with them 114.4 s against 112 s charged (the 105-job relation with them: 111.4 s)."""
    per_shot = 8 * (270 + 540 + 504 + 1008) * 1e-6
    assert round(2950 * per_shot, 1) == 54.8
    assert round(J.qpu_seconds(68, 2950, per_shot), 1) == 114.4
    assert abs(J.qpu_seconds(68, 2950, per_shot) - 112) / 112 < 0.025
    assert round(2.6 + 0.000269 * 68 * 2950 + 2950 * per_shot, 1) == 111.4


def test_delays_to_dt():
    qc = QuantumCircuit(1)
    qc.delay(1.0, 0, unit="us")
    (c,) = J.delays_to_dt([qc], DT)
    (x,) = c.data
    assert (x.operation.unit, x.operation.duration) == ("dt", 250)
    with pytest.raises(ValueError, match="no dt"):
        J.delays_to_dt([qc], None)
    qd = QuantumCircuit(1)
    qd.delay(250, 0, unit="dt")
    assert J.delays_to_dt([qd], None)[0] is qd            # delays in dt are kept, with or without a dt
    odd = QuantumCircuit(1)
    odd.delay(1.0, 0, unit="ns")
    with pytest.raises(ValueError, match="whole number of dt"):
        J.delays_to_dt([odd], DT)


def test_a_job_without_delay_keeps_its_manifest(tmp_path):
    rep = FF.build(_dev(), [_item("a")], str(tmp_path / "o"), shots=1000, ff_only=["a"], park=True, park_likely=True)
    man = json.load(open(tmp_path / "o" / "job" / "manifest.json"))
    assert "delays" not in man
    assert rep["qpu_seconds_model"] == round(J.qpu_seconds(rep["circuits"], 1000), 1)


def test_delays_recorded_priced_and_converted(tmp_path):
    rep = FF.build(_dev(), [_item("a", 2.0, "us")], str(tmp_path / "o"), shots=1000, ff_only=["a"], park=True,
                   park_likely=True)
    man = json.load(open(tmp_path / "o" / "job" / "manifest.json"))
    d = man["delays"]
    assert d["dt_seconds"] == DT and d["limit_us"] == J.DELAY_LIMIT_US and d["allow_long_delay"] is False
    (pc,) = d["per_circuit"]
    assert (pc["dt"], pc["us"]) == (500, 2.0)
    assert rep["qpu_seconds_model"] == round(J.qpu_seconds(rep["circuits"], 1000, 2e-6), 1)
    assert d["qpu_seconds_delays"] == round(2e-6 * 1000, 1)
    from qiskit import qpy
    cs = qpy.load(open(tmp_path / "o" / "job" / "circuits.qpy", "rb"))
    units = {x.operation.unit for c in cs for x in c.data if x.operation.name == "delay"}
    assert units == {"dt"}                                 # the delay reached the job file, in dt


def test_the_limit_refuses_before_writing(tmp_path):
    long_dt = int((J.DELAY_LIMIT_US + 10) * 1e-6 / DT)
    with pytest.raises(RuntimeError, match="allow_long_delay"):
        FF.build(_dev(), [_item("a", long_dt)], str(tmp_path / "o"), shots=1000, ff_only=["a"], park=True,
                 park_likely=True)
    assert not os.path.exists(tmp_path / "o" / "job")
    rep = FF.build(_dev(), [_item("a", long_dt)], str(tmp_path / "p"), shots=1000, ff_only=["a"], park=True,
                   park_likely=True, allow_long_delay=True)
    man = json.load(open(tmp_path / "p" / "job" / "manifest.json"))
    assert man["delays"]["allow_long_delay"] is True and man["delays"]["longest_us"] > J.DELAY_LIMIT_US
    assert rep["qpu_seconds_model"] > J.qpu_seconds(rep["circuits"], 1000)


def test_without_dt(tmp_path):
    with pytest.raises((ValueError, FF.Refused), match="no dt"):
        FF.build(_dev(None), [_item("a", 2.0, "us")], str(tmp_path / "o"), shots=1000, ff_only=["a"], park=True,
                 park_likely=True)
    long_dt = int((J.DELAY_LIMIT_US + 10) * 1e-6 / DT)
    rep = FF.build(_dev(None), [_item("a", long_dt)], str(tmp_path / "p"), shots=1000, ff_only=["a"], park=True,
                   park_likely=True)                       # as before 0.3.2: recorded in dt, neither limited nor priced
    man = json.load(open(tmp_path / "p" / "job" / "manifest.json"))
    assert man["delays"]["per_circuit"][0]["us"] is None
    assert rep["qpu_seconds_model"] == round(J.qpu_seconds(rep["circuits"], 1000), 1)
