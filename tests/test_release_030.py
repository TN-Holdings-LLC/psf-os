"""release-0.3.0: the safety guard (a declared psf-os whose park default differs from this one's refuses a build that
does not say park), the version and park setting in every manifest (psf_os_build, left out by loop.same_job), and
parking in build_bundle and build_pack_fixed (as build: records, CZ, park=False as before)."""
import json
import os

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

import psf_os
from psf_os import device as Dv
from psf_os import feedforward as FF
from psf_os import loop as LP

SPEC = dict(kind="echo_clifford", D=24)


def _ipe_like():
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


def test_version_is_0_3_0():
    assert psf_os.__version__ == "0.3.0"


def test_parks_by_default_reads_versions_and_commits():
    assert FF.parks_by_default("0.3.0") and FF.parks_by_default("v0.4.1")
    assert not FF.parks_by_default("0.2.0") and not FF.parks_by_default("a164fc5") and not FF.parks_by_default("")


def test_the_guard_refuses_a_build_that_does_not_say_park(tmp_path, monkeypatch):
    dev = Dv.fetch_fake("FakeFez")
    items = [("ip", _ipe_like(), dict(success={2: 1}))]
    with pytest.raises(FF.Refused, match="pass park=False"):
        FF.build(dev, items, str(tmp_path / "a"), shots=500, sentinel_spec=SPEC, declared="a164fc5")
    assert not os.path.exists(tmp_path / "a")
    monkeypatch.setenv("PSF_OS_DECLARED", "a164fc5")
    with pytest.raises(FF.Refused, match="does not park by default"):
        FF.build(dev, items, str(tmp_path / "b"), shots=500, sentinel_spec=SPEC)
    FF.build(dev, items, str(tmp_path / "c"), shots=500, sentinel_spec=SPEC, park=False)      # said: built
    man = json.load(open(tmp_path / "c" / "job" / "manifest.json"))
    assert man["psf_os_build"] == dict(version="0.3.0", park=False, park_set="explicit", declared="a164fc5")
    monkeypatch.setenv("PSF_OS_DECLARED", "0.3.0")
    FF.build(dev, items, str(tmp_path / "d"), shots=500, sentinel_spec=SPEC)                  # the same default
    man = json.load(open(tmp_path / "d" / "job" / "manifest.json"))
    assert man["psf_os_build"]["park"] is True and man["psf_os_build"]["park_set"] == "default"


def test_the_guard_follows_the_recorded_default_in_verify(tmp_path, monkeypatch):
    # a recorded night (no "park" in run.json) is rebuilt under park_default(False): a164fc5 agrees, no refusal
    monkeypatch.setenv("PSF_OS_DECLARED", "a164fc5")
    with FF.park_default(False):
        assert FF.resolve_park(None) == (False, "default")


def test_same_job_leaves_psf_os_build_out(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("ip", _ipe_like(), dict(success={2: 1}))]
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    FF.build(dev, items, a, shots=500, sentinel_spec=SPEC, park=False)
    FF.build(dev, items, b, shots=500, sentinel_spec=SPEC, park=False)
    m = json.load(open(os.path.join(a, "job", "manifest.json")))
    m.pop("psf_os_build")                                          # as a manifest written before 0.3.0
    json.dump(m, open(os.path.join(a, "job", "manifest.json"), "w"), indent=1)
    assert LP.same_job(os.path.join(a, "job"), os.path.join(b, "job"))[0] is None


def test_build_bundle_parks_each_sub_item(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("ia", _ipe_like(), dict(success={2: 1})), ("ib", _ipe_like(), dict(success={2: 1}))]
    on, off = str(tmp_path / "on"), str(tmp_path / "off")
    r1 = FF.build_bundle(dev, items, on, tag="pk", shots=500, sentinel_spec=SPEC, allow_conditional_neighbours=True)
    r0 = FF.build_bundle(dev, items, off, tag="pk", shots=500, sentinel_spec=SPEC, allow_conditional_neighbours=True,
                         park=False)
    rec = json.load(open(os.path.join(on, "runs", "ff_pk.json")))
    for t in ("ia", "ib"):
        parked = rec["bundle"][t]["park"]["arms"]["ff"]["parked"]
        assert parked and all(w["physical"] == [rec["bundle"][t]["qubits"][1]] for w in parked)
    rec0 = json.load(open(os.path.join(off, "runs", "ff_pk.json")))
    assert all("park" not in rec0["bundle"][t] for t in ("ia", "ib"))
    assert r1["cz"] == r0["cz"]
    assert json.load(open(os.path.join(on, "job", "manifest.json")))["psf_os_build"]["park"] is True
    assert json.load(open(os.path.join(on, "checks.json")))["items"]["pk"]["park"]["ia"]["park"] is True
    FF.run_local(on)
    cards = FF.cards(on)
    assert cards["ia"]["ff"]["share"] == 1.0 and cards["ib"]["ff"]["share"] == 1.0


def test_build_bundle_park_false_is_as_the_context(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("ia", _ipe_like(), dict(success={2: 1})), ("ib", _ipe_like(), dict(success={2: 1}))]
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    FF.build_bundle(dev, items, a, tag="pk", shots=500, sentinel_spec=SPEC, allow_conditional_neighbours=True,
                    park=False)
    with FF.park_default(False):
        FF.build_bundle(dev, items, b, tag="pk", shots=500, sentinel_spec=SPEC, allow_conditional_neighbours=True)
    for f in sorted(os.listdir(os.path.join(a, "runs"))):
        assert open(os.path.join(a, "runs", f), "rb").read() == open(os.path.join(b, "runs", f), "rb").read()


def test_build_pack_fixed_parks_the_ff_arms(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("ia", _ipe_like(), dict(success={2: 1})), ("ib", _ipe_like(), dict(success={2: 1}))]
    on, off = str(tmp_path / "on"), str(tmp_path / "off")
    r1 = FF.build(dev, items, on, shots=500, sentinel_spec=SPEC, pack_fixed=True)
    r0 = FF.build(dev, items, off, shots=500, sentinel_spec=SPEC, pack_fixed=True, park=False)
    rec = json.load(open(os.path.join(on, "runs", "ff_ia.json")))
    assert rec["park"]["arms"]["ff"]["parked"] and "park" not in json.load(open(os.path.join(off, "runs", "ff_ia.json")))
    assert r1["cz"] == r0["cz"] and all(r1["items"][t]["cz"] == r0["items"][t]["cz"] for t in ("ia", "ib"))
    FF.run_local(on)
    assert FF.cards(on)["ia"]["ff"]["share"] == 1.0


def test_park_skip_holds_both_forms_in_one_job(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [("pk", _ipe_like(), dict(success={2: 1})), ("pl", _ipe_like(), dict(success={2: 1}))]
    out = str(tmp_path / "j")
    FF.build(dev, items, out, shots=500, sentinel_spec=SPEC, park=True, park_skip=["pl"])
    rk = json.load(open(os.path.join(out, "runs", "ff_pk.json")))
    rl = json.load(open(os.path.join(out, "runs", "ff_pl.json")))
    assert rk["park"]["arms"]["ff"]["parked"] and rl["park"] == {"park": False, "skipped": "park_skip"}
    assert FF.park_of_record(rl) is False and FF.park_of_record(rk) is True
    ref = str(tmp_path / "ref")
    FF.build(dev, [("pl", _ipe_like(), dict(success={2: 1}))], ref, shots=500, sentinel_spec=SPEC, park=False)
    assert open(os.path.join(out, "runs", "ff_pl.qpy"), "rb").read() == open(os.path.join(ref, "runs", "ff_pl.qpy"),
                                                                             "rb").read()
    with pytest.raises(FF.Refused, match="park_skip names unknown"):
        FF.build(dev, items, str(tmp_path / "x"), shots=500, sentinel_spec=SPEC, park=True, park_skip=["zz"])
