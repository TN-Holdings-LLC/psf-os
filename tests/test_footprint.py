"""psf_os.footprint (2026-11-20): the qubits not in a computational basis state while a mid-circuit measurement is
read, counted exactly over every branch."""
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from psf_os.footprint import wait_footprint


def _qc(n, nc):
    return QuantumCircuit(QuantumRegister(n, "q"), ClassicalRegister(nc, "c"))


def test_final_measurements_are_not_waits():
    qc = _qc(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    r = wait_footprint(qc)
    assert r["waits"] == [] and r["max"] == 0


def test_ghz_waits_on_three_and_basis_states_do_not_count():
    qc = _qc(5, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)                  # GHZ on 0, 1, 2
    qc.x(3)                      # 3 in |1>: a basis state, not a carrier
    qc.measure(4, 0)             # a mid-circuit read of an idle qubit
    qc.h(0)
    qc.measure(0, 1)
    r = wait_footprint(qc)
    assert len(r["waits"]) == 1 and r["waits"][0]["carriers"] == [0, 1, 2] and r["waits"][0]["measured"] == [4]


def test_branches_and_conditionals():
    qc = _qc(3, 2)
    qc.h(0)
    qc.measure(0, 0)             # wait 1: nothing else carries (1, 2 at |0>)
    with qc.if_test((qc.clbits[0], 1)):
        qc.h(1)                  # in one branch 1 becomes |+>
    qc.measure(2, 1)             # wait 2: 1 carries in the branch where c0 = 1 (any branch counts); 0 is |0> or |1>
    qc.h(1)
    qc.measure(1, 0)
    r = wait_footprint(qc)
    assert [w["carriers"] for w in r["waits"]] == [[], [1]] and r["total"] == 1


def test_measured_qubit_reused_after_its_read():
    qc = _qc(2, 2)
    qc.h(1)
    qc.measure(0, 0)             # 1 in |+> waits
    qc.x(0)
    qc.measure(1, 1)
    r = wait_footprint(qc)
    assert r["waits"][0]["carriers"] == [1]


def test_build_records_the_footprint_only_when_asked(tmp_path):
    import json
    from psf_os import device as Dv
    from psf_os import feedforward as FF
    dev = Dv.fetch_fake("FakeFez")
    qc, sp = FF.rus_chain(2, 1.1, m=2)
    kw = dict(sentinel_spec=dict(kind="echo_clifford", D=24))
    r0 = FF.build(dev, [("r", qc, sp)], str(tmp_path / "plain"), **kw)
    r1 = FF.build(dev, [("r", qc, sp)], str(tmp_path / "fp"), footprint=True, **kw)
    assert "footprint_max" not in r0["items"]["r"] and r1["items"]["r"]["footprint_max"] >= 0
    for f in ("ff_r.qpy", "ff_r_off.qpy", "ff_r_floor.qpy"):
        assert open(str(tmp_path / "plain" / "runs" / f), "rb").read() == open(str(tmp_path / "fp" / "runs" / f),
                                                                                "rb").read()
    rec = json.load(open(str(tmp_path / "fp" / "runs" / "ff_r.json")))
    assert "wait_footprint" in json.dumps(rec)


def test_fence_keeps_the_target_out_of_the_wait(tmp_path):
    """after-2026-11-20 (the workplace VS's finding): compiled without the fence, IPE's target waits in a
    superposition (the compiler merges its return to |1> with the next round's preparation across the wait); with
    fence=True the footprint as sent is 0 at every wait, the CZ count is the same and the noiseless distribution of
    every arm is unchanged."""
    import json
    import os
    import sys
    import pytest
    from psf_os import device as Dv
    from psf_os import feedforward as FF
    from psf_os.footprint import fence_waits
    nb = os.environ.get("TOQB_DIR")
    if not nb or not os.path.isdir(os.path.join(nb, "notes", "ffapps_2026-11-17")):
        pytest.skip("needs the lab notebook (TOQB_DIR) for apps.ipe")
    sys.path.insert(0, os.path.join(nb, "notes", "ffapps_2026-11-17"))
    import apps as AP
    dev = Dv.fetch_fake("FakeFez")
    qc, succ = AP.ipe(3, 5, True)
    off, _ = AP.ipe(3, 5, False)
    item = [("ipe", qc, dict(success=succ, off_circuit=off, form="blocks"))]
    kw = dict(sentinel_spec=dict(kind="echo_clifford", D=24), footprint=True)
    r0 = FF.build(dev, item, str(tmp_path / "plain"), **kw)
    r1 = FF.build(dev, item, str(tmp_path / "fence"), fence=True, **kw)
    assert r0["items"]["ipe"]["footprint_max"] >= 1 and r1["items"]["ipe"]["footprint_max"] == 0
    assert r0["items"]["ipe"]["cz"] == r1["items"]["ipe"]["cz"]
    assert FF.exact_distribution(fence_waits(qc)) == FF.exact_distribution(qc)
    # the workplace VS's その6: fenced at the reads only, the target still waits in a superposition at the phase
    # feedback (counted with conditionals=True); fenced at every wait it does not
    from psf_os.footprint import wait_footprint
    ch = FF.place(dev, 2)["chain"]
    comp = FF.Compiler(dev)
    reads = wait_footprint(comp.run(FF.on_device(fence_waits(qc), ch, dev.num_qubits)), span=True, conditionals=True)
    every = wait_footprint(comp.run(FF.on_device(fence_waits(qc, conditionals=True), ch, dev.num_qubits)), span=True,
                           conditionals=True)
    assert reads["max"] >= 1 and any(w.get("kind") == "conditional" for w in reads["waits"])
    assert every["max"] == 0
    assert FF.exact_distribution(fence_waits(qc, conditionals=True)) == FF.exact_distribution(qc)
    rec = json.load(open(os.path.join(str(tmp_path / "fence"), "runs", "ff_ipe.json")))
    assert '"fence": true' in json.dumps(rec)
    FF.build(dev, item, str(tmp_path / "plain2"), sentinel_spec=dict(kind="echo_clifford", D=24))
    FF.build(dev, item, str(tmp_path / "plain3"), sentinel_spec=dict(kind="echo_clifford", D=24), fence=False)
    assert open(str(tmp_path / "plain2" / "job" / "circuits.qpy"), "rb").read() == \
        open(str(tmp_path / "plain3" / "job" / "circuits.qpy"), "rb").read()


def test_record_only_checks_change_nothing_of_the_job(tmp_path):
    """after-2026-11-20 (home's three conditions): record_checks="record" (the default) writes OUT/checks.json only: job/,
    runs/ and build.json are byte for byte those of record_checks="off" (the manifest but its time stamp); a CZ difference
    is noted, never refused; a circuit above the footprint's bounds is recorded as not counted, and the build does not
    wait on it."""
    import json
    import os
    import time
    from psf_os import device as Dv
    from psf_os import feedforward as FF
    dev = Dv.fetch_fake("FakeFez")
    kw = dict(sentinel_spec=dict(kind="echo_clifford", D=24))
    qc = _qc(3, 3)                                     # a CX pair the compiler cancels: written 2 CZ more than compiled
    qc.h(0)
    qc.cz(0, 1)
    qc.cz(0, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    qc.measure(1, 1)
    qc.measure(2, 2)
    item = [("c", qc, dict(success={2: 0}))]
    FF.build(dev, item, str(tmp_path / "rec"), **kw)
    FF.build(dev, item, str(tmp_path / "off"), record_checks="off", **kw)
    a, b = tmp_path / "rec", tmp_path / "off"
    assert not (b / "checks.json").exists()
    for rel in ("job/circuits.qpy", "build.json") + tuple("runs/" + f for f in sorted(os.listdir(b / "runs"))):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
    ma, mb = (json.load(open(x / "job" / "manifest.json")) for x in (a, b))
    for m in (ma, mb):
        m.pop("written")
    assert ma == mb
    row = json.load(open(a / "checks.json"))["items"]["c"]
    assert row["cz_written"]["ff"] == 2 and row["cz_compiled"]["ff"] == 0 and not row["cz_same"]
    assert row["footprint_sent"]["ff"]["counted"] and row["footprint_written"]["counted"]
    big = FF.teleport_test(14)                         # 15 qubits: above MAX_QUBITS
    t = time.time()
    FF.build(dev, [("big", big if hasattr(big, "data") else big[0], dict(success={14: 0}))], str(tmp_path / "big"),
             **kw)
    assert time.time() - t < 120
    r = json.load(open(tmp_path / "big" / "checks.json"))["items"]["big"]
    assert r["footprint_sent"]["ff"]["counted"] is False and "above" in r["footprint_sent"]["ff"]["why"]
    deep = FF.teleport_test(12)                        # 13 qubits, 12 reads: above MAX_WORK
    from psf_os.footprint import footprint_or_none
    t = time.time()
    f = footprint_or_none(deep if hasattr(deep, "data") else deep[0])
    assert f["counted"] is False and time.time() - t < 60
