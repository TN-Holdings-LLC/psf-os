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
