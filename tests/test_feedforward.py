"""psf_os.feedforward (2026-11-16): the accepted form and its refusals, the arms (ff, off, floor), the noiseless
answer card of the teleport test (ff 1, off 3/4 at k = 1 and 1/2 beyond, floor all zero, no shot discarded), the
pre-submit check inside control-flow blocks, and the IBM submission path offline (SamplerV2 in local mode)."""
import json
import os
import shutil

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from psf_os import device as Dv
from psf_os import feedforward as FF
from psf_os import ibm


def _qc(n=2, nc=2):
    return QuantumCircuit(QuantumRegister(n, "q"), ClassicalRegister(nc, "c"))


def test_teleport_test_is_accepted():
    for k in (1, 2, 4):
        a = FF.analyse(FF.teleport_test(k))
        assert a == dict(n=k + 1, clbits=k + 1, mid=k, conditionals=k, cz=k, cz_in_blocks=0)
        off = FF.strip_conditionals(FF.teleport_test(k))
        assert FF.analyse(off)["conditionals"] == 0


def test_refusals_say_why():
    qc = _qc()
    qc.cx(0, 1)
    with pytest.raises(FF.Refused, match="CZ only"):
        FF.analyse(qc)
    qc = _qc(3, 3)
    qc.cz(0, 2)
    with pytest.raises(FF.Refused, match="neighbours"):
        FF.analyse(qc)
    qc = _qc()
    qc.measure(0, 0)
    qc.measure(1, 0)
    with pytest.raises(FF.Refused, match="written twice"):
        FF.analyse(qc)
    qc = _qc()
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.h(1)
    qc.measure(1, 1)
    with pytest.raises(FF.Refused, match="only X or Z"):
        FF.analyse(qc)
    qc = _qc()
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    with pytest.raises(FF.Refused, match="before it is measured"):
        FF.analyse(qc)
    qc = QuantumCircuit(QuantumRegister(2), ClassicalRegister(1, "a"), ClassicalRegister(1, "b"))
    with pytest.raises(FF.Refused, match="one classical register"):
        FF.analyse(qc)
    qc = _qc()
    qc.reset(0)
    qc.measure([0, 1], [0, 1])
    with pytest.raises(FF.Refused, match="reset"):
        FF.analyse(qc)


def test_noiseless_card_ff_off_floor(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    items = [(f"tp{k}", FF.teleport_test(k), dict(success={k: 0}, bound=0.75)) for k in (1, 2, 4)]
    out = str(tmp_path / "j")
    rep = FF.build(dev, items, out, shots=2000, sentinel_spec=dict(kind="echo_clifford", D=24))
    assert rep["circuits"] == 3 * 3 + 4 and rep["items"]["tp4"]["cz"] == 4
    names = sorted(f for f in os.listdir(os.path.join(out, "runs")) if f.startswith("ff_") and f.endswith(".json"))
    assert len(names) == 9
    rec = json.load(open(os.path.join(out, "runs", "ff_tp2.json")))
    assert rec["psfos_job"]["kind"] == "feedforward" and rec["chain_rule"]["rule"] == "C" and rec["mid"] == 2
    FF.run_local(out)
    cards = FF.cards(out)
    for k in (1, 2, 4):
        c = cards[f"tp{k}"]
        assert c["ff"]["share"] == 1.0 and c["ff"]["discarded"] == 0 and c["ff"]["counted"] == 2000
        assert c["floor"]["all_zero"] == 1.0
        want = 0.75 if k == 1 else 0.5
        assert abs(c["off"]["share"] - want) < 0.05
        assert c["off_best_fixed_rule_after_the_fact"] < 0.8
        assert c["above_bound_in_se"] is None or c["above_bound_in_se"] > 10
    with pytest.raises(FF.Refused, match="already run"):
        FF.run_local(out)
    with pytest.raises(FF.Refused, match="exists"):
        FF.build(dev, items, out)


def test_floor_strips_inside_blocks_and_compiler_keeps_conditionals():
    dev = Dv.fetch_fake("FakeFez")
    pl = FF.place(dev, 3)
    comp = FF.Compiler(dev)
    c = comp.run(FF.on_device(FF.teleport_test(2), pl["chain"], dev.num_qubits))
    names = [n for n, _ in FF.walk(c)]
    assert names.count("if_else") == 2 and "x" in names
    f = FF.floor_of(c)
    fn = [n for n, _ in FF.walk(f)]
    assert "sx" not in fn and "x" not in fn and fn.count("if_else") == 2 and fn.count("cz") == names.count("cz")
    assert all(all(v.values()) for v in (FF.checks(dev, c, pl["chain"]), FF.checks(dev, f, pl["chain"])))


def test_pre_submit_check_reads_inside_blocks():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeFez
    t = FakeFez().target                       # an IBM backend target (problems() reads its failed marks)
    _, failed_c, failed_q = Dv.failed_elements(t)
    good = next(qa for qa in t["cz"] if qa and tuple(sorted(qa)) not in {tuple(sorted(c)) for c in failed_c}
                and not set(qa) & set(failed_q))
    a, b = good
    far = next(q for q in range(t.num_qubits) if q not in failed_q and (a, q) not in t["cz"] and q != a)
    qc = QuantumCircuit(QuantumRegister(t.num_qubits, "q"), ClassicalRegister(1, "c"))
    qc.measure(a, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.cz(a, far)                             # not a coupler: only visible inside the block
    assert any(f"cz({a}, {far}) not supported" in p for p in ibm.problems([qc], t))
    qc2 = QuantumCircuit(QuantumRegister(t.num_qubits, "q"), ClassicalRegister(1, "c"))
    qc2.measure(a, 0)
    with qc2.if_test((qc2.clbits[0], 1)):
        qc2.x(b)
    assert ibm.problems([qc2], t) == []


def test_submission_path_offline(tmp_path):
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    dev = Dv.fetch_fake("FakeMarrakesh")
    out = str(tmp_path / "j")
    FF.build(dev, [("tp1", FF.teleport_test(1), dict(success={1: 0}, bound=0.75))], out, shots=200,
             sentinel_spec=dict(kind="echo_clifford", D=24))

    class Svc:
        allows_fake = True

        def backend(self, name):
            b = FakeMarrakesh()
            b.name = name
            return b
    jid = ibm.submit(Svc(), os.path.join(out, "job"))
    sub = json.load(open(os.path.join(out, "job", "submission.json")))
    assert jid and "twirling off" in sub["options"] and sub["sampler_options"]["dynamical_decoupling.enable"] is False


# ---- repeat-until-success (2026-11-16 evening): form "blocks", CY / CRY lowering, the neuron chain

def test_lowering_is_exact():
    from qiskit.quantum_info import Operator
    for c, t in ((0, 1), (1, 0)):
        q = QuantumCircuit(2)
        q.cy(c, t)
        assert Operator(q).equiv(Operator(FF.lower(q)))
        q = QuantumCircuit(2)
        q.cry(0.73, c, t)
        assert Operator(q).equiv(Operator(FF.lower(q)))
    assert {i.operation.name for i in FF.lower(q).data} <= {"ry", "h", "cz"}


def test_blocks_form_and_ibm_rules():
    qc, spec = FF.rus_chain(2, 1.1, m=3)
    a = FF.analyse(qc, form="blocks")
    assert a["conditionals"] == 4 and a["n"] == 4 and a["cz_in_blocks"] == 12
    with pytest.raises(FF.Refused, match="only X or Z"):
        FF.analyse(qc)                                   # the narrow form still refuses sub-circuits
    for bad, word in (("measure", "measure"), ("reset", "reset"), ("nest", "inside a conditional")):
        q = _qc(2, 2)
        q.measure(0, 0)
        with q.if_test((q.clbits[0], 1)):
            if bad == "measure":
                q.measure(1, 1)
            elif bad == "reset":
                q.reset(1)
            else:
                with q.if_test((q.clbits[0], 1)):
                    q.x(1)
        if bad != "measure":
            q.measure(1, 1)
        with pytest.raises(FF.Refused, match=word):
            FF.analyse(q, form="blocks")


def test_rus_exact_and_noiseless_card(tmp_path):
    for L in (1, 2, 3):
        for th in (0.8, 1.1):
            _, sp = FF.rus_chain(L, th)
            assert abs(sp["exact"] - FF.rus_exact_statevector(L, th)) < 1e-12
    dev = Dv.fetch_fake("FakeFez")
    items = []
    for L in (1, 2):
        qc, sp = FF.rus_chain(L, 1.1, m=3)
        items.append((f"rus{L}", qc, sp))
    out = str(tmp_path / "r")
    rep = FF.build(dev, items, out, shots=4000, sentinel_spec=dict(kind="echo_clifford", D=24))
    assert rep["items"]["rus2"]["cz_in_blocks"] == 12 and rep["circuits"] == 2 * 3 + 4
    FF.run_local(out)
    cards = FF.cards(out)
    c1, c2 = cards["rus1"], cards["rus2"]
    # desk (notebook rus_neuron_desk.txt, 20000 shots): L 1 ff 0.919 (still failed 0.034), post 0.935, off 0.793;
    # L 2 ff 0.849 (0.064), post 0.851, off 0.634
    assert abs(c1["ff"]["p1"] - 0.919) < 0.02 and abs(c1["ff"]["still_failed_after_m"] - 0.034) < 0.015
    assert abs(c1["off"]["p1"] - 0.793) < 0.03 and abs(c2["off"]["p1"] - 0.634) < 0.03
    assert abs(c2["ff"]["p1"] - 0.849) < 0.03 and abs(c2["post"]["p1"] - 0.851) < 0.04
    assert c2["post"]["discarded"] > 0 and c2["ff"]["discarded"] == 0 and c2["off"]["discarded"] == 0
    assert c1["floor"]["all_zero"] == 1.0 and c2["floor"]["all_zero"] == 1.0


def test_pre_submit_check_walks_nested_blocks():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeFez
    t = FakeFez().target
    _, failed_c, failed_q = Dv.failed_elements(t)
    a, b = next(qa for qa in t["cz"] if qa and tuple(sorted(qa)) not in {tuple(sorted(c)) for c in failed_c}
                and not set(qa) & set(failed_q))
    far = next(q for q in range(t.num_qubits) if q not in failed_q and (a, q) not in t["cz"] and q != a)
    qc = QuantumCircuit(QuantumRegister(t.num_qubits, "q"), ClassicalRegister(1, "c"))
    qc.measure(a, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(b)
        with qc.if_test((qc.clbits[0], 1)):
            qc.cz(a, far)                        # two levels down
    assert any(f"cz({a}, {far}) not supported" in p for p in ibm.problems([qc], t))


def test_explicit_off_arm(tmp_path):
    """A circuit without conditionals (repeat-until-success written coherently) with its own off arm (one try)."""
    from qiskit import transpile
    dev = Dv.fetch_fake("FakeFez")

    def line(tries):
        qc = _qc(3, 3)
        qc.ry(2.2, 0)
        qc.cz(0, 1)
        if tries > 1:
            qc.cx(1, 2)
            qc.cx(2, 1)
        qc.measure([0, 1, 2], [0, 1, 2])
        return transpile(qc, basis_gates=["cz", "rz", "sx", "x", "measure"], optimization_level=0)
    out = str(tmp_path / "o")
    rep = FF.build(dev, [("c", line(2), dict(success={2: 0}, off_circuit=line(1)))], out, shots=200,
                   sentinel_spec=dict(kind="echo_clifford", D=24))
    assert rep["items"]["c"]["cz"] == 3 and rep["items"]["c"]["off_cz"] == 1
    rec = json.load(open(os.path.join(out, "runs", "ff_c_off.json")))
    assert rec["off_explicit"] is True
    with pytest.raises(FF.Refused, match="off arm"):
        FF.build(dev, [("d", line(2), dict(success={2: 0}, off_circuit=_qc(2, 2)))], str(tmp_path / "p"))
