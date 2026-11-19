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


# ---- top-level delays (2026-11-17): a delay holds a schedule; refused inside a conditional block

def _with_delays(k=2):
    """The teleport test with top-level delays on every qubit before each conditional and before the last
    measurement (dt units, multiples of 16)."""
    src = FF.teleport_test(k)
    out = src.copy_empty_like()
    for ins in src.data:
        if ins.operation.name in ("if_else",) or (ins.operation.name == "measure" and ins is src.data[-1]):
            for q in range(src.num_qubits):
                out.delay(160, q, unit="dt")
        out.append(ins.operation, ins.qubits, ins.clbits)
    return src, out


def test_top_level_delay_accepted_and_kept(tmp_path):
    from qiskit import qpy
    dev = Dv.fetch_fake("FakeMarrakesh")
    src, qc = _with_delays(2)
    assert sum(1 for i in qc.data if i.operation.name == "delay") > 0
    assert FF.analyse(qc) == FF.analyse(src)                   # a delay is not a gate: same counts
    out = str(tmp_path / "d")
    FF.build(dev, [("tp2", qc, dict(success={2: 0}, bound=0.75))], out, shots=400,
             sentinel_spec=dict(kind="echo_clifford", D=24))
    circs = qpy.load(open(os.path.join(out, "job", "circuits.qpy"), "rb"))
    n_in = sum(1 for i in qc.data if i.operation.name == "delay")
    kept = [sum(1 for n, _ in FF.walk(c) if n == "delay") for c in circs]
    assert kept.count(n_in) == 3                                   # ff, off and floor keep every delay
    # the pre-submit check against an IBM target accepts top-level delays
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    assert ibm.problems([c for c, k in zip(circs, kept) if k == n_in], FakeMarrakesh().target) == []
    # noiselessly the delays change nothing: the same answer card as without them
    out0 = str(tmp_path / "n")
    FF.build(dev, [("tp2", src, dict(success={2: 0}, bound=0.75))], out0, shots=400,
             sentinel_spec=dict(kind="echo_clifford", D=24))
    FF.run_local(out)
    FF.run_local(out0)
    c1, c0 = FF.cards(out)["tp2"], FF.cards(out0)["tp2"]
    assert c1["ff"]["share"] == c0["ff"]["share"] == 1.0
    assert c1["floor"]["all_zero"] == c0["floor"]["all_zero"] == 1.0


def test_delay_exact_distribution_unchanged():
    from qiskit.quantum_info import Statevector
    qc = _qc(2, 2)
    qc.h(0)
    qc.delay(320, 0, unit="dt")
    qc.cz(0, 1)
    qc.delay(48, 1, unit="dt")
    qc.h(1)
    ref = _qc(2, 2)
    ref.h(0)
    ref.cz(0, 1)
    ref.h(1)
    qc.measure([0, 1], [0, 1])
    FF.analyse(qc)                                                 # accepted
    a = Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict()
    b = Statevector(ref).probabilities_dict()
    assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-12 for k in a)


def test_delay_inside_block_refused():
    for form in ("pauli", "blocks"):
        qc = _qc(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.delay(160, 1, unit="dt")
            qc.x(1)
        qc.measure(1, 1)
        with pytest.raises(FF.Refused, match="delay inside a conditional block"):
            FF.analyse(qc, form=form)


# ---- 2026-11-17: placement given from outside, and bundles (the packing test)

def test_given_chain_accepted_and_refused(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    ch = FF.place(dev, 3)["chain"]
    other = [q + 1 for q in ch] if all(dev.qubit_ok(q + 1) for q in ch) else ch
    rep = FF.build(dev, [("tp2", FF.teleport_test(2), dict(success={2: 0}))], str(tmp_path / "g"),
                   sentinel_spec=dict(kind="echo_clifford", D=24), chains={"tp2": other})
    assert rep["items"]["tp2"]["chain"] == other
    rec = json.load(open(os.path.join(str(tmp_path / "g"), "runs", "ff_tp2.json")))
    assert rec["chain_rule"]["rule"] == "given"
    bad = [ch[0], ch[2], ch[1]]                                   # not joined in this order
    with pytest.raises(FF.Refused, match="working couplers"):
        FF.build(dev, [("tp2", FF.teleport_test(2), dict(success={2: 0}))], str(tmp_path / "h"),
                 chains={"tp2": bad})
    with pytest.raises(FF.Refused, match="has 2 qubits"):
        FF.given_chain(dev, ch[:2], 3)
    with pytest.raises(FF.Refused, match="unknown items"):
        FF.build(dev, [("tp2", FF.teleport_test(2), dict(success={2: 0}))], str(tmp_path / "i"), chains={"x": ch})


def test_bundle_two_copies_give_two_single_cards(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    qc, sp = FF.rus_chain(1, 1.1, m=3)
    items = [("tpa", FF.teleport_test(2), dict(success={2: 0}, bound=0.75)),
             ("tpb", FF.teleport_test(2), dict(success={2: 0}, bound=0.75)),
             ("rusa", qc, sp), ("rusb", qc, sp)]
    out = str(tmp_path / "b")
    rep = FF.build_bundle(dev, items, out, shots=4000, sentinel_spec=dict(kind="echo_clifford", D=24),
                          allow_conditional_neighbours=True)   # the mechanics, not the rule (2026-11-18)
    assert rep["circuits"] == 3 + 4
    segs = [set(rep["items"][t]["chain"]) for t in ("tpa", "tpb", "rusa", "rusb")]
    assert all(not (a & b) for i, a in enumerate(segs) for b in segs[i + 1:])
    single = str(tmp_path / "s")
    FF.build(dev, [("rus1", qc, sp)], single, shots=4000, sentinel_spec=dict(kind="echo_clifford", D=24),
             chains={"rus1": rep["items"]["rusa"]["chain"]})
    FF.run_local(out)
    FF.run_local(single)
    cards, cs = FF.cards(out), FF.cards(single)
    assert set(cards) == {"tpa", "tpb", "rusa", "rusb"}
    for t in ("tpa", "tpb"):                                       # noiseless: deterministic success, floor all zero
        assert cards[t]["ff"]["share"] == 1.0 and cards[t]["ff"]["counted"] == 4000
        assert cards[t]["floor"]["all_zero"] == 1.0 and abs(cards[t]["off"]["share"] - 0.5) < 0.05
    for t in ("rusa", "rusb"):                                     # the same distribution as the item alone
        for arm in ("ff", "off"):
            a, b = cards[t][arm], cs["rus1"][arm]
            se = (a["p1"] * (1 - a["p1"]) / a["counted"] + b["p1"] * (1 - b["p1"]) / b["counted"]) ** 0.5
            assert abs(a["p1"] - b["p1"]) < 4 * se
        assert cards[t]["floor"]["all_zero"] == 1.0 and cards[t]["exact"] == cs["rus1"]["exact"]


def test_bundle_refuses_an_instruction_across_items():
    qc = _qc(4, 4)
    qc.h(0)
    qc.cz(1, 2)                                                    # joins item 0 (qubits 0, 1) and item 1 (2, 3)
    qc.measure(range(4), range(4))
    with pytest.raises(FF.Refused, match="spans sub-items"):
        FF.bundle_acceptance({"ff": qc}, [{0, 1}, {2, 3}], [{0, 1}, {2, 3}])
    qc = _qc(4, 4)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):                            # item 1's qubit conditioned on item 0's bit
        qc.x(2)
    with pytest.raises(FF.Refused, match="spans sub-items"):
        FF.bundle_acceptance({"ff": qc}, [{0, 1}, {2, 3}], [{0, 1}, {2, 3}])
    qc = _qc(4, 4)
    qc.barrier([1, 2])
    with pytest.raises(FF.Refused, match="spans sub-items"):
        FF.bundle_acceptance({"ff": qc}, [{0, 1}, {2, 3}], [{0, 1}, {2, 3}])


def test_bundle_time_model_counts_three_arms_and_the_sentinel(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    reps = []
    for n in (2, 5):
        items = [(f"tp{j}", FF.teleport_test(1), dict(success={1: 0})) for j in range(n)]
        reps.append(FF.build_bundle(dev, items, str(tmp_path / f"t{n}"), sentinel_spec=dict(kind="echo_clifford", D=24),
                                    allow_conditional_neighbours=True))
    assert reps[0]["circuits"] == reps[1]["circuits"] == 3 + 4
    assert reps[0]["qpu_seconds_model"] == reps[1]["qpu_seconds_model"]


def test_run_local_splits_a_large_bundle_into_independent_parts(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    qc, sp = FF.rus_chain(1, 1.1, m=3)
    items = [(f"r{j}", qc, sp) for j in range(9)]                 # 9 x 2 qubits + gaps > LOCAL_MAX_QUBITS
    out = str(tmp_path / "big")
    rep = FF.build_bundle(dev, items, out, shots=2000, sentinel_spec=dict(kind="echo_clifford", D=24),
                          allow_conditional_neighbours=True)
    assert len(rep["chain"]) > FF.LOCAL_MAX_QUBITS
    import qiskit.qpy as qpy
    ff = qpy.load(open(os.path.join(out, "job", "circuits.qpy"), "rb"))[0]
    assert len(FF.components(ff)) == 9
    FF.run_local(out)
    cards = FF.cards(out)
    for j in range(9):                                             # desk value of L 1, m 3: ff 0.919
        c = cards[f"r{j}"]
        assert abs(c["ff"]["p1"] - 0.919) < 0.03 and c["floor"]["all_zero"] == 1.0


# ---- v0.2 prototype (2026-11-18): the three build rules

def _pack_items():
    """Ten items as tonight's P-A (rd1..5, ea1..5) when the notebook is on TOQB_DIR; else small rus and teleport items."""
    nb = os.environ.get("TOQB_DIR")
    if nb and os.path.isdir(os.path.join(nb, "notes", "early_2026-11-17")):
        import sys
        for d in ("notes/early_2026-11-17", "notes/compound_2026-11-17", "notes/rus_coherent_2026-11-17",
                  "notes/timebudget_2026-11-17", "gpu"):
            p = os.path.join(nb, d)
            if p not in sys.path:
                sys.path.insert(0, p)
        import compound as CP
        import early as EA
        return CP.items() + EA.items()
    out = []
    for L in (1, 2):
        qc, sp = FF.rus_chain(L, 1.1, m=2)
        out.append((f"rus{L}", qc, sp))
    out += [(f"tp{k}", FF.teleport_test(k), dict(success={k: 0})) for k in (1, 2)]
    return out


def test_pack_fixed_matches_the_single_build(tmp_path):
    dev = Dv.fetch_fake("FakeMarrakesh")
    items = _pack_items()
    a, b = str(tmp_path / "single"), str(tmp_path / "packed")
    pl = FF.bundle_place(dev, [FF.analyse(qc, form=sp.get("form", "pauli"))["n"] for _, qc, sp in items])
    chains = {t: [pl["chain"][k] for k in s] for (t, _, _), s in zip(items, pl["segments"])}
    rs = FF.build(dev, items, a, sentinel_spec=dict(kind="echo_clifford", D=24), chains=chains, max_seconds=90)
    rp = FF.build(dev, items, b, sentinel_spec=dict(kind="echo_clifford", D=24), pack_fixed=True, max_seconds=90)
    n = len(items)
    assert rs["circuits"] == 3 * n + 4 and rp["circuits"] == n + 2 + 4
    from psf_os import job as J
    assert rp["qpu_seconds_model"] == round(J.qpu_seconds(n + 2 + 4, 4000), 1) < rs["qpu_seconds_model"]
    man = json.load(open(os.path.join(b, "job", "manifest.json")))
    assert man["pack_fixed"]["items"] == [t for t, _, _ in items] and set(man["conditionals_per_item"]) == set(rp["items"])
    # the ff arms: the same bytes as the single build's on the same qubits
    for t, _, _ in items:
        assert open(os.path.join(a, "runs", f"ff_{t}.qpy"), "rb").read() == open(os.path.join(b, "runs", f"ff_{t}.qpy"),
                                                                                "rb").read()
    # the packed off arm restricted to each item: the same exact distribution as the single off arm
    from qiskit import qpy
    (poff,) = qpy.load(open(os.path.join(b, "runs", f"ff_{FF.FIXED_PACK_TAG}_off.qpy"), "rb"))
    table = json.load(open(os.path.join(b, "runs", f"ff_{FF.FIXED_PACK_TAG}_off.json")))["fixed_pack_table"]
    for t, _, _ in items:
        (soff,) = qpy.load(open(os.path.join(a, "runs", f"ff_{t}_off.qpy"), "rb"))
        info = table[t]
        mine = FF.restrict(poff, info["qubits"], list(range(info["bit_offset"], info["bit_offset"] + info["clbits"])))
        ref = FF.restrict(soff, info["qubits"], list(range(info["clbits"])))
        dm, dr = FF.exact_distribution(mine), FF.exact_distribution(ref)
        assert max(abs(dm.get(k, 0) - dr.get(k, 0)) for k in set(dm) | set(dr)) < 1e-12
    # noiseless: every card of the packed build reads as the single build's (same ff circuits; off within 4 SE; floor 1)
    FF.run_local(a)
    FF.run_local(b)
    ca, cb = FF.cards(a), FF.cards(b)
    assert set(ca) == set(cb) == {t for t, _, _ in items}
    for t in ca:
        assert cb[t]["floor"]["all_zero"] == ca[t]["floor"]["all_zero"] == 1.0
        for arm in ("ff", "off"):
            x, y = ca[t][arm], cb[t][arm]
            key = "p1" if "p1" in x else "share"
            p, q, nx, ny = x[key], y[key], x["counted"], y["counted"]
            se = (p * (1 - p) / nx + q * (1 - q) / ny) ** 0.5
            assert abs(p - q) <= 4 * se + 1e-12


def test_conditional_neighbours_refused_unless_allowed(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    qc, sp = FF.rus_chain(2, 1.1, m=2)
    two = [("a", qc, sp), ("b", qc, sp)]
    with pytest.raises(FF.Refused, match="hold conditional operations.*2026-11-17.*2026-11-18"):
        FF.build_bundle(dev, two, str(tmp_path / "x"), sentinel_spec=dict(kind="echo_clifford", D=24))
    assert not os.path.exists(str(tmp_path / "x"))
    out = str(tmp_path / "y")
    FF.build_bundle(dev, two, out, sentinel_spec=dict(kind="echo_clifford", D=24), allow_conditional_neighbours=True)
    man = json.load(open(os.path.join(out, "job", "manifest.json")))
    assert man["conditional_neighbours"] == dict(items=["a", "b"], allowed=True, used=True)
    assert man["conditionals_per_item"] == {"a": 2, "b": 2}
    # one conditional item beside a fixed one (tonight's B1): built without the flag
    off = FF.strip_conditionals(FF.lower(qc))
    out = str(tmp_path / "z")
    FF.build_bundle(dev, [("a", qc, sp), ("b", off, dict(success={4: 0}, form="blocks"))], out,
                    sentinel_spec=dict(kind="echo_clifford", D=24))
    man = json.load(open(os.path.join(out, "job", "manifest.json")))
    assert man["conditional_neighbours"] == dict(items=["a"], allowed=False, used=False)


def test_several_conditionals_inside_one_item_are_not_neighbours(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    qc, sp = FF.rus_chain(3, 1.1, m=3)                           # 6 conditionals in one item
    tp = FF.strip_conditionals(FF.teleport_test(1))
    out = str(tmp_path / "w")
    FF.build_bundle(dev, [("big", qc, sp), ("fx", tp, dict(success={1: 0}))], out,
                    sentinel_spec=dict(kind="echo_clifford", D=24))
    man = json.load(open(os.path.join(out, "job", "manifest.json")))
    assert man["conditionals_per_item"] == {"big": 6, "fx": 0} and man["conditional_neighbours"]["used"] is False


def test_build_environment_recorded_and_compared(tmp_path, capsys):
    from psf_os import job as J
    dev = Dv.fetch_fake("FakeFez")
    out = str(tmp_path / "e")
    FF.build(dev, [("tp1", FF.teleport_test(1), dict(success={1: 0}))], out, sentinel_spec=dict(kind="echo_clifford", D=24))
    man = json.load(open(os.path.join(out, "job", "manifest.json")))
    env = man["env"]
    assert set(env) == {"python", "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy"} and env["qiskit"]
    assert J.verify_env(man)["same"] is True and J.verify_env(os.path.join(out, "job"))["same"] is True
    man["env"] = dict(env, **{"qiskit-aer": "0.0.1"})
    r = J.verify_env(man)
    assert r["same"] is False and "qiskit-aer" in r["differ"] and "sentinel" in r["note"]
    assert "verify_env:" in capsys.readouterr().out
    del man["env"]
    assert J.verify_env(man)["same"] is None


def test_pack_fixed_on_tonights_pa_record_gives_the_same_ff_bytes(tmp_path):
    nb = os.environ.get("TOQB_DIR")
    pa = os.path.join(nb or "", "results", "parallel-night-2026-11-18", "pa")
    if not nb or not os.path.isdir(pa):
        pytest.skip("needs TOQB_DIR with results/parallel-night-2026-11-18")
    from psf_os.device import Device
    dev = Device.load(os.path.join(nb, "results", "parallel-night-2026-11-18", "fetch_in", "ibm_marrakesh", "dev"))
    items = _pack_items()
    out = str(tmp_path / "pf")
    FF.build(dev, items, out, sentinel_spec=dict(kind="echo_clifford", D=24), pack_fixed=True)
    for t, _, _ in items:
        a = open(os.path.join(pa, "runs", f"ff_{t}.qpy"), "rb").read()
        b = open(os.path.join(out, "runs", f"ff_{t}.qpy"), "rb").read()
        if a != b:
            # 2026-11-19 (workplace, and the workplace VS's check): the bytes follow the versions that wrote them (the
            # QPY format: qiskit 2.5.2 writes 17, 2.6.1 writes 18; an angle's last digit, 1.8e-15 at most, with another
            # numpy / Python), so across machines the same gates are required instead, both files read by this qiskit
            # (psf_os.gatehash: parameters to 1e-10, the conditional blocks included). On the machine that wrote the
            # records the bytes are equal and this branch is not taken
            from qiskit import qpy
            from psf_os.gatehash import circuits_hash
            ca = qpy.load(open(os.path.join(pa, "runs", f"ff_{t}.qpy"), "rb"))
            cb = qpy.load(open(os.path.join(out, "runs", f"ff_{t}.qpy"), "rb"))
            assert circuits_hash(ca) == circuits_hash(cb), t


def test_bundle_gap_and_bit_order(tmp_path):
    """Interference tests (2026-11-18 evening): gap and bit_order vary placement and bit order apart; the defaults give
    the bundle as before, byte for byte."""
    dev = Dv.fetch_fake("FakeFez")
    qc, sp = FF.rus_chain(2, 1.1, m=2)
    two = [("a", qc, sp), ("b", qc, sp)]
    kw = dict(sentinel_spec=dict(kind="echo_clifford", D=24), allow_conditional_neighbours=True)
    r0 = FF.build_bundle(dev, two, str(tmp_path / "d0"), **kw)
    r1 = FF.build_bundle(dev, two, str(tmp_path / "d1"), gap=1, bit_order=["a", "b"], **kw)
    for f in ("ff_pack.qpy", "ff_pack_off.qpy", "ff_pack_floor.qpy"):
        assert open(str(tmp_path / "d0" / "runs" / f), "rb").read() == open(str(tmp_path / "d1" / "runs" / f), "rb").read()
    m1 = json.load(open(str(tmp_path / "d1" / "job" / "manifest.json")))
    assert "bundle_gap" not in m1 and "bit_order" not in m1
    r3 = FF.build_bundle(dev, two, str(tmp_path / "g3"), gap=3, **kw)
    ch = r3["chain"]
    assert ch.index(r3["items"]["b"]["chain"][0]) - ch.index(r3["items"]["a"]["chain"][-1]) - 1 >= 3
    assert json.load(open(str(tmp_path / "g3" / "job" / "manifest.json")))["bundle_gap"] == 3
    rb = FF.build_bundle(dev, two, str(tmp_path / "bo"), bit_order=["b", "a"], **kw)
    assert rb["items"]["a"]["chain"] == r0["items"]["a"]["chain"]                 # placement unchanged
    assert rb["items"]["b"]["bit_offset"] == 0 and rb["items"]["a"]["bit_offset"] == 5
    FF.run_local(str(tmp_path / "bo"))
    c = FF.cards(str(tmp_path / "bo"))
    assert abs(c["a"]["ff"]["p1"] - c["b"]["ff"]["p1"]) < 0.04 and c["a"]["floor"]["all_zero"] == 1.0
    with pytest.raises(FF.Refused, match="bit_order"):
        FF.build_bundle(dev, two, str(tmp_path / "x"), bit_order=["a"], **kw)
    with pytest.raises(FF.Refused, match="gap"):
        FF.build_bundle(dev, two, str(tmp_path / "y"), gap=0, **kw)
