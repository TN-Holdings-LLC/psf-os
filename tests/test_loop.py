"""psf_os.loop (v0.2, 2026-11-19): the rule with the rungs +M then +F, the ledger, and the night runner with a fake
experiment (no circuit, no IBM): rounds until a fixed point, the round limit, the budget before a round is built, a
check FAIL, a pending job resumed without a second submission, verify SAME on the records and DIFFERENT after a
change."""
import json
import os

import pytest

from psf_os import device as Dv
from psf_os import loop as LP


def row(stack, verdict="NOT SHOWN", **k):
    return dict(stack=list(stack), verdict=verdict, overshoot=verdict == "OVERSHOOT", **k)


def rnd(k, rows, dr=None, spent=0.0):
    return dict(round=k, device="ibm_fake", rows=rows, dr=dr or {}, model_seconds_so_far=spent)


# ------------------------------------------------------------------------------------------------ the rule

def test_rule_r1_r2_and_rungs_in_order():
    dr = {"5": dict(gap=0.10, two_se=0.02, sig=True), "7": dict(gap=0.10, two_se=0.02, sig=True)}
    r = rnd(1, {"1": row(["P1"], "OVERSHOOT", output_qubit=5),
                "2": row(["P1"], "PRESCRIPTION HELPS", output_qubit=5),           # helps and Dr against P1: M first
                "3": row(["P1", "M"], "PRESCRIPTION HELPS", output_qubit=7),      # M already: F next
                "4": row(["P2", "B"], "PRESCRIPTION HELPS", share_gap=0.05, share_gap_two_se=0.02,
                         output_qubit_nf=7)}, dr)
    R = LP.next_stacks([r], q_choice=lambda key, row_, d: 2)
    assert R["stacks"] == {"1": ["P2", "B"], "2": ["P1", "M"], "3": ["P2", "M", "F"], "4": ["P2", "B", "Q2"]}
    assert R["why"]["1"].startswith("R1") and R["why"]["2"].startswith("RM") and R["why"]["3"].startswith("RF")
    assert R["why"]["4"] == "R2 frames -> +Q2" and not R["stop"]
    # the other order: F before M
    R2 = LP.next_stacks([r], rungs=("F", "M"), q_choice=lambda key, row_, d: 2)
    assert R2["stacks"]["2"] == ["P2", "F"]
    # R2 without a choice keeps; the gap not significant keeps
    R3 = LP.next_stacks([r], q_choice=lambda key, row_, d: None)
    assert R3["stacks"]["4"] == ["P2", "B"]
    with pytest.raises(ValueError):
        LP.next_stacks([r], rungs=("X",))


def test_rule_frame_direction():
    up = {"5": dict(gap=0.10, two_se=0.02, sig=True)}          # |1> reads worse: against P1, not against P2
    down = {"5": dict(gap=-0.10, two_se=0.02, sig=True)}
    for base, dr, switched in (("P1", up, True), ("P1", down, False), ("P2", down, True), ("P2", up, False)):
        R = LP.next_stacks([rnd(1, {"1": row([base], "NOT SHOWN", output_qubit=5)}, dr)], rungs=("F",))
        assert (R["stacks"]["1"][-1] == "F") is switched, (base, dr)


def test_rule_stop_loss_and_stops():
    a = rnd(1, {"1": row(["P2", "M", "F"], "NOT SHOWN"), "2": row(["P1"], "PRESCRIPTION HELPS")})
    b = rnd(2, {"1": row(["P2", "M", "F"], "NOT SHOWN"), "2": row(["P1", "M"], "PRESCRIPTION HELPS")})
    R = LP.next_stacks([a, b])
    assert R["stacks"]["1"] == ["P1", "M"]                     # F removed: the base switches back
    assert R["why"]["1"].startswith("R3")
    c = rnd(1, {"1": row(["P1", "M"], "PRESCRIPTION HELPS")})
    R = LP.next_stacks([c])
    assert R["stop"] and R["stop_reason"].startswith("fixed point")
    d = rnd(4, {"1": row(["P1"], "PRESCRIPTION HELPS")})
    R = LP.next_stacks([d], k_max=4)
    assert R["stop"] and R["stop_reason"] == "round limit K = 4"
    e = rnd(1, {"1": row(["P1"], "PRESCRIPTION HELPS")}, spent=200.0)
    R = LP.next_stacks([e], next_cost=50.0, budget=240.0)
    assert R["stop"] and R["stop_reason"].startswith("budget")


# ------------------------------------------------------------------------------------------------ the ledger

def test_ledger_start():
    fam = {"P1": "flip", "P2": "flip", "P0": "P0"}.get
    last = dict(night="n1", device="ibm_fake", calibration_time="t1",
                last_stacks={"1": ["P2", "B"], "2": ["P1", "M"], "3": ["P0"]})
    tonight = {"1": ["P1"], "2": ["P1"], "3": ["P1"]}
    st, dec = LP.ledger_start([], "ibm_fake", "t1", tonight, fam)
    assert st == tonight and dec["why"] == "no ledger" and not dec["used"]
    st, dec = LP.ledger_start([last], "ibm_other", "t1", tonight, fam)
    assert st == tonight and dec["why"].startswith("other device")
    st, dec = LP.ledger_start([last], "ibm_fake", "t2", tonight, fam)
    assert st == tonight and dec["why"].startswith("other calibration")
    st, dec = LP.ledger_start([last], "ibm_fake", None, tonight, fam)
    assert st == tonight and dec["why"].startswith("other calibration")
    st, dec = LP.ledger_start([last], "ibm_fake", "t1", tonight, fam)
    assert st == {"1": ["P2", "B"], "2": ["P1", "M"], "3": ["P1"]} and dec["used"]
    assert dec["per_key"]["3"].startswith("diagnostics disagree")
    e = LP.ledger_entry("n2", "ibm_fake", "t1", LP.RUNGS, [rnd(1, {"1": row(["P1"], "NOT SHOWN", g=0.01)})], dec)
    assert e["last_stacks"] == {"1": ["P1"]} and e["rounds_run"] == 1 and e["ledger_used"]


# ------------------------------------------------------------------------------------------------ the runner

TABLE = {"P1": "PRESCRIPTION HELPS", "P1+M": "PRESCRIPTION HELPS", "P2": "NOT SHOWN"}


class Fake:
    """A fake experiment: a 'job' is two files; the verdict of a stack is TABLE's; seconds per job given."""
    name = "fake"
    order = ["ibm_fake"]
    k_max, budget_s = 4, 240.0

    def __init__(self, seconds=20.0, fail_check_round=None, pending_first=False):
        self.seconds, self.fail_check_round, self.pending_first = seconds, fail_check_round, pending_first
        self.simulated = []

    def fetch(self, base):
        raise AssertionError("no fetch in dry runs")

    def calibration_time(self, devdir):
        return "cal-1"

    def _job(self, folder, payload):
        os.makedirs(os.path.join(folder, "job"))
        os.makedirs(os.path.join(folder, "runs"))
        open(os.path.join(folder, "job", "circuits.qpy"), "wb").write(json.dumps(payload, sort_keys=True).encode())
        json.dump(dict(written="x", payload=payload), open(os.path.join(folder, "job", "manifest.json"), "w"))
        json.dump(dict(qpu_seconds_model=self.seconds), open(os.path.join(folder, "build.json"), "w"))

    def diagnostics(self):
        def build(devdir, folder, dry):
            self._job(folder, dict(diag=True))

        def score(folder):
            json.dump(dict(ok=True), open(os.path.join(folder, "d_scores.json"), "w"))
        return [("dw", build, lambda f: True, score, "d_scores.json")]

    def first_rule(self, out, diag, device):
        return dict(round=1, device=device, stacks={"1": ["P1"], "2": ["P2"]}, why={"1": "pick", "2": "pick"},
                    model_seconds_so_far=self.seconds)

    @staticmethod
    def family(base):
        return {"P1": "flip", "P2": "flip"}.get(base, base)

    def build(self, devdir, folder, rule_path, dry):
        R = json.load(open(rule_path))
        self._job(folder, dict(stacks=R["stacks"], round=R["round"]))
        json.dump(R, open(os.path.join(folder, "rule.json"), "w"))
        return True

    def check(self, folder):
        R = json.load(open(os.path.join(folder, "rule.json")))
        return R["round"] != self.fail_check_round

    def score(self, folder):
        R = json.load(open(os.path.join(folder, "rule.json")))
        cnt = json.load(open(os.path.join(folder, "job", "raw_counts.json")))["rows"][0]["counts"]
        rows = {k: row(st, TABLE.get("+".join(st), "NOT SHOWN"), g=cnt["n"] / 1000, output_qubit=5)
                for k, st in R["stacks"].items()}
        out = dict(round=R["round"], device=R["device"], stacks=R["stacks"], rows=rows, dr={},
                   model_seconds_so_far=R.get("model_seconds_so_far", 0.0) + self.seconds)
        json.dump(out, open(os.path.join(folder, "round_scores.json"), "w"), indent=1)
        return out

    def q_choice(self, rf, key, r, dr):
        return None

    def compound(self, rounds):
        return dict(rounds=len(rounds))

    def predictions(self, out):
        return dict(rounds=len([d for d in os.listdir(out) if d.startswith("r") and d[1:].isdigit()]))

    def simulate(self, folder, noise, seed):
        tag = os.path.basename(folder)
        self.simulated.append(tag)
        json.dump(dict(job_id=f"local-{seed}", local=True), open(os.path.join(folder, "job", "submission.json"), "w"))
        if self.pending_first and self.simulated.count(tag) == 1 and tag == "r1":
            return                                              # the result is not there yet: a pending job
        json.dump(dict(rows=[dict(pub=0, counts={"n": 100 + seed % 7})]),
                  open(os.path.join(folder, "job", "raw_counts.json"), "w"))

    def dry_seed(self, tag):
        return 1000 + (0 if tag == "dw" else int(tag[1:]))


@pytest.fixture
def devdir(tmp_path):
    d = str(tmp_path / "dev")
    Dv.fetch_fake("FakeFez").save(d)
    return d


def test_night_to_a_fixed_point_and_verify(tmp_path, devdir):
    exp = Fake()
    out = str(tmp_path / "n")
    led = str(tmp_path / "ledger.json")
    res = LP.start(out, exp, dry_devdir=devdir, ledger=led).go()
    assert res["reason"].startswith("the rule: fixed point") and res["rounds"] == 3
    R2 = json.load(open(os.path.join(out, "RULE2.json")))
    assert R2["stacks"] == {"1": ["P1", "M"], "2": ["P2"]}
    assert json.load(open(os.path.join(out, "RULE3.json")))["stacks"]["2"] == ["P0"]       # R3: NOT SHOWN twice
    assert exp.simulated == ["dw", "r1", "r2", "r3"]
    assert json.load(open(os.path.join(out, "predictions.json"))) == dict(rounds=3)
    assert json.load(open(led))[-1]["last_stacks"] == {"1": ["P1", "M"], "2": ["P0"]}
    assert LP.verify(out, Fake(), resim=True)
    # a changed raw count is seen
    p = os.path.join(out, "r1", "job", "raw_counts.json")
    d = json.load(open(p))
    d["rows"][0]["counts"]["n"] += 1
    json.dump(d, open(p, "w"))
    assert not LP.verify(out, Fake(), resim=True)
    # the second night starts from the ledger: same device and calibration time, family agrees
    out2 = str(tmp_path / "n2")
    LP.start(out2, Fake(), dry_devdir=devdir, ledger=led).go()
    R1 = json.load(open(os.path.join(out2, "RULE1.json")))
    assert R1["ledger"]["used"] and R1["stacks"]["1"] == ["P1", "M"] and R1["diagnostics_stacks"]["1"] == ["P1"]
    assert R1["stacks"]["2"] == ["P2"] and R1["ledger"]["per_key"]["2"].startswith("diagnostics disagree")
    assert LP.verify(out2, Fake())


def test_round_limit_budget_and_check_fail(tmp_path, devdir):
    res = LP.start(str(tmp_path / "k"), Fake(), dry_devdir=devdir, k_max=1).go()
    assert res["reason"] == "the rule: round limit K = 1 (after round 1)"
    out = str(tmp_path / "b")
    res = LP.start(out, Fake(seconds=30.0), dry_devdir=devdir, budget=70.0).go()
    assert res["reason"].startswith("budget") and res["rounds"] == 1
    assert not os.path.exists(os.path.join(out, "r2"))                  # not built
    assert json.load(open(os.path.join(out, "budget_r2.json")))["estimate_s"] == 30.0
    out = str(tmp_path / "c")
    exp = Fake(fail_check_round=2)
    res = LP.start(out, exp, dry_devdir=devdir).go()
    assert res["reason"] == "r2: check FAIL (check.txt); not sent"
    assert os.path.isdir(os.path.join(out, "r2", "job_NOT_SENT")) and "r2" not in exp.simulated


def test_resume_never_submits_twice(tmp_path, devdir):
    out = str(tmp_path / "p")
    exp = Fake(pending_first=True)
    res = LP.start(out, exp, dry_devdir=devdir).go()
    assert res["resumable"] and "dry run has no result" in res["reason"] and exp.simulated == ["dw", "r1"]
    res = LP.Night(out, exp).go()                                         # still pending: not sent again
    assert res["resumable"] and exp.simulated == ["dw", "r1"]
    json.dump(dict(rows=[dict(pub=0, counts={"n": 100 + 1001 % 7})]),
              open(os.path.join(out, "r1", "job", "raw_counts.json"), "w"))   # the result arrives
    res = LP.Night(out, exp).go()
    assert res["reason"].startswith("the rule: fixed point") and exp.simulated == ["dw", "r1", "r2", "r3"]
    assert len([f for f in os.listdir(out) if f.startswith("stop_")]) == 2  # the earlier stops kept


def test_start_refuses(tmp_path, devdir):
    with pytest.raises(SystemExit, match="zip"):
        LP.start(str(tmp_path / "a"), Fake())
    with pytest.raises(SystemExit, match="dry runs only"):
        LP.start(str(tmp_path / "b"), Fake(), noise={"gamma": 0.1}, zip_path=str(tmp_path / "z.zip"))
    os.makedirs(tmp_path / "c")
    with pytest.raises(SystemExit, match="exists"):
        LP.start(str(tmp_path / "c"), Fake(), dry_devdir=devdir)


# ------------------------------------------------------------------------------------------------ 2026-11-19 (19c)

def test_job_state_by_value():
    from psf_os import ibm
    assert [ibm.job_state(x) for x in ("DONE", "JobStatus.DONE", "QUEUED", "RUNNING", "ERROR", "CANCELLED",
                                       "Cancelled", "weird")] == ["DONE", "DONE", "WAIT", "WAIT", "FAILED", "FAILED",
                                                                 "FAILED", "UNKNOWN"]


def test_collect_reads_the_state_as_a_value(tmp_path, devdir, monkeypatch):
    """Device path: collect_status.json decides; a state not known is not taken for a pending job."""
    out = str(tmp_path / "d")
    n = LP.start(out, Fake(), zip_path=str(tmp_path / "z.zip"))
    folder = os.path.join(out, "r1")
    os.makedirs(os.path.join(folder, "job"))

    def fake_run(state, status):
        def run(cmd, **k):
            json.dump(dict(state=state, status=status), open(os.path.join(folder, "job", "collect_status.json"), "w"))
            return type("R", (), dict(stdout=f"status {status}; run collect again later", stderr="", returncode=0))()
        return run
    for state, status, want in (("WAIT", "QUEUED", "WAIT"), ("FAILED", "ERROR", "FAILED (ERROR)"),
                                ("UNKNOWN", "SOMETHING", "UNKNOWN (SOMETHING)")):
        monkeypatch.setattr(LP.subprocess, "run", fake_run(state, status))
        assert n.collect(folder) == want


def _job(d, circs, roles, version=None):
    from qiskit import qpy
    os.makedirs(d)
    with open(os.path.join(d, "circuits.qpy"), "wb") as f:
        qpy.dump(circs, f, **({"version": version} if version else {}))
    cells = [dict(name=f"c{i}", role=r, points=1) for i, r in enumerate(roles)]
    json.dump(dict(written="t", cells=cells, env={"x": version}), open(os.path.join(d, "manifest.json"), "w"))


def test_same_job_across_versions_and_the_sentinel_apart(tmp_path):
    import math
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit import qpy

    def circ(theta, cond=True):
        qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"), name="c")   # a fixed name: equal bytes
        qc.rz(theta, 0)
        qc.sx(0)
        qc.measure(0, 0)
        if cond:
            with qc.if_test((qc.clbits[0], 1)):
                qc.x(1)
        qc.measure(1, 1)
        return qc
    roles = ["user", "sentinel"]
    v_old = max(10, qpy.QPY_COMPATIBILITY_VERSION) if hasattr(qpy, "QPY_COMPATIBILITY_VERSION") else None
    _job(str(tmp_path / "a"), [circ(0.3), circ(math.pi)], roles)
    _job(str(tmp_path / "b"), [circ(0.3), circ(math.pi)], roles)
    assert LP.same_job(str(tmp_path / "a"), str(tmp_path / "b")) == (None, "bytes")
    if v_old:                                                   # another QPY format: other bytes, the same gates
        _job(str(tmp_path / "c"), [circ(0.3), circ(math.pi)], roles, version=v_old)
        d, how = LP.same_job(str(tmp_path / "a"), str(tmp_path / "c"))
        assert d is None and how.startswith("test circuits' gates")
    _job(str(tmp_path / "e"), [circ(0.3 + 1e-14), circ(-math.pi)], roles)   # last digits; pi written as -pi
    d, how = LP.same_job(str(tmp_path / "a"), str(tmp_path / "e"))
    assert d is None and "sentinel differs" not in how
    _job(str(tmp_path / "f"), [circ(0.3), circ(1.0)], roles)                 # the sentinel alone differs
    d, how = LP.same_job(str(tmp_path / "a"), str(tmp_path / "f"))
    assert d is None and "sentinel differs (beside" in how
    _job(str(tmp_path / "g"), [circ(0.3, cond=False), circ(math.pi)], roles)  # a test circuit differs
    assert LP.same_job(str(tmp_path / "a"), str(tmp_path / "g"))[0] == "test circuits differ (gates)"
