"""psf_os.loop's paired rule (loop-paired-2026-11-21): the incumbent and the candidate in one job, decided by their
difference inside it; the next untried step; the stops; round 1 with the ledger; a night with a fake experiment run to
its stop, verified SAME and DIFFERENT after a change, resumed without a second submission; paired off by default;
feedforward.build ff_only (default: byte for byte as before)."""
import json
import os

import pytest

from psf_os import device as Dv
from psf_os import feedforward as FF
from psf_os import loop as LP

EX = 0.90


def prow(inc, cand, inc_p, cand_p, se=0.005, over=False, step="M", tried=(), pick="P1", inc_info=None,
         cand_info=None):
    return dict(stack=list(inc), candidate=list(cand) if cand else None, step=step if cand else None,
                tried=list(tried), pick=pick, exact=EX, inc_p=inc_p, inc_se=se, cand_p=cand_p, cand_se=se,
                cand_overshoot=over, inc_info=inc_info or {"output_qubit": 5}, cand_info=cand_info or
                {"output_qubit": 5})


def rnd(k, rows, dr=None, spent=0.0):
    return dict(round=k, device="ibm_fake", rows=rows, dr=dr or {}, model_seconds_so_far=spent)


# ------------------------------------------------------------------------------------------------ the decision

def test_decision_every_branch():
    two_se = 2 * (2 * 0.005 ** 2) ** 0.5                                       # 0.01414
    adopt = LP.paired_decision(prow(["P1"], ["P1", "M"], 0.80, 0.80 + two_se + 0.001))
    assert adopt["adopted"] and adopt["word"] == "CANDIDATE ADOPTED"
    rej = LP.paired_decision(prow(["P1"], ["P1", "M"], 0.80, 0.80 - two_se - 0.001))
    assert not rej["adopted"] and rej["word"] == "CANDIDATE NOT ADOPTED"
    kept = LP.paired_decision(prow(["P1"], ["P1", "M"], 0.80, 0.80 + two_se / 2))
    assert not kept["adopted"] and kept["word"] == "INCUMBENT KEPT"
    over = LP.paired_decision(prow(["P0"], ["P1"], 0.70, 0.89, over=True))       # far nearer, but overshoots
    assert not over["adopted"] and over["word"] == "CANDIDATE OVERSHOOTS" and over["d"] > 0
    # distance to exact, not the share: a candidate above exact by more than the incumbent is below is not nearer
    far = LP.paired_decision(prow(["P1"], ["P1", "M"], 0.88, 0.95))
    assert far["d"] < 0 and not far["adopted"]
    assert all(w in LP.PAIRED_WORDS for w in (adopt["word"], rej["word"], kept["word"], over["word"]))


# ------------------------------------------------------------------------------------------------ the next step

def test_next_candidate_every_branch():
    up = {"5": dict(gap=0.10, two_se=0.02, sig=True), "7": dict(gap=0.10, two_se=0.02, sig=True)}
    nc = LP.paired_next_candidate
    assert nc(["P0"], [], {}, {}, pick="P1") == (["P1"], "base")
    assert nc(["P0"], ["base"], {}, {}, pick="P1") == (None, None)                    # the base was not adopted
    assert nc(["P0"], ["base"], {}, {}, pick="P1", overshoot=True) == (["P2", "B"], "B")
    assert nc(["P0"], ["base", "B"], {}, {}, pick="P1", overshoot=True) == (None, None)
    assert nc(["P1"], [], {"output_qubit": 5}, up) == (["P1", "M"], "M")              # M first
    assert nc(["P1"], ["M"], {"output_qubit": 5}, up) == (["P2", "F"], "F")           # M not adopted: F
    assert nc(["P1", "M"], ["M"], {"output_qubit": 5}, up) == (["P2", "M", "F"], "F")
    assert nc(["P1", "M"], ["M"], {"output_qubit": 5}, {}) == (None, None)            # Dr not against: no F
    assert nc(["P1"], [], {"output_qubit": 5}, up, rungs=("F", "M")) == (["P2", "F"], "F")
    assert nc(["P1", "M", "F"], ["M", "F"], {"output_qubit": 5}, up) == (None, None)
    b = {"share_gap": 0.05, "share_gap_two_se": 0.02, "output_qubit_nf": 7}
    assert nc(["P2", "B"], ["B"], b, up, q_choice=lambda info: 2) == (["P2", "B", "Q2"], "Q")
    assert nc(["P2", "B"], ["B"], b, up, q_choice=lambda info: None) == (None, None)
    assert nc(["P2", "B"], ["B"], dict(b, share_gap=0.01), up, q_choice=lambda info: 2) == (None, None)
    assert nc(["P2", "B"], ["B", "Q"], b, up, q_choice=lambda info: 2) == (None, None)
    assert nc(["P2", "B", "Q2"], ["B", "Q"], b, up, q_choice=lambda info: 2) == (None, None)


def test_next_stacks_paired_tried_and_stops():
    up = {"5": dict(gap=0.10, two_se=0.02, sig=True)}
    r = rnd(1, {"1": prow(["P0"], ["P1"], 0.70, 0.80, step="base"),                 # adopted -> next M
                "2": prow(["P0"], ["P1"], 0.70, 0.70, step="base"),                 # kept -> no step left
                "3": prow(["P0"], ["P1"], 0.70, 0.95, over=True, step="base"),      # overshoot -> B
                "4": prow(["P0"], None, 0.70, None)}, up)                           # no candidate
    R = LP.next_stacks_paired([r], q_choice=lambda key, info, dr: None)
    assert R["paired"] and R["round"] == 2
    assert R["stacks"] == {"1": ["P1"], "2": ["P0"], "3": ["P0"], "4": ["P0"]}
    assert R["candidates"] == {"1": ["P1", "M"], "2": None, "3": ["P2", "B"], "4": None}
    assert R["steps"] == {"1": "M", "2": None, "3": "B", "4": None}
    assert R["tried"] == {"1": ["base"], "2": ["base"], "3": ["base"], "4": []}
    assert R["decisions"]["1"]["word"] == "CANDIDATE ADOPTED" and R["decisions"]["4"] is None
    assert not R["stop"]
    # the adopted candidate's readings drive the next step (its output qubit for F)
    r2 = rnd(2, {"1": prow(["P1"], ["P1", "M"], 0.80, 0.78, step="M", tried=["base"],
                           cand_info={"output_qubit": 9}, inc_info={"output_qubit": 5})}, up)
    R2 = LP.next_stacks_paired([r, r2])
    assert R2["stacks"]["1"] == ["P1"] and R2["candidates"]["1"] == ["P2", "F"]    # M not adopted, F on q5
    assert R2["tried"]["1"] == ["base", "M"]
    r3 = rnd(3, {"1": prow(["P1"], ["P1", "M"], 0.80, 0.85, step="M", tried=["base"],
                           cand_info={"output_qubit": 9})}, up)
    R3 = LP.next_stacks_paired([r3])
    assert R3["stacks"]["1"] == ["P1", "M"] and R3["candidates"]["1"] is None        # q9 has no Dr: no F
    assert R3["stop"] and R3["stop_reason"].startswith("no step left")
    R4 = LP.next_stacks_paired([dict(r, round=4)], k_max=4)
    assert R4["stop"] and R4["stop_reason"] == "round limit K = 4"
    R5 = LP.next_stacks_paired([dict(r, model_seconds_so_far=200.0)], next_cost=50.0, budget=240.0)
    assert R5["stop"] and R5["stop_reason"].startswith("budget")
    with pytest.raises(ValueError):
        LP.next_stacks_paired([r], rungs=("X",))


def test_climb_on_kept_option():
    up = {"5": dict(gap=0.10, two_se=0.02, sig=True)}
    r = rnd(1, {"1": prow(["P0"], ["P1"], 0.70, 0.705, step="base"),                  # kept (within error)
                "2": prow(["P0"], ["P1"], 0.70, 0.60, step="base"),                   # not adopted
                "3": prow(["P0"], ["P1"], 0.70, 0.95, over=True, step="base")}, up)   # overshoot
    off = LP.next_stacks_paired([r])
    assert off["candidates"] == {"1": None, "2": None, "3": ["P2", "B"]} and "climb_on_kept" not in off
    on = LP.next_stacks_paired([r], climb_on_kept=True)
    assert on["stacks"] == {"1": ["P0"], "2": ["P0"], "3": ["P0"]}
    assert on["candidates"] == {"1": ["P1", "M"], "2": None, "3": ["P2", "B"]}
    assert on["steps"]["1"] == "M" and on["tried"]["1"] == ["base"] and on["climb_on_kept"]
    assert "climbing on the kept P1" in on["why"]["1"]


def test_paired_rule1_and_the_ledger():
    fam = {"P1": "flip", "P2": "flip", "P0": "P0"}.get
    R = dict(round=1, device="ibm_fake", stacks={"1": ["P1"], "2": ["P2"], "3": ["P0"]}, why={})
    last = dict(night="n1", device="ibm_fake", calibration_time="t1", last_stacks={"1": ["P1", "M"], "2": ["P0"]})
    st, dec = LP.ledger_start([last], "ibm_fake", "t1", R["stacks"], fam)
    P = LP.paired_rule1(R, st, dec)
    assert P["stacks"] == {"1": ["P0"], "2": ["P0"], "3": ["P0"]}
    assert P["candidates"] == {"1": ["P1", "M"], "2": ["P2"], "3": None}
    assert P["steps"] == {"1": "ledger", "2": "base", "3": None} and P["pick"] == {"1": "P1", "2": "P2", "3": "P0"}
    assert P["diagnostics_stacks"] == R["stacks"] and not P.get("stop")
    # the ledger's candidate not adopted: the diagnostics' base next
    assert LP.paired_next_candidate(["P0"], ["ledger"], {}, {}, pick="P1") == (["P1"], "base")
    st, dec = LP.ledger_start([last], "ibm_fake", "t2", R["stacks"], fam)
    P = LP.paired_rule1(R, st, dec)
    assert P["candidates"]["1"] == ["P1"] and P["steps"]["1"] == "base"
    P0 = LP.paired_rule1(dict(R, stacks={"1": ["P0"]}), {"1": ["P0"]}, dict(per_key={}))
    assert P0["stop"] and P0["candidates"] == {"1": None}


# ------------------------------------------------------------------------------------------------ the runner

TRUE_P = {"P0": 0.70, "P1": 0.80, "P1+M": 0.85, "P2": 0.78, "P2+F": 0.79, "P1+M+F": 0.84, "P2+M+F": 0.84,
          "P2+B": 0.83}


class FakePaired:
    """A fake paired experiment: the share of a stack is TRUE_P's, the candidate's moved by the job's count."""
    name = "fake-paired"
    order = ["ibm_fake"]
    k_max, budget_s = 6, 400.0

    def __init__(self, seconds=20.0, pending_first=False):
        self.seconds, self.pending_first = seconds, pending_first
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
        assert R["paired"]
        self._job(folder, dict(stacks=R["stacks"], candidates=R["candidates"], round=R["round"]))
        json.dump(R, open(os.path.join(folder, "rule.json"), "w"))
        return True

    def check(self, folder):
        return True

    def score(self, folder):
        R = json.load(open(os.path.join(folder, "rule.json")))
        n = json.load(open(os.path.join(folder, "job", "raw_counts.json")))["rows"][0]["counts"]["n"]
        rows = {}
        for k, inc in R["stacks"].items():
            c = R["candidates"][k]
            ip = TRUE_P["+".join(inc)]
            cp = TRUE_P.get("+".join(c), 0.5) + (n - 100) / 1e4 if c else None
            rows[k] = prow(inc, c, ip, cp, step=R["steps"][k], tried=R["tried"][k], pick=R["pick"][k])
        out = dict(round=R["round"], device=R["device"], paired=True, stacks=R["stacks"], rows=rows,
                   dr={"5": dict(gap=0.1, two_se=0.02, sig=True)},
                   model_seconds_so_far=R.get("model_seconds_so_far", 0.0) + self.seconds)
        json.dump(out, open(os.path.join(folder, "round_scores.json"), "w"), indent=1)
        return out

    def q_choice(self, rf, key, info, dr):
        return None

    def compound(self, rounds):
        return dict(rounds=len(rounds))

    def predictions(self, out):
        return dict(rounds=len([d for d in os.listdir(out) if d.startswith("r") and d[1:].isdigit()]))

    def simulate(self, folder, noise, seed):
        tag = os.path.basename(folder)
        self.simulated.append(tag)
        json.dump(dict(job_id=f"local-{seed}", local=True), open(os.path.join(folder, "job", "submission.json"), "w"))
        if self.pending_first and self.simulated.count(tag) == 1 and tag == "r2":
            return
        json.dump(dict(rows=[dict(pub=0, counts={"n": 100 + seed % 3})]),
                  open(os.path.join(folder, "job", "raw_counts.json"), "w"))

    def dry_seed(self, tag):
        return 1000 + (0 if tag == "dw" else int(tag[1:]))


@pytest.fixture
def devdir(tmp_path):
    d = str(tmp_path / "dev")
    Dv.fetch_fake("FakeFez").save(d)
    return d


def test_paired_night_to_its_stop_and_verify(tmp_path, devdir):
    exp = FakePaired()
    out = str(tmp_path / "n")
    led = str(tmp_path / "ledger.json")
    res = LP.start(out, exp, dry_devdir=devdir, ledger=led, paired=True).go()
    assert json.load(open(os.path.join(out, "run.json")))["paired"] is True
    assert res["reason"].startswith("the rule: no step left")
    R1 = json.load(open(os.path.join(out, "RULE1.json")))
    assert R1["stacks"] == {"1": ["P0"], "2": ["P0"]} and R1["candidates"] == {"1": ["P1"], "2": ["P2"]}
    # L1: P1 adopted, P1+M adopted, F (P2+M+F, 0.84) kept against 0.85; L2: P2 adopted, M (no table: 0.5) not
    # adopted, F not offered (Dr on q5 is not against the P2 frame) -> stop
    last = json.load(open(os.path.join(out, f"RULE{res['rounds'] + 1}.json")))
    assert last["stacks"] == {"1": ["P1", "M"], "2": ["P2"]}
    assert last["tried"]["1"] == ["base", "M", "F"] and last["tried"]["2"] == ["base", "M"]     # Dr not against P2
    assert json.load(open(led))[-1]["last_stacks"] == {"1": ["P1", "M"], "2": ["P2"]}
    assert json.load(open(led))[-1]["paired"] is True
    assert LP.verify(out, FakePaired(), resim=True)
    p = os.path.join(out, "r1", "job", "raw_counts.json")
    d = json.load(open(p))
    d["rows"][0]["counts"]["n"] += 1
    json.dump(d, open(p, "w"))
    assert not LP.verify(out, FakePaired(), resim=True)
    # a RULE changed by hand is seen
    p2 = os.path.join(out, "RULE2.json")
    d["rows"][0]["counts"]["n"] -= 1
    json.dump(d, open(p, "w"))
    R2 = json.load(open(p2))
    R2["candidates"]["1"] = ["P1", "F"]
    json.dump(R2, open(p2, "w"))
    assert not LP.verify(out, FakePaired())
    # the second paired night: same device and calibration time, the ledger's stack is round 1's candidate
    out2 = str(tmp_path / "n2")
    LP.start(out2, FakePaired(), dry_devdir=devdir, ledger=led, paired=True).go()
    R1b = json.load(open(os.path.join(out2, "RULE1.json")))
    assert R1b["candidates"]["1"] == ["P1", "M"] and R1b["steps"]["1"] == "ledger"
    assert LP.verify(out2, FakePaired())


def test_paired_resume_never_submits_twice(tmp_path, devdir):
    out = str(tmp_path / "p")
    exp = FakePaired(pending_first=True)
    res = LP.start(out, exp, dry_devdir=devdir, paired=True).go()
    assert res["resumable"] and exp.simulated == ["dw", "r1", "r2"]
    res = LP.Night(out, exp).go()
    assert res["resumable"] and exp.simulated == ["dw", "r1", "r2"]
    json.dump(dict(rows=[dict(pub=0, counts={"n": 100 + 1002 % 3})]),
              open(os.path.join(out, "r2", "job", "raw_counts.json"), "w"))
    res = LP.Night(out, exp).go()
    assert res["reason"].startswith("the rule: no step left") and exp.simulated.count("r2") == 1
    assert LP.verify(out, FakePaired())


def test_paired_off_by_default(tmp_path, devdir):
    out = str(tmp_path / "d")
    LP.start(out, FakePaired(), dry_devdir=devdir)
    assert "paired" not in json.load(open(os.path.join(out, "run.json")))


# ------------------------------------------------------------------------------------------------ ff_only

def test_build_ff_only(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    its = [("a", *FF.rus_chain(1, 1.1, m=2)), ("b", *FF.rus_chain(1, 1.1, m=3))]
    FF.build(dev, its, str(tmp_path / "x"), shots=100)
    FF.build(dev, its, str(tmp_path / "y"), shots=100, ff_only=())
    for f in ("job/circuits.qpy", "build.json"):
        assert open(tmp_path / "x" / f, "rb").read() == open(tmp_path / "y" / f, "rb").read()
    assert sorted(os.listdir(tmp_path / "x" / "runs")) == sorted(os.listdir(tmp_path / "y" / "runs"))
    rep = FF.build(dev, its, str(tmp_path / "z"), shots=100, ff_only=["a"])
    runs = sorted(f for f in os.listdir(tmp_path / "z" / "runs") if f.endswith(".json") and f.startswith("ff_"))
    assert "ff_a.json" in runs and "ff_a_off.json" not in runs and "ff_a_floor.json" not in runs
    assert "ff_b_off.json" in runs and "ff_b_floor.json" in runs
    assert json.load(open(tmp_path / "z" / "runs" / "ff_a.json"))["ff_only"] is True
    base = json.load(open(tmp_path / "x" / "build.json"))["circuits"]
    assert rep["circuits"] == base - 2
    with pytest.raises(FF.Refused, match="ff_only"):
        FF.build(dev, its, str(tmp_path / "w"), shots=100, ff_only=["nope"])
