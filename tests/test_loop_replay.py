"""psf_os.loop's replay (ledger-replay-2026-11-21): a dry night answered from a recorded night's raw counts, cell by
cell (name and gates), stopping with "記録にない" when a circuit is not in the record; the record replayed gives its
RULE files byte for byte; the ledger path started from a replayed record; a broken ledger is not used and is named;
replay is off by default."""
import json
import math
import os

import pytest

from psf_os import device as Dv
from psf_os import loop as LP

from test_loop import Fake, TABLE  # noqa: F401  (the fake experiment of test_loop)


class QpyFake(Fake):
    """Fake with real job folders: one cell per key, a one-qubit circuit whose angle names the key's stack (not the
    round), and a ring-sentinel cell; the builds' dry flag is recorded."""

    def __init__(self, **k):
        super().__init__(**k)
        self.build_flags = []

    @staticmethod
    def _angle(text):
        return (sum(ord(c) * (i + 1) for i, c in enumerate(text)) % 997) / 997 * math.pi

    def _job(self, folder, payload, dry=None):
        from qiskit import QuantumCircuit, qpy
        os.makedirs(os.path.join(folder, "job"))
        os.makedirs(os.path.join(folder, "runs"))
        keys = payload.get("stacks") or {"d": ["diag"]}
        cells, pubs, circs = [], [], []
        for i, (k, st) in enumerate(sorted(keys.items())):
            qc = QuantumCircuit(1, 1)
            qc.rz(self._angle("+".join(st)), 0)
            qc.measure(0, 0)
            cells.append(dict(cell=i, name=f"ff_k{k}", role="user", points=1))
            pubs.append(dict(cell=i, point=0))
            circs.append(qc)
        s = QuantumCircuit(1, 1)
        s.rz(0.123 if dry else 0.456, 0)                                    # the sentinel follows the dry flag
        s.measure(0, 0)
        cells.append(dict(cell=len(cells), name="ksent", role="sentinel", points=1))
        pubs.append(dict(cell=len(cells) - 1, point=0))
        circs.append(s)
        with open(os.path.join(folder, "job", "circuits.qpy"), "wb") as f:
            qpy.dump(circs, f)
        json.dump(dict(written="x", cells=cells, pubs=pubs, fetch=dict(backend="ibm_fake")),
                  open(os.path.join(folder, "job", "manifest.json"), "w"))
        json.dump(dict(qpu_seconds_model=self.seconds), open(os.path.join(folder, "build.json"), "w"))

    def diagnostics(self):
        def build(devdir, folder, dry):
            self.build_flags.append(("dw", dry))
            self._job(folder, dict(diag=True), dry)

        def score(folder):
            json.dump(dict(ok=True), open(os.path.join(folder, "d_scores.json"), "w"))
        return [("dw", build, lambda f: True, score, "d_scores.json")]

    def build(self, devdir, folder, rule_path, dry):
        R = json.load(open(rule_path))
        self.build_flags.append((os.path.basename(folder), dry))
        self._job(folder, dict(stacks=R["stacks"]), dry)
        json.dump(R, open(os.path.join(folder, "rule.json"), "w"))
        return True

    def score(self, folder):
        R = json.load(open(os.path.join(folder, "rule.json")))
        rows_ = json.load(open(os.path.join(folder, "job", "raw_counts.json")))["rows"]
        n = sum(r["counts"]["n"] for r in rows_[:-1])
        from test_loop import row
        rows = {k: row(st, TABLE.get("+".join(st), "NOT SHOWN"), g=n / 1000, output_qubit=5)
                for k, st in R["stacks"].items()}
        out = dict(round=R["round"], device=R["device"], stacks=R["stacks"], rows=rows, dr={},
                   model_seconds_so_far=R.get("model_seconds_so_far", 0.0) + self.seconds)
        json.dump(out, open(os.path.join(folder, "round_scores.json"), "w"), indent=1)
        return out

    def simulate(self, folder, noise, seed):
        tag = os.path.basename(folder)
        self.simulated.append(tag)
        man = json.load(open(os.path.join(folder, "job", "manifest.json")))
        json.dump(dict(job_id=f"local-{seed}", local=True, backend="ibm_fake"),
                  open(os.path.join(folder, "job", "submission.json"), "w"))
        json.dump(dict(rows=[dict(pub=i, counts={"n": 100 + (seed + i) % 7}) for i in range(len(man["pubs"]))]),
                  open(os.path.join(folder, "job", "raw_counts.json"), "w"))


@pytest.fixture
def devdir(tmp_path):
    d = str(tmp_path / "dev")
    Dv.fetch_fake("FakeFez").save(d)
    return d


@pytest.fixture
def record(tmp_path, devdir):
    out = str(tmp_path / "rec")
    led = str(tmp_path / "rec_ledger.json")
    res = LP.start(out, QpyFake(), dry_devdir=devdir, ledger=led).go()
    assert res["reason"].startswith("the rule: fixed point") and res["rounds"] == 3
    return out, led


def _rules(out):
    return {f: open(os.path.join(out, f), "rb").read() for f in sorted(os.listdir(out)) if f.startswith("RULE")}


def test_replay_of_the_record_is_the_record(tmp_path, devdir, record):
    rec, _ = record
    exp = QpyFake()
    out = str(tmp_path / "rp")
    res = LP.start(out, exp, dry_devdir=devdir, replay_from=rec).go()
    assert exp.simulated == []                                           # nothing simulated
    assert all(dry is False for _, dry in exp.build_flags)               # the device night's builds
    assert res["reason"] == json.load(open(os.path.join(rec, "stop.json")))["reason"]
    assert _rules(out) == _rules(rec)                                     # byte for byte
    r = json.load(open(os.path.join(out, "r2", "replay.json")))
    assert r["primary"] == "r2" and r["one_recorded_job"] and r["test_cells_from"] == ["r2"]
    assert json.load(open(os.path.join(out, "r1", "job", "raw_counts.json")))["rows"][:-1] == \
        json.load(open(os.path.join(rec, "r1", "job", "raw_counts.json")))["rows"][:-1]
    assert json.load(open(os.path.join(out, "run.json")))["replay_from"] == os.path.abspath(rec)
    assert LP.verify(out, QpyFake())
    # a changed replayed count is seen by verify (replayed again from the record)
    p = os.path.join(out, "r1", "job", "raw_counts.json")
    d = json.load(open(p))
    d["rows"][0]["counts"]["n"] += 1
    json.dump(d, open(p, "w"))
    assert not LP.verify(out, QpyFake())


def test_replay_with_the_ledger_from_cells_of_two_jobs(tmp_path, devdir, record):
    rec, _ = record
    # the record's final stacks {1: P1+M, 2: P0}; key 2's family disagrees with tonight's P2: the diagnostics win
    line = dict(night="hand", device=Dv.Device.load(devdir).name, calibration_time="cal-1",
                last_stacks={"1": ["P1", "M"], "2": ["P0"]})
    led = str(tmp_path / "led.json")
    json.dump([line], open(led, "w"))
    out = str(tmp_path / "rl")
    exp = QpyFake()
    LP.start(out, exp, dry_devdir=devdir, ledger=led, replay_from=rec).go()
    R1 = json.load(open(os.path.join(out, "RULE1.json")))
    assert R1["ledger"]["used"] and R1["stacks"] == {"1": ["P1", "M"], "2": ["P2"]}
    r = json.load(open(os.path.join(out, "r1", "replay.json")))
    assert r["primary"] == "r2" and r["test_cells_from"] == ["r2"]       # P1+M and P2 side by side in r2
    assert exp.simulated == []


def test_replay_stops_on_a_circuit_not_in_the_record(tmp_path, devdir, record):
    rec, _ = record
    line = dict(night="hand", device=Dv.Device.load(devdir).name, calibration_time="cal-1",
                last_stacks={"1": ["P2", "M", "F"], "2": ["P2"]})       # P2+M+F was never run
    led = str(tmp_path / "led.json")
    json.dump([line], open(led, "w"))
    out = str(tmp_path / "rm")
    exp = QpyFake()
    res = LP.start(out, exp, dry_devdir=devdir, ledger=led, replay_from=rec).go()
    assert "記録にない" in res["reason"] and "ff_k1" in res["reason"] and "ff_k2" not in res["reason"]
    assert res["replay_missing"] and not res["resumable"] and exp.simulated == []
    assert not os.path.exists(os.path.join(out, "r1", "job", "raw_counts.json"))
    assert not os.path.exists(os.path.join(out, "r1", "job", "submission.json"))


def test_start_refuses_replay_outside_dry_runs(tmp_path, devdir, record):
    rec, _ = record
    with pytest.raises(SystemExit, match="dry runs only"):
        LP.start(str(tmp_path / "x"), QpyFake(), zip_path="z.zip", replay_from=rec)
    with pytest.raises(SystemExit, match="no noise"):
        LP.start(str(tmp_path / "y"), QpyFake(), dry_devdir=devdir, noise={"p": 0.1}, replay_from=rec)
    with pytest.raises(SystemExit, match="not a folder"):
        LP.start(str(tmp_path / "z"), QpyFake(), dry_devdir=devdir, replay_from=str(tmp_path / "none"))


def test_replay_off_by_default(tmp_path, devdir):
    exp = QpyFake()
    out = str(tmp_path / "d")
    LP.start(out, exp, dry_devdir=devdir).go()
    assert "replay_from" not in json.load(open(os.path.join(out, "run.json")))
    assert all(dry is True for _, dry in exp.build_flags)                # dry builds as before
    assert exp.simulated == ["dw", "r1", "r2", "r3"]
    assert not any(os.path.exists(os.path.join(out, d, "replay.json")) for d in os.listdir(out))


def test_a_broken_ledger_is_not_used_and_named(tmp_path):
    fam = Fake.family
    tonight = {"1": ["P1"], "2": ["P2"]}
    p = tmp_path / "led.json"
    p.write_text('[{"night": "cut')
    led = LP.ledger_load(str(p))
    st, dec = LP.ledger_start(led, "ibm_fake", "t1", tonight, fam)
    assert st == tonight and not dec["used"] and dec["why"].startswith("ledger unreadable (the file is not JSON")
    p.write_text('{"night": "x"}')
    st, dec = LP.ledger_start(LP.ledger_load(str(p)), "ibm_fake", "t1", tonight, fam)
    assert not dec["used"] and "not a list of nights" in dec["why"]
    st, dec = LP.ledger_start(["x"], "ibm_fake", "t1", tonight, fam)
    assert not dec["used"] and "last line is not a night" in dec["why"]
    st, dec = LP.ledger_start([dict(device="ibm_fake", calibration_time="t1", last_stacks=["P1"])], "ibm_fake", "t1",
                              tonight, fam)
    assert not dec["used"] and "last_stacks is not a table" in dec["why"]
    st, dec = LP.ledger_start([dict(device="ibm_fake", calibration_time="t1", last_stacks={"1": "P1", "2": ["P2"]})],
                              "ibm_fake", "t1", tonight, fam)
    assert st == tonight and dec["used"] and "not a list of layers" in dec["per_key"]["1"]
    # L missing: that key keeps tonight's stack
    st, dec = LP.ledger_start([dict(device="ibm_fake", calibration_time="t1", last_stacks={"2": ["P2", "M"]})],
                              "ibm_fake", "t1", tonight, fam)
    assert st == {"1": ["P1"], "2": ["P2", "M"]} and dec["per_key"]["1"] == "no stack for this key in the ledger"
    # a good ledger reads as before
    assert LP.ledger_load(None) == [] and LP.ledger_load(str(tmp_path / "none.json")) == []


def test_replay_prefers_the_job_of_the_same_name_at_a_tie(tmp_path):
    # a night that ran the same circuits in r1 and r2 (as 2026-11-19's rounds 1 and 2): each answered from its own job
    exp = QpyFake()
    rec = tmp_path / "rec"
    for i, tag in enumerate(("r1", "r2")):
        exp._job(str(rec / tag), dict(stacks={"1": ["P1"]}), False)
        json.dump(dict(rows=[dict(pub=0, counts={"n": 10 + i}), dict(pub=1, counts={"n": 0})]),
                  open(rec / tag / "job" / "raw_counts.json", "w"))
    for tag, n in (("r2", 11), ("r1", 10)):
        f = str(tmp_path / "new" / tag)
        exp._job(f, dict(stacks={"1": ["P1"]}), False)
        r = LP.replay_job(str(rec), f)
        assert r["primary"] == tag
        assert json.load(open(os.path.join(f, "job", "raw_counts.json")))["rows"][0]["counts"]["n"] == n


def test_replay_is_strict_cells_are_never_gathered_from_two_jobs(tmp_path):
    # release-0.3.0 (the owner's decision of 2026-11-21): key 1's stack is only in r2, key 2's only in r1; each cell
    # is in the record, but not in one recorded job: 記録にない, nothing written, the missing cell and the closest job named
    exp = QpyFake()
    rec = tmp_path / "rec"
    for tag, stacks in (("r1", {"1": ["P1"], "2": ["P0"]}), ("r2", {"1": ["P1", "M"], "2": ["P2"]})):
        exp._job(str(rec / tag), dict(stacks=stacks), False)
        json.dump(dict(rows=[dict(pub=i, counts={"n": 10 + i}) for i in range(3)]),
                  open(rec / tag / "job" / "raw_counts.json", "w"))
    f = str(tmp_path / "new" / "r1")
    exp._job(f, dict(stacks={"1": ["P1", "M"], "2": ["P0"]}), False)
    with pytest.raises(LP.NotInRecord) as e:
        LP.replay_job(str(rec), f)
    assert "記録にない" in str(e.value) and "not gathered from two jobs" in str(e.value)
    assert e.value.primary == "r1" and e.value.missing == ["ff_k1"]
    assert not os.path.exists(os.path.join(f, "job", "raw_counts.json"))
    assert not os.path.exists(os.path.join(f, "replay.json"))


def test_a_strict_stop_names_the_missing_cells_in_stop_json(tmp_path, devdir, record):
    rec, _ = record
    line = dict(night="hand", device=Dv.Device.load(devdir).name, calibration_time="cal-1",
                last_stacks={"1": ["P2", "M", "F"], "2": ["P2"]})
    led = str(tmp_path / "led.json")
    json.dump([line], open(led, "w"))
    out = str(tmp_path / "rs")
    res = LP.start(out, QpyFake(), dry_devdir=devdir, ledger=led, replay_from=rec).go()
    st = json.load(open(os.path.join(out, "stop.json")))
    assert st["not_in_record"]["job"] == "r1" and st["not_in_record"]["missing"] == ["ff_k1"]
    assert st["not_in_record"]["closest_recorded_job"] and res["replay_missing"]
