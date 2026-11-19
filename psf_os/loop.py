"""The compounding loop (psf-os v0.2; workplace Claude, 2026-11-19, home handover 2026-11-19 section 4.1): a night of
rounds in which each round's scores choose the next round's prescription, run end to end on the device or locally,
resumable, verifiable from its records. Moved here from the lab notebook's gpu/os2_run.py (the device path, declared
for 2026-11-18), gpu/os3_night.py (the rungs +M and +F) and gpu/os3_run.py (the ledger); those files are unchanged.

What psf-os holds (no circuit of any one test):
  * the rule (`next_stacks`): per key (e.g. a chain length L), first match wins, every threshold "the quantity >= 2 SE
    of itself":
      R1 overshoot: OVERSHOOT and no B                                  -> P2 + B (layers F, M dropped)
      R2 frames:    B, no Q, the nf frame keeps less than rd, and the output qubit's readout gap Dr is significant
                    -> + Q<offset> (the offset the experiment's q_choice names; none -> keep)
      then the rungs in the order given (default ("M", "F"): +M first, then +F; home handover 2026-11-19):
      RM one more try: PRESCRIPTION HELPS, no B, no M                    -> + M
      RF frame by readout direction: no B, no F, base P1 or P2, Dr on the stack's output qubit against the base's read
                    frame                                               -> switch the base (P1 <-> P2) + F
      R3 stop-loss: NOT SHOWN twice with the same stack                  -> remove the last layer (F: switch back)
      R4 keep
    Stops: a fixed point (no key changes, none overshoots, none NOT SHOWN for the first time), the round limit, the
    budget (model seconds), and in the runner a check FAIL, a refused build, a job that ends in error.
  * the ledger (`ledger_start`): the last night's final stacks are used for round 1 only on the same device with the
    same calibration time (the device record's time stamp, never an error number), and per key only where tonight's
    diagnostics agree with the ledger's family of base; where they disagree the diagnostics win. Every decision (used /
    not used, and why) is written into RULE1.json.
  * the runner (`Night`): fetch, diagnostics, round 1 (rule of the experiment + ledger), then per round: the budget
    first (a build in a temporary folder gives the round's model seconds), build, check, submit, queue after, collect,
    score, rule; finish: compound, predictions, job timing, zip. Every step is skipped when its output exists, so
    `resume` continues where the night stopped; a job folder holding a submission.json is never submitted again.
    `verify` rebuilds, re-scores and re-rules from the records: SAME / DIFFERENT per item.

The experiment (the circuits, their scores, the first rule, the diagnostics) is given from outside as an object with
the methods listed in `Experiment` (the lab notebook's gpu/os_loop.py is the one for the RUS chain). IBM is touched only
by `python -m psf_os submit` / `collect`, the experiment's fetch and the queue read after each submission; no key is
read (the saved account); dry runs never touch IBM.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

RUNGS = ("M", "F")
FRAME_SWITCH = {"P1": "P2", "P2": "P1"}
COLLECT_WAIT_MIN = 60
VERDICTS = ("OVERSHOOT", "PRESCRIPTION HELPS", "NOT SHOWN")


class Experiment:
    """What a night needs from the experiment (duck-typed; this class only documents it).

    name                                    str
    k_max, budget_s                         round limit and model-second budget (defaults of a night)
    order                                   the devices in the declared order (for the queue read)
    fetch(base) -> bool                     device records and queue into base (base/queue.json, base/<dev>/dev)
    calibration_time(devdir) -> str | None  the record's calibration time stamp
    diagnostics() -> [(tag, build, check, score, scores_file)]
        build(devdir, folder, dry); check(folder) -> bool; score(folder) (writes folder/scores_file)
    first_rule(out, diag_folders, device) -> dict   round 1 (writes nothing): round, device, stacks {key: [layers]},
                                            why {key: str}, pick {key: base}, model_seconds_so_far
    family(base) -> str                     the ledger's agreement test (e.g. P1 and P2 are one family)
    build(devdir, folder, rule_path, dry) -> bool   the round's job (folder/job, folder/build.json)
    check(folder) -> bool                   the pre-run check
    score(folder) -> dict                   writes folder/round_scores.json (rows {key: {...}}, dr, stacks, round)
    q_choice(round_folder, key, row, dr) -> offset | None   R2's choice
    compound(rounds) -> dict; predictions(out) -> dict      beside / declared predictions, from the records
    simulate(folder, noise, seed)           dry runs only: write job/submission.json and job/raw_counts.json
    dry_seed(tag) -> int
    """


def now():
    t = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    return f"{t.isoformat()} ({t.astimezone(ZoneInfo('Europe/Zurich')).strftime('%Y-%m-%d %H:%M:%S')} CET)"


# ------------------------------------------------------------------------------------------------ the rule

def base_of(st):
    return st[0]


def frame_reads_one_as_one(base):
    """True when the base reads logical 1 as |1> (P0, P1, P3, NF); False for P2 (read flipped)."""
    return base != "P2"


def against_frame(base, d):
    """Dr (gap = P(0|0) - P(1|1), two_se) on the output qubit shows the gap against the base's read frame."""
    if d is None:
        return False
    gap, two = d["gap"], d["two_se"]
    if frame_reads_one_as_one(base):
        return gap > 0 and gap >= two
    return -gap > 0 and -gap >= two


def _significant(x, two_se):
    return x is not None and two_se is not None and x > 0 and x >= two_se


def next_stacks(rounds, rungs=RUNGS, q_choice=None, k_max=4, budget=240.0, next_cost=None):
    """The next round's rule from the rounds scored so far (each a round_scores dict: round, device, rows {key: row},
    dr {qubit: {gap, two_se, sig}}, model_seconds_so_far). A row holds: stack, verdict, overshoot; with B also
    share_gap, share_gap_two_se, output_qubit_nf; without B output_qubit. q_choice(key, row, dr) -> offset | None."""
    for r in rungs:
        if r not in RUNGS:
            raise ValueError(f"rung {r!r} is not one of {RUNGS}")
    cur = rounds[-1]
    prev = rounds[-2] if len(rounds) > 1 else None
    dr = cur.get("dr") or {}
    keys = list(cur["rows"])
    stacks, why = {}, {}
    for key in keys:
        row = cur["rows"][key]
        st = list(row["stack"])
        new, reason = st, "R4 keep"
        if row.get("verdict") == "OVERSHOOT" and "B" not in st:
            new, reason = ["P2", "B"], "R1 overshoot -> P2 + B"
        elif "B" in st and not any(x.startswith("Q") for x in st):
            qn = str(row.get("output_qubit_nf"))
            if (_significant(row.get("share_gap"), row.get("share_gap_two_se")) and qn in dr and dr[qn].get("sig")
                    and q_choice is not None):
                o = q_choice(key, row, dr)
                if o:
                    new, reason = st + [f"Q{o}"], f"R2 frames -> +Q{o}"
        elif st != ["P0"] and "B" not in st:
            for rung in rungs:
                if rung == "M" and "M" not in st and row.get("verdict") == "PRESCRIPTION HELPS":
                    new, reason = st + ["M"], "RM helps -> one more try (m 3) + M"
                    break
                if rung == "F" and "F" not in st and base_of(st) in FRAME_SWITCH:
                    q = str(row.get("output_qubit"))
                    if against_frame(base_of(st), dr.get(q)):
                        nb = FRAME_SWITCH[base_of(st)]
                        new = [nb] + st[1:] + ["F"]
                        reason = (f"RF frame: Dr on q{q} gap {dr[q]['gap']:+.4f} (2 SE {dr[q]['two_se']:.4f}) against "
                                  f"the {base_of(st)} frame -> {nb} + F")
                        break
        if new is st and prev is not None:
            prow = prev["rows"].get(key, {})
            if (row.get("verdict") == "NOT SHOWN" and prow.get("verdict") == "NOT SHOWN"
                    and list(prow.get("stack", [])) == st):
                if st[-1] == "F":
                    new = [FRAME_SWITCH.get(base_of(st), base_of(st))] + st[1:-1]
                else:
                    new = st[:-1] if len(st) > 1 else ["P0"]
                reason = "R3 stop-loss: NOT SHOWN twice with the same stack -> " + "+".join(new)
        stacks[key], why[key] = new, reason
    k = cur["round"]
    spent = cur.get("model_seconds_so_far", 0.0)
    unchanged = all(stacks[key] == list(cur["rows"][key]["stack"]) for key in keys)
    any_over = any(cur["rows"][key].get("verdict") == "OVERSHOOT" for key in keys)
    pending = any(cur["rows"][key].get("verdict") == "NOT SHOWN" and not (
        prev is not None and prev["rows"].get(key, {}).get("verdict") == "NOT SHOWN") for key in keys)
    stop, sreason = False, None
    if unchanged and not any_over and not pending:
        stop, sreason = True, "fixed point: every L keeps its stack, none overshoots, none NOT SHOWN for the first time"
    elif k >= k_max:
        stop, sreason = True, f"round limit K = {k_max}"
    elif next_cost is not None and spent + next_cost > budget:
        stop, sreason = True, f"budget: {spent:.1f} + {next_cost:.1f} > {budget:.0f} s"
    return dict(round=k + 1, device=cur["device"], rungs=list(rungs), stacks=stacks, why=why, stop=stop,
                stop_reason=sreason, model_seconds_so_far=spent)


# ------------------------------------------------------------------------------------------------ the ledger

def ledger_load(path):
    return json.load(open(path)) if path and os.path.exists(path) else []


def ledger_start(ledger, device, cal_time, stacks, family):
    """Round 1 stacks from tonight's stacks (the diagnostics' pick) and the ledger's last night. Returns (stacks,
    decision). Used only on the same device with the same calibration time, and per key only where the family of
    tonight's base equals the ledger's; otherwise tonight's diagnostics stand."""
    out = {k: list(v) for k, v in stacks.items()}
    dec = dict(used=False, per_key={}, last_night=None)
    if not ledger:
        dec["why"] = "no ledger"
        return out, dec
    last = ledger[-1]
    dec["last_night"] = dict(night=last.get("night"), device=last.get("device"),
                             calibration_time=last.get("calibration_time"))
    if last.get("device") != device:
        dec["why"] = f"other device ({last.get('device')} -> {device}): start from tonight's diagnostics"
        return out, dec
    if not cal_time or last.get("calibration_time") != cal_time:
        dec["why"] = (f"other calibration ({last.get('calibration_time')} -> {cal_time}): start from tonight's "
                      "diagnostics")
        return out, dec
    for k, st in stacks.items():
        prev = (last.get("last_stacks") or {}).get(k)
        if prev is None:
            dec["per_key"][k] = "no stack for this key in the ledger"
            continue
        if family(prev[0]) != family(st[0]):
            dec["per_key"][k] = (f"diagnostics disagree (tonight {'+'.join(st)}, ledger {'+'.join(prev)}): the "
                                 "diagnostics win, ledger not used")
            continue
        out[k] = list(prev)
        dec["per_key"][k] = f"ledger used: start from {'+'.join(prev)} (same device and calibration time; family agrees)"
        dec["used"] = True
    dec["why"] = "same device and calibration time"
    return out, dec


def ledger_entry(night, device, cal_time, rungs, rounds, decision):
    """The night's line for the ledger, from its scored rounds."""
    e = dict(night=night, device=device, calibration_time=cal_time, rungs=list(rungs), rounds_run=len(rounds),
             ledger_used=decision.get("used"), ledger_why=decision.get("why"))
    if rounds:
        last = rounds[-1]
        e.update(last_stacks={k: last["rows"][k]["stack"] for k in last["rows"]},
                 last_g={k: last["rows"][k].get("g") for k in last["rows"]},
                 last_verdict={k: last["rows"][k].get("verdict") for k in last["rows"]})
    return e


# ------------------------------------------------------------------------------------------------ the runner

def _quiet(fn, log_file, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    with open(log_file, "a") as f:
        f.write(buf.getvalue())
    return r


def budget_text(spent, est, budget, k):
    return (f"budget: the diagnostics and rounds built take {spent:.1f} s by the model, round {k} would add {est:.1f} s "
            f"({spent + est:.1f} s > {budget:.0f} s); round {k} not built, nothing more sent")


class Night:
    """One night in OUT. cfg (OUT/run.json): dry, dry_devdir, noise, k_max, budget, rungs, ledger, zip, device."""

    def __init__(self, out, exp):
        self.out = os.path.abspath(out)
        self.exp = exp
        self.cfg_file = os.path.join(self.out, "run.json")

    # ---------------------------------------------------------------------------------------------- bookkeeping
    def log(self, msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(os.path.join(self.out, "run_log.txt"), "a") as f:
            f.write(line + "\n")

    def cfg(self):
        return json.load(open(self.cfg_file))

    def set_cfg(self, **kw):
        c = self.cfg()
        c.update(kw)
        json.dump(c, open(self.cfg_file, "w"), indent=1)

    def stop(self, reason, resumable=False, **extra):
        d = dict(reason=reason, resumable=resumable, written=now(), **extra)
        json.dump(d, open(os.path.join(self.out, "stop.json"), "w"), indent=1)
        self.log(f"STOP: {reason}" + (" (resumable)" if resumable else ""))
        return d

    def device_dir(self):
        return os.path.join(self.out, "fetch_in", self.cfg()["device"], "dev")

    # ---------------------------------------------------------------------------------------------- IBM or local
    def submit(self, folder):
        job = os.path.join(folder, "job")
        tag = os.path.basename(folder)
        if os.path.exists(os.path.join(job, "submission.json")):
            self.log(f"{tag}: already submitted (submission.json), not sent again")
            return True
        if self.cfg()["dry"]:
            seed = self.exp.dry_seed(tag)
            self.exp.simulate(folder, self.cfg().get("noise"), seed)
            self.log(f"{tag}: run locally ({'example noise' if self.cfg().get('noise') else 'noiseless'}, seed {seed})")
            return True
        r = subprocess.run([sys.executable, "-m", "psf_os", "submit", "--job", job], capture_output=True, text=True)
        self.log(f"{tag}: submit -> {r.stdout.strip()} {r.stderr.strip()}")
        if not os.path.exists(os.path.join(job, "submission.json")):
            return False
        self.queue_after(tag)
        return True

    def queue_after(self, tag):
        """Pending jobs per device right after a submission (the saved account; no key, no quality number)."""
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            s = QiskitRuntimeService()
            q = {n: s.backend(n).status().pending_jobs for n in self.exp.order}
            d = dict(read=dt.datetime.now(dt.timezone.utc).isoformat(), pending=q, after=tag)
        except Exception as exc:  # noqa: BLE001 - a failed read is recorded, never fatal
            d = dict(read=dt.datetime.now(dt.timezone.utc).isoformat(), error=type(exc).__name__, after=tag)
        json.dump(d, open(os.path.join(self.out, tag, "queue_after.json"), "w"), indent=1)
        json.dump(d, open(os.path.join(self.out, f"queue_after_{tag}.json"), "w"), indent=1)

    def collect(self, folder):
        job = os.path.join(folder, "job")
        if os.path.exists(os.path.join(job, "raw_counts.json")):
            return "DONE"
        if self.cfg()["dry"]:
            return "WAIT"
        r = subprocess.run([sys.executable, "-m", "psf_os", "collect", "--job", job, "--wait-min",
                            str(COLLECT_WAIT_MIN)], capture_output=True, text=True)
        self.log(f"{os.path.basename(folder)}: collect -> {r.stdout.strip()} {r.stderr.strip()}")
        if os.path.exists(os.path.join(job, "raw_counts.json")):
            return "DONE"
        # 2026-11-19: the job's state as a value (psf_os.ibm.collect writes job/collect_status.json), not a word
        # looked for in printed text; a state not known is not taken for a pending job
        cs = _read(os.path.join(job, "collect_status.json"))
        if cs is None:
            return "UNKNOWN (no collect_status.json: " + (r.stderr.strip() or r.stdout.strip())[-200:] + ")"
        state = cs.get("state")
        if state == "WAIT":
            return "WAIT"
        if state == "DONE":
            return "UNKNOWN (DONE but no raw_counts.json)"
        return f"{state} ({cs.get('status')})"

    # ---------------------------------------------------------------------------------------------- steps
    def fetch(self):
        base = os.path.join(self.out, "fetch_in")
        c = self.cfg()
        if not os.path.exists(os.path.join(base, "queue.json")):
            if c["dry"]:
                from psf_os.device import Device
                dev = Device.load(c["dry_devdir"])
                os.makedirs(os.path.join(base, dev.name))
                shutil.copytree(c["dry_devdir"], os.path.join(base, dev.name, "dev"))
                json.dump(dict(dry=True, pending=None, chosen=[dev.name], read=None),
                          open(os.path.join(base, "queue.json"), "w"), indent=1)
            elif not self.exp.fetch(base) or not os.path.exists(os.path.join(base, "queue.json")):
                return False
        q = json.load(open(os.path.join(base, "queue.json")))
        if not c.get("device"):
            self.set_cfg(device=q["chosen"][0])
            self.log(f"device {q['chosen'][0]} (queue {q.get('pending')})")
        return True

    def spent(self):
        tot = 0.0
        for d in sorted(os.listdir(self.out)):
            p = os.path.join(self.out, d)
            if os.path.isdir(os.path.join(p, "job")) and os.path.exists(os.path.join(p, "build.json")):
                tot += json.load(open(os.path.join(p, "build.json")))["qpu_seconds_model"]
        return tot

    def budget_ok(self, folder):
        b = self.cfg()["budget"]
        tot = self.spent()
        if tot > b + 1e-9:
            os.rename(os.path.join(folder, "job"), os.path.join(folder, "job_NOT_SENT"))
            self.stop(f"budget: the night would take {tot:.1f} s by the model with {os.path.basename(folder)}, over "
                      f"{b:.0f} s; {os.path.basename(folder)} not sent (job_NOT_SENT)", budget_s=b, model_s=tot)
            return False
        return True

    def run_job(self, folder, check, score):
        """check -> budget -> submit -> collect -> score, each skipped when its output exists. True, or a stop dict."""
        name = os.path.basename(folder)
        if not os.path.exists(os.path.join(folder, "job", "submission.json")):
            if not os.path.exists(os.path.join(folder, "check_ok")):
                ok = _quiet(check, os.path.join(folder, "check.txt"), folder)
                if ok is False:
                    os.rename(os.path.join(folder, "job"), os.path.join(folder, "job_NOT_SENT"))
                    return self.stop(f"{name}: check FAIL (check.txt); not sent")
                open(os.path.join(folder, "check_ok"), "w").write(now() + "\n")
                self.log(f"{name}: check PASS")
            if not self.budget_ok(folder):
                return json.load(open(os.path.join(self.out, "stop.json")))
            if not self.submit(folder):
                return self.stop(f"{name}: submission refused or failed (see run_log.txt)")
        st = self.collect(folder)
        if st == "WAIT":
            if self.cfg()["dry"]:
                return self.stop(f"{name}: the dry run has no result for this job (simulation output missing); run "
                                 "resume after it is written", resumable=True)
            return self.stop(f"{name}: not finished after {COLLECT_WAIT_MIN} min; run resume", resumable=True)
        if st != "DONE":
            return self.stop(f"{name}: the job ended {st}")
        _quiet(score, os.path.join(folder, "score.txt"), folder)
        self.log(f"{name}: scored")
        return True

    def diagnostics(self):
        for tag, build, check, score, scores_file in self.exp.diagnostics():
            folder = os.path.join(self.out, tag)
            if not os.path.exists(folder):
                try:
                    _quiet(build, os.path.join(self.out, f"build_{tag}.txt"), self.device_dir(), folder,
                           self.cfg()["dry"])
                except (Exception, SystemExit) as exc:  # noqa: BLE001
                    self.stop(f"{tag}: build refused ({type(exc).__name__}: {exc})")
                    return False
                self.log(f"{tag}: built ({json.load(open(os.path.join(folder, 'build.json')))['qpu_seconds_model']} s)")
            if not os.path.exists(os.path.join(folder, scores_file)):
                if self.run_job(folder, check, score) is not True:
                    return False
        return True

    def round1(self):
        p = os.path.join(self.out, "RULE1.json")
        if os.path.exists(p):
            return
        c = self.cfg()
        diag = [os.path.join(self.out, t) for t, *_ in self.exp.diagnostics()]
        R = _quiet(self.exp.first_rule, os.path.join(self.out, "round1.txt"), self.out, diag, c["device"])
        cal = self.exp.calibration_time(self.device_dir())
        led = ledger_load(c.get("ledger"))
        stacks, dec = ledger_start(led, c["device"], cal, R["stacks"], self.exp.family)
        for k in stacks:
            if stacks[k] != R["stacks"][k]:
                R["why"][k] = dec["per_key"][k]
        # the diagnostics' stacks and the ledger line read are kept, so that verify re-derives round 1 from the records
        R.update(diagnostics_stacks=R["stacks"], stacks=stacks, ledger=dec, ledger_last=led[-1] if led else None,
                 calibration_time=cal, rungs=c["rungs"])
        json.dump(R, open(p, "w"), indent=1)
        self.log("RULE1.json: " + json.dumps(stacks) + f"; ledger: {dec['why']}")

    def round_estimate(self, rule_path):
        tmp = tempfile.mkdtemp(prefix="loop_est_")
        try:
            d = os.path.join(tmp, "r")
            ok = _quiet(self.exp.build, os.path.join(tmp, "x.txt"), self.device_dir(), d, rule_path, self.cfg()["dry"])
            if ok is False or not os.path.exists(os.path.join(d, "build.json")):
                raise SystemExit("the round's build was refused: " + open(os.path.join(tmp, "x.txt")).read()[-300:])
            return float(json.load(open(os.path.join(d, "build.json")))["qpu_seconds_model"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def next_rule(self, k):
        c = self.cfg()
        rounds = [json.load(open(os.path.join(self.out, f"r{j}", "round_scores.json"))) for j in range(1, k + 1)]
        rf = os.path.join(self.out, f"r{k}")
        return next_stacks(rounds, c["rungs"], q_choice=lambda key, row, dr: self.exp.q_choice(rf, key, row, dr),
                           k_max=c["k_max"], budget=c["budget"])

    def rounds(self):
        c = self.cfg()
        self.round1()
        k = 1
        while True:
            rule_k = os.path.join(self.out, f"RULE{k}.json")
            R = json.load(open(rule_k))
            if R.get("stop"):
                return self.stop(f"the rule: {R['stop_reason']} (after round {k - 1})", rounds=k - 1)
            folder = os.path.join(self.out, f"r{k}")
            if not os.path.exists(folder):
                try:
                    est = self.round_estimate(rule_k)
                except (Exception, SystemExit) as exc:  # noqa: BLE001
                    return self.stop(f"r{k}: build refused ({type(exc).__name__}: {exc})", rounds=k - 1)
                spent = self.spent()
                json.dump(dict(round=k, spent_s=spent, estimate_s=est, budget_s=c["budget"]),
                          open(os.path.join(self.out, f"budget_r{k}.json"), "w"), indent=1)
                if spent + est > c["budget"] + 1e-9:
                    return self.stop(budget_text(spent, est, c["budget"], k), rounds=k - 1, spent_s=spent,
                                     estimate_s=est, budget_s=c["budget"])
                try:
                    ok = _quiet(self.exp.build, os.path.join(self.out, f"build_r{k}.txt"), self.device_dir(), folder,
                                rule_k, c["dry"])
                except (Exception, SystemExit) as exc:  # noqa: BLE001
                    return self.stop(f"r{k}: build refused ({type(exc).__name__}: {exc})", rounds=k - 1)
                if ok is False or not os.path.isdir(os.path.join(folder, "job")):
                    return self.stop(f"r{k}: build refused (build_r{k}.txt)", rounds=k - 1)
                self.log(f"r{k}: built {json.dumps(R['stacks'])} "
                         f"({json.load(open(os.path.join(folder, 'build.json')))['qpu_seconds_model']} s)")
            if not os.path.isdir(os.path.join(folder, "job")):
                return json.load(open(os.path.join(self.out, "stop.json")))
            if not os.path.exists(os.path.join(folder, "round_scores.json")):
                r = self.run_job(folder, self.exp.check, self.exp.score)
                if r is not True:
                    return r
            nxt = os.path.join(self.out, f"RULE{k + 1}.json")
            if not os.path.exists(nxt):
                Rn = self.next_rule(k)
                json.dump(Rn, open(nxt, "w"), indent=1)
                self.log(f"RULE{k + 1}.json: {json.dumps(Rn['stacks'])} " + json.dumps(Rn["why"]))
            k += 1

    def scored_rounds(self):
        ds = sorted([d for d in os.listdir(self.out) if d.startswith("r") and d[1:].isdigit() and
                     os.path.exists(os.path.join(self.out, d, "round_scores.json"))], key=lambda d: int(d[1:]))
        return ds, [json.load(open(os.path.join(self.out, d, "round_scores.json"))) for d in ds]

    def finish(self):
        c = self.cfg()
        names, rounds = self.scored_rounds()
        if rounds:
            json.dump(self.exp.compound(rounds), open(os.path.join(self.out, "compound.json"), "w"), indent=1)
            pr = self.exp.predictions(self.out)
            json.dump(pr, open(os.path.join(self.out, "predictions.json"), "w"), indent=1)
            self.log("predictions: " + json.dumps(pr))
        rows = {}
        for d in [t for t, *_ in self.exp.diagnostics()] + names:
            p = os.path.join(self.out, d)
            if os.path.isdir(p):
                sub = _read(os.path.join(p, "job", "submission.json")) or {}
                raw = _read(os.path.join(p, "job", "raw_counts.json")) or {}
                bj = _read(os.path.join(p, "build.json")) or {}
                rows[d] = dict(job_id=sub.get("job_id"), submitted=sub.get("submitted"), fetched=raw.get("fetched"),
                               status=raw.get("status"), qpu_seconds_model=bj.get("qpu_seconds_model"),
                               metrics=sub.get("metrics") or raw.get("metrics"),
                               queue_after=_read(os.path.join(p, "queue_after.json")))
        rows["queue_before"] = None if c["dry"] else _read(os.path.join(self.out, "fetch_in", "queue.json"))
        rows["model_seconds_total"] = self.spent()
        json.dump(rows, open(os.path.join(self.out, "job_timing.json"), "w"), indent=1)
        led = c.get("ledger")
        if led and not os.path.exists(os.path.join(self.out, "ledger_written")):
            R1 = json.load(open(os.path.join(self.out, "RULE1.json"))) if os.path.exists(
                os.path.join(self.out, "RULE1.json")) else {}
            e = ledger_entry(os.path.basename(self.out), c["device"], R1.get("calibration_time"), c["rungs"], rounds,
                             R1.get("ledger", {}))
            json.dump(ledger_load(led) + [e], open(led, "w"), indent=1, default=str)
            open(os.path.join(self.out, "ledger_written"), "w").write(now() + "\n")
            self.log(f"ledger: night appended to {led}")
        z = c.get("zip")
        if z:
            import zipfile
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
                for dp, dn, fn in os.walk(self.out):
                    for n in sorted(fn):
                        p = os.path.join(dp, n)
                        f.write(p, os.path.join(os.path.basename(self.out), os.path.relpath(p, self.out)))
            self.log(f"zipped to {z}")

    def go(self):
        st = os.path.join(self.out, "stop.json")
        if os.path.exists(st):
            s = json.load(open(st))
            if not s.get("resumable"):
                self.log(f"the night has stopped: {s['reason']} (not resumable)")
                return s
            base = st.replace(".json", f"_{dt.datetime.now().strftime('%H%M%S')}")
            dest, i = base + ".json", 1
            while os.path.exists(dest):                       # an earlier stop of the same second is kept
                dest, i = f"{base}_{i}.json", i + 1
            os.rename(st, dest)
        if not self.fetch():
            return self.stop("fetch failed")
        if not self.diagnostics():
            res = json.load(open(st))
        else:
            res = self.rounds()
        if not res.get("resumable"):
            self.finish()
        return res


def _read(p):
    return json.load(open(p)) if os.path.exists(p) else None


def start(out, exp, dry_devdir=None, noise=None, k_max=None, budget=None, rungs=RUNGS, ledger=None, zip_path=None,
          extra=None):
    """A new night in OUT (refused if OUT exists). Device runs need zip_path; noise / k_max / budget are for dry runs."""
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: use resume")
    if not dry_devdir and (noise or k_max or budget):
        raise SystemExit("noise, k_max and budget are for dry runs only")
    if not dry_devdir and not zip_path:
        raise SystemExit("a zip file is required for a device run")
    os.makedirs(out)
    from psf_os import job as J
    cfg = dict(build_env=J.build_env(), dry=bool(dry_devdir), dry_devdir=os.path.abspath(dry_devdir) if dry_devdir else None, noise=noise,
               k_max=k_max or exp.k_max, budget=budget or exp.budget_s, rungs=list(rungs),
               ledger=os.path.abspath(ledger) if ledger else None, zip=os.path.abspath(zip_path) if zip_path else None,
               started=now(), device=None, experiment=exp.name, **(extra or {}))
    json.dump(cfg, open(os.path.join(out, "run.json"), "w"), indent=1)
    return Night(out, exp)


# ------------------------------------------------------------------------------------------------ verify

def first_difference(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}/{k} missing on one side"
            d = first_difference(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, float) or isinstance(b, float):
        try:
            return None if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)) else f"{path}: {a} != {b}"
        except TypeError:
            pass
    return None if a == b else f"{path}: {a!r} != {b!r}"


VERSION_KEYS = ("written", "env", "circuits_sha256", "qpy_sha256")


def _split_digests(job_dir):
    """(digest of the test circuits, digest of the sentinel's), by the manifest's roles."""
    from qiskit import qpy
    from psf_os.gatehash import circuits_digest
    man = json.load(open(os.path.join(job_dir, "manifest.json")))
    roles = [cell["role"] for cell in man["cells"] for _ in range(cell["points"])]
    cs = qpy.load(open(os.path.join(job_dir, "circuits.qpy"), "rb"))
    return (circuits_digest([c for c, r in zip(cs, roles) if r != "sentinel"]),
            circuits_digest([c for c, r in zip(cs, roles) if r == "sentinel"]))


def same_job(rec_job, new_job):
    """(difference or None, how). circuits.qpy byte for byte; when the bytes differ (another qiskit / numpy / Python
    writes other bytes for the same circuits: the QPY format, an angle's last digits) the gates are compared instead
    (psf_os.gatehash.circuits_digest: conditional blocks opened, RZ in (-pi, pi], parameters to 1e-10), the test
    circuits and the ring sentinel apart: the sentinel's one-qubit synthesis follows the qiskit-aer / numpy versions
    (2026-11-18), so a sentinel that differs only so is reported beside and is not a difference of the job. The
    manifest must agree but for its time, its build environment and, when the bytes differ, the byte hashes."""
    a = open(os.path.join(rec_job, "circuits.qpy"), "rb").read()
    b = open(os.path.join(new_job, "circuits.qpy"), "rb").read()
    how = "bytes"
    if a != b:
        ta, sa = _split_digests(rec_job)
        tb, sb = _split_digests(new_job)
        if ta != tb:
            return "test circuits differ (gates)", "gates"
        how = "test circuits' gates (the bytes differ: other versions)" + (
            "" if sa == sb else "; the ring sentinel differs (beside, not judged)")
    ma, mb = (json.load(open(os.path.join(x, "manifest.json"))) for x in (rec_job, new_job))
    drop = VERSION_KEYS if how != "bytes" else ("written", "env")
    for m in (ma, mb):
        for k in drop:
            m.pop(k, None)
    return first_difference(ma, mb), how


def verify(out, exp, resim=False):
    """Rebuild, re-score and re-rule every step from the records: a line SAME / DIFFERENT per item."""
    out = os.path.abspath(out)
    cfg = json.load(open(os.path.join(out, "run.json")))
    devdir = os.path.join(out, "fetch_in", cfg["device"], "dev")
    lines, bad = [], []

    def note(what, diff):
        how = None
        if isinstance(diff, tuple):
            diff, how = diff
        tail = f": {diff}" if diff is not None else (f" ({how})" if how and how != "bytes" else "")
        lines.append(f"{'SAME     ' if diff is None else 'DIFFERENT'} {what}" + tail)
        if diff is not None:
            bad.append(what)

    def rescore(rec, fn, scores, tmp, extra=()):
        d = os.path.join(tmp, os.path.basename(rec) + "_rescore")
        shutil.copytree(os.path.join(rec, "job"), os.path.join(d, "job"))
        shutil.copytree(os.path.join(rec, "runs"), os.path.join(d, "runs"))
        for f in ("build.json",) + tuple(extra):
            if os.path.exists(os.path.join(rec, f)):
                shutil.copy2(os.path.join(rec, f), os.path.join(d, f))
        _quiet(fn, os.path.join(tmp, "x.txt"), d)
        return first_difference(json.load(open(os.path.join(rec, scores))), json.load(open(os.path.join(d, scores))))

    def job_of(rec):
        for j in ("job", "job_NOT_SENT"):
            if os.path.isdir(os.path.join(rec, j)):
                return os.path.join(rec, j)
        return None

    tmp = tempfile.mkdtemp(prefix="loop_verify_")
    try:
        diag = exp.diagnostics()
        for tag, build, check, score, scores_file in diag:
            rec = os.path.join(out, tag)
            if not os.path.isdir(rec):
                continue
            new = os.path.join(tmp, tag)
            _quiet(build, os.path.join(tmp, "x.txt"), devdir, new, cfg["dry"])
            j = job_of(rec)
            note(f"{tag} job rebuilt", same_job(j, os.path.join(new, "job")) if j else "no job recorded")
            if os.path.exists(os.path.join(rec, scores_file)):
                note(f"{tag} re-scored", rescore(rec, score, scores_file, tmp))
        p1 = os.path.join(out, "RULE1.json")
        if os.path.exists(p1):
            R = _quiet(exp.first_rule, os.path.join(tmp, "x.txt"), out, [os.path.join(out, t) for t, *_ in diag],
                       cfg["device"])
            rec1 = json.load(open(p1))
            note("round 1: the diagnostics' stacks re-ruled", first_difference(rec1.get("diagnostics_stacks"),
                                                                               R["stacks"]))
            led = [rec1["ledger_last"]] if rec1.get("ledger_last") else []
            stacks, dec = ledger_start(led, cfg["device"], rec1.get("calibration_time"), R["stacks"], exp.family)
            note("round 1: the ledger re-applied", first_difference(dict(stacks=rec1["stacks"], ledger=rec1["ledger"]),
                                                                    dict(stacks=stacks, ledger=dec)))
        k = 1
        while os.path.exists(os.path.join(out, f"RULE{k}.json")):
            rec = os.path.join(out, f"r{k}")
            if not os.path.isdir(rec):
                break
            new = os.path.join(tmp, f"r{k}")
            _quiet(exp.build, os.path.join(tmp, "x.txt"), devdir, new, os.path.join(out, f"RULE{k}.json"), cfg["dry"])
            j = job_of(rec)
            if j:
                note(f"r{k} job rebuilt", same_job(j, os.path.join(new, "job")))
            if os.path.exists(os.path.join(rec, "round_scores.json")):
                note(f"r{k} re-scored", rescore(rec, exp.score, "round_scores.json", tmp, extra=("rule.json",)))
                nxt = os.path.join(out, f"RULE{k + 1}.json")
                if os.path.exists(nxt):
                    rounds = [json.load(open(os.path.join(out, f"r{i}", "round_scores.json"))) for i in range(1, k + 1)]
                    Rn = next_stacks(rounds, cfg["rungs"], q_choice=lambda key, row, dr, rf=rec: exp.q_choice(
                        rf, key, row, dr), k_max=cfg["k_max"], budget=cfg["budget"])
                    note(f"RULE{k + 1}.json re-ruled", first_difference(json.load(open(nxt)), Rn))
            k += 1
        if resim and cfg["dry"]:
            for d in sorted(os.listdir(out)):
                rec = os.path.join(out, d)
                if not os.path.exists(os.path.join(rec, "job", "raw_counts.json")):
                    continue
                cp = os.path.join(tmp, d + "_resim")
                shutil.copytree(os.path.join(rec, "job"), os.path.join(cp, "job"),
                                ignore=shutil.ignore_patterns("raw_counts.json", "submission.json"))
                shutil.copytree(os.path.join(rec, "runs"), os.path.join(cp, "runs"))
                try:
                    exp.simulate(cp, cfg.get("noise"), exp.dry_seed(d))
                    a_ = json.load(open(os.path.join(rec, "job", "raw_counts.json")))["rows"]
                    b_ = json.load(open(os.path.join(cp, "job", "raw_counts.json")))["rows"]
                    diff = first_difference(a_, b_)
                except (Exception, SystemExit) as exc:  # noqa: BLE001 - a damaged record is a difference
                    diff = f"re-simulation failed ({type(exc).__name__})"
                note(f"{d} re-simulated (dry, seed {exp.dry_seed(d)})", diff)
        pp = os.path.join(out, "predictions.json")
        if os.path.exists(pp):
            note("predictions re-read", first_difference(json.load(open(pp)), exp.predictions(out)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(lines))
    print("SAME" if not bad else f"DIFFERENT ({len(bad)}: {', '.join(bad)})")
    return not bad
