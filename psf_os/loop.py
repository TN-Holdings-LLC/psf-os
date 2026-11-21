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
  * the paired rule (`next_stacks_paired`, loop-paired-2026-11-21; off by default): per key an incumbent and a
    candidate in one job, decided by their difference inside that job (adopted at d >= 2 SE, not adopted at
    d <= -2 SE or when the candidate overshoots, the incumbent kept between); the next candidate is the next untried
    step (B after an overshoot, the diagnostics' base on P0, R2's Q on B, then the rungs); a step not adopted is not
    tried again that night; a key with no step left stops; the night stops when every key has stopped.
  * the ledger (`ledger_start`): the last night's final stacks are used for round 1 only on the same device with the
    same calibration time (the device record's time stamp, never an error number), and per key only where tonight's
    diagnostics agree with the ledger's family of base; where they disagree the diagnostics win. Every decision (used /
    not used, and why) is written into RULE1.json.
  * the runner (`Night`): fetch, diagnostics, round 1 (rule of the experiment + ledger), then per round: the budget
    first (a build in a temporary folder gives the round's model seconds), build, check, submit, queue after, collect,
    score, rule; finish: compound, predictions, job timing, zip. Every step is skipped when its output exists, so
    `resume` continues where the night stopped; a job folder holding a submission.json is never submitted again.
    `verify` rebuilds, re-scores and re-rules from the records: SAME / DIFFERENT per item.
  * replay (`start(..., replay_from=RECORD)`, ledger-replay-2026-11-21; off by default): a dry night whose jobs are
    answered from a recorded night's raw counts instead of a simulation. Each built circuit is looked up in the
    record by its cell name and its gates (psf_os.gatehash); the ring sentinel by its name only (it is never judged
    and its synthesis follows the versions). A circuit the record does not hold stops the night ("記録にない",
    not resumable): there is no silent fall back to a simulation. The builds of a replayed night are the device
    night's (no dry Clifford sentinel). Every job's replay.json names the recorded job of every cell.

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


def rejudge(rnd, between_job_sigma):
    """Receiver prototype (after-2026-11-20; for the loopvar test's DRIFTS only): a copy of one round's scores with the
    verdict of every row that holds g and g_se judged again with a between-job term, SE = sqrt(g_se^2 + sigma^2)
    (sigma: the excess between-job standard deviation of g, e.g. loopvar's s_b, the same for every key unless a dict
    {key: sigma} is given). PRESCRIPTION HELPS when g > 0 and g >= 2 SE, else NOT SHOWN; OVERSHOOT is kept as scored
    (its threshold is on P, which loopvar does not measure between jobs). Each changed row keeps the shot-only values
    as g_se_shot and verdict_shot and records between_job_sigma. sigma None or 0 returns the round unchanged."""
    if not between_job_sigma:
        return rnd
    out = json.loads(json.dumps(rnd))
    for key, row in out["rows"].items():
        sig = between_job_sigma.get(key, 0.0) if isinstance(between_job_sigma, dict) else float(between_job_sigma)
        if "g" not in row or "g_se" not in row or row.get("verdict") not in VERDICTS:
            continue
        se = (row["g_se"] ** 2 + sig ** 2) ** 0.5
        v = "OVERSHOOT" if row.get("overshoot") else (
            "PRESCRIPTION HELPS" if (row["g"] > 0 and row["g"] >= 2 * se) else "NOT SHOWN")
        row.update(g_se_shot=row["g_se"], verdict_shot=row["verdict"], between_job_sigma=sig, g_se=se, verdict=v)
    return out


def next_stacks(rounds, rungs=RUNGS, q_choice=None, k_max=4, budget=240.0, next_cost=None, between_job_sigma=None):
    """The next round's rule from the rounds scored so far (each a round_scores dict: round, device, rows {key: row},
    dr {qubit: {gap, two_se, sig}}, model_seconds_so_far). A row holds: stack, verdict, overshoot; with B also
    share_gap, share_gap_two_se, output_qubit_nf; without B output_qubit. q_choice(key, row, dr) -> offset | None.
    between_job_sigma (receiver prototype, after-2026-11-20; default None = the rule as declared): every round's
    verdicts judged again by rejudge() before the rule reads them; the result then records between_job_sigma."""
    for r in rungs:
        if r not in RUNGS:
            raise ValueError(f"rung {r!r} is not one of {RUNGS}")
    if between_job_sigma:
        rounds = [rejudge(r, between_job_sigma) for r in rounds]
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
    res = dict(round=k + 1, device=cur["device"], rungs=list(rungs), stacks=stacks, why=why, stop=stop,
               stop_reason=sreason, model_seconds_so_far=spent)
    if between_job_sigma:
        res["between_job_sigma"] = between_job_sigma
    return res


# ------------------------------------------------------------------------------------------------ the paired rule
# loop-paired-2026-11-21 (home instruction "輪の改造: 比べる 2 つの積み方を同じジョブに入れる"; the owner's decision of
# 2026-11-20 20:05 JST). The 2026-11-20 loopvar test: copies in one job agree (pooled p 0.625), jobs minutes apart moved
# beyond shot noise at L 2 (p 0.006). The rule above compares round k's g with round k-1's across jobs; the paired rule
# puts the incumbent and the next candidate in one job and decides by their difference inside it. Off by default
# (Night's cfg "paired"); every night recorded before is ruled and verified as before.

PAIRED_WORDS = ("CANDIDATE ADOPTED", "CANDIDATE NOT ADOPTED", "INCUMBENT KEPT", "CANDIDATE OVERSHOOTS")


def paired_decision(row):
    """One key's decision inside one job. row: exact, inc_p, inc_se, cand_p, cand_se, cand_overshoot.
    d = |P_incumbent - exact| - |P_candidate - exact| (positive: the candidate is nearer), SE from both ff arms' shots.
    OVERSHOOT of the candidate (R1's test, P > noiseless + 2 SE): not adopted whatever d. d >= 2 SE: adopted;
    d <= -2 SE: not adopted; between: the incumbent is kept (the stack with fewer layers)."""
    ex = row["exact"]
    d = abs(row["inc_p"] - ex) - abs(row["cand_p"] - ex)
    se = (row["inc_se"] ** 2 + row["cand_se"] ** 2) ** 0.5
    if row.get("cand_overshoot"):
        word, adopted = "CANDIDATE OVERSHOOTS", False
    elif d > 0 and d >= 2 * se:
        word, adopted = "CANDIDATE ADOPTED", True
    elif d < 0 and -d >= 2 * se:
        word, adopted = "CANDIDATE NOT ADOPTED", False
    else:
        word, adopted = "INCUMBENT KEPT", False
    return dict(d=d, se=se, two_se=2 * se, word=word, adopted=adopted)


def paired_next_candidate(inc, tried, info, dr, rungs=RUNGS, q_choice=None, pick=None, overshoot=False):
    """(candidate stack, step) or (None, None) when no step is left. inc: the incumbent; tried: the steps tried this
    night (adopted or not; a step not adopted is never tried again that night); info: the incumbent's own readings
    (output_qubit; with B share_gap, share_gap_two_se, output_qubit_nf), from the job that measured it last.
    Steps, first match wins:
      B     the last candidate overshot, no B yet, B not tried          -> P2 + B (R1)
      base  the incumbent is P0, base not tried, the diagnostics' pick   -> [pick]
      Q     the incumbent has B, no Q, Q not tried, R2's conditions     -> + Q<offset>
      M, F  in the order of rungs, on an incumbent other than P0 without B: M not in it and not tried -> + M;
            F not in it and not tried, base P1 or P2, Dr on its output qubit against its read frame -> switch + F
    A key with no step left stops."""
    tried = set(tried)
    inc = list(inc)
    if overshoot and "B" not in inc and "B" not in tried:
        return ["P2", "B"], "B"
    if inc == ["P0"]:
        if "base" not in tried and pick and pick != "P0":
            return [pick], "base"
        return None, None
    if "B" in inc:
        if any(x.startswith("Q") for x in inc) or "Q" in tried or q_choice is None:
            return None, None
        qn = str(info.get("output_qubit_nf"))
        if _significant(info.get("share_gap"), info.get("share_gap_two_se")) and qn in dr and dr[qn].get("sig"):
            o = q_choice(info)
            if o:
                return inc + [f"Q{o}"], "Q"
        return None, None
    for rung in rungs:
        if rung == "M" and "M" not in inc and "M" not in tried:
            return inc + ["M"], "M"
        if rung == "F" and "F" not in inc and "F" not in tried and base_of(inc) in FRAME_SWITCH:
            if against_frame(base_of(inc), dr.get(str(info.get("output_qubit")))):
                return [FRAME_SWITCH[base_of(inc)]] + inc[1:] + ["F"], "F"
    return None, None


def _paired_stop(R, k, k_max, budget, next_cost):
    if all(c is None for c in R["candidates"].values()):
        return True, "no step left on any key: every key keeps its incumbent"
    if k >= k_max:
        return True, f"round limit K = {k_max}"
    if next_cost is not None and R["model_seconds_so_far"] + next_cost > budget:
        return True, f"budget: {R['model_seconds_so_far']:.1f} + {next_cost:.1f} > {budget:.0f} s"
    return False, None


def paired_rule1(R, stacks, decision):
    """Round 1 of a paired night from the experiment's first rule R (stacks: the diagnostics' pick per key) and the
    stacks after the ledger: every incumbent P0; the candidate is the ledger's stack where the ledger was used for that
    key (step "ledger"), otherwise the diagnostics' stack (step "base"); a key whose candidate would be P0 has none."""
    R = dict(R)
    diag = {k: list(v) for k, v in R["stacks"].items()}
    cands, steps, why = {}, {}, dict(R.get("why", {}))
    for k, st in stacks.items():
        used = str(decision.get("per_key", {}).get(k, "")).startswith("ledger used")
        if list(st) == ["P0"]:
            cands[k], steps[k] = None, None
            why[k] = "the diagnostics pick P0: no candidate"
        else:
            cands[k], steps[k] = list(st), ("ledger" if used else "base")
            why[k] = (decision["per_key"][k] if used else f"candidate {'+'.join(st)} from the diagnostics") + \
                " against the incumbent P0 in one job"
    R.update(paired=True, diagnostics_stacks=diag, ledger_stacks={k: list(v) for k, v in stacks.items()},
             stacks={k: ["P0"] for k in stacks}, candidates=cands, steps=steps, tried={k: [] for k in stacks},
             pick={k: diag[k][0] for k in diag}, why=why)
    R.setdefault("model_seconds_so_far", 0.0)
    if all(c is None for c in cands.values()):
        R.update(stop=True, stop_reason="no step left on any key: every key keeps its incumbent")
    return R


def next_stacks_paired(rounds, rungs=RUNGS, q_choice=None, k_max=4, budget=240.0, next_cost=None,
                       climb_on_kept=False):
    """The next paired rule from the rounds scored so far. Each row of the last round holds: stack (the incumbent),
    candidate, step, tried, pick, exact, inc_p, inc_se, cand_p, cand_se, cand_overshoot, and the readings of both for
    the next step (inc_info, cand_info: output_qubit; with B share_gap, share_gap_two_se, output_qubit_nf).
    q_choice(key, info, dr) -> offset | None (R2's choice). Returns the rule with stacks (incumbents), candidates,
    steps, tried, the decisions of this round and the stop.
    climb_on_kept (option, default False: the rule as the home instruction wrote it): when the decision is
    INCUMBENT KEPT (the candidate within error of the incumbent), the next candidate is built on the kept candidate
    instead of the incumbent (e.g. P0 kept against P1: the next candidate is P1 + M, against P0). A step not adopted
    (d <= -2 SE) or an overshoot ends that path as before. Simulated nights (notes/loop_paired_2026-11-21): with weak
    illustrative noise the base alone may do nothing while base + M does; without the option such a key stops at P0."""
    for r in rungs:
        if r not in RUNGS:
            raise ValueError(f"rung {r!r} is not one of {RUNGS}")
    cur = rounds[-1]
    dr = cur.get("dr") or {}
    stacks, cands, steps, tried, why, decisions, pick = {}, {}, {}, {}, {}, {}, {}
    for key, row in cur["rows"].items():
        inc = list(row["stack"])
        tr = list(row.get("tried", []))
        pick[key] = row.get("pick")
        over = False
        info = row.get("inc_info") or {}
        if row.get("candidate"):
            dec = paired_decision(row)
            decisions[key] = dec
            if row.get("step") and row["step"] not in tr:
                tr.append(row["step"])
            over = bool(row.get("cand_overshoot"))
            if dec["adopted"]:
                inc, info = list(row["candidate"]), row.get("cand_info") or {}
            head = (f"{'+'.join(row['candidate'])} against {'+'.join(row['stack'])}: d {dec['d']:+.4f} (2 SE "
                    f"{dec['two_se']:.4f}) {dec['word']}")
        else:
            decisions[key] = None
            head = f"{'+'.join(inc)} kept (no candidate)"
        qc = (lambda inf, key=key: q_choice(key, inf, dr)) if q_choice is not None else None
        if row.get("candidate"):
            nc, step = paired_next_candidate(inc, tr, info, dr, rungs, qc, pick[key], over)
            if (climb_on_kept and decisions[key]["word"] == "INCUMBENT KEPT"
                    and list(row["candidate"]) != ["P0"]):
                nc, step = paired_next_candidate(list(row["candidate"]), tr, row.get("cand_info") or {}, dr,
                                                 rungs, qc, pick[key], False)
                if nc is not None:
                    why_climb = f"; climbing on the kept {'+'.join(row['candidate'])}"
                    head += why_climb
        else:
            nc, step = None, None                        # a key that has stopped stays stopped
        stacks[key], cands[key], steps[key], tried[key] = inc, nc, step, tr
        why[key] = head + (f"; next candidate {'+'.join(nc)} (step {step})" if nc else "; no step left")
    res = dict(round=cur["round"] + 1, device=cur["device"], rungs=list(rungs), paired=True, stacks=stacks,
               candidates=cands, steps=steps, tried=tried, pick=pick, decisions=decisions, why=why,
               model_seconds_so_far=cur.get("model_seconds_so_far", 0.0))
    if climb_on_kept:
        res["climb_on_kept"] = True
    stop, sreason = _paired_stop(res, cur["round"], k_max, budget, next_cost)
    res.update(stop=stop, stop_reason=sreason)
    return res


# ------------------------------------------------------------------------------------------------ the ledger

def ledger_load(path):
    """The ledger's lines. A file that is not a JSON list (ledger-replay-2026-11-21: before, the night stopped with a
    Python error) is read as one line {"unreadable": why}, which ledger_start does not use and names."""
    if not path or not os.path.exists(path):
        return []
    try:
        d = json.load(open(path))
    except (OSError, ValueError) as exc:
        return [dict(unreadable=f"not JSON ({type(exc).__name__})")]
    if not isinstance(d, list):
        return [dict(unreadable=f"not a list of nights ({type(d).__name__})")]
    return d


def _ledger_line_problem(last):
    """Why the ledger's last line cannot be used as a line, or None."""
    if not isinstance(last, dict):
        return f"its last line is not a night ({type(last).__name__})"
    if "unreadable" in last:
        return f"the file is {last['unreadable']}"
    ls = last.get("last_stacks")
    if ls is not None and not isinstance(ls, dict):
        return f"its last_stacks is not a table of keys ({type(ls).__name__})"
    return None


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
    bad = _ledger_line_problem(last)
    if bad:
        dec["why"] = f"ledger unreadable ({bad}): start from tonight's diagnostics"
        return out, dec
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
        if not (isinstance(prev, list) and prev and all(isinstance(x, str) for x in prev)):
            dec["per_key"][k] = f"the ledger's stack for this key is not a list of layers ({prev!r}): ledger not used"
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


# ------------------------------------------------------------------------------------------------ replay

class NotInRecord(Exception):
    """A circuit of a replayed night that the record does not hold."""


def _record_jobs(record):
    """[(folder name, job dir)] of every recorded job with raw counts, in name order."""
    out = []
    for d in sorted(os.listdir(record)):
        j = os.path.join(record, d, "job")
        if os.path.exists(os.path.join(j, "raw_counts.json")) and os.path.exists(os.path.join(j, "manifest.json")):
            out.append((d, j))
    return out


def _cells_of(job_dir):
    """[(cell name, role, point, gate digest)] per pub, in pub order."""
    from qiskit import qpy
    from psf_os.gatehash import circuits_digest
    man = json.load(open(os.path.join(job_dir, "manifest.json")))
    cells = {c["cell"]: c for c in man["cells"]}
    cs = qpy.load(open(os.path.join(job_dir, "circuits.qpy"), "rb"))
    out = []
    for p, c in zip(man["pubs"], cs):
        cell = cells[p["cell"]]
        out.append((cell["name"], cell["role"], p["point"], circuits_digest([c])))
    return out


def replay_job(record, folder):
    """Answer the job in `folder` from the recorded night `record`: job/raw_counts.json (rows in this job's pub order),
    job/submission.json (local, replay) and replay.json. The recorded job holding the most of this job's test cells is
    the primary source; a cell it lacks comes from the first other recorded job holding it (name and gates equal).
    The sentinel is taken by its name from the primary source. Raises NotInRecord, writing nothing, when a test
    circuit is in no recorded job."""
    job = os.path.join(folder, "job")
    new = _cells_of(job)
    srcs = []
    for name, j in _record_jobs(record):
        cells = _cells_of(j)
        rows = json.load(open(os.path.join(j, "raw_counts.json")))["rows"]
        idx = {}
        for k, (n, role, pt, dg) in enumerate(cells):
            key = (n, pt, "sentinel") if role == "sentinel" else (n, pt, dg)
            idx.setdefault(key, rows[k]["counts"])
        srcs.append((name, idx))
    keys = [(n, pt, "sentinel") if role == "sentinel" else (n, pt, dg) for n, role, pt, dg in new]
    test = [k for k, (_, role, _, _) in zip(keys, new) if role != "sentinel"]
    if not srcs:
        raise NotInRecord(f"記録にない: the record {record} holds no job with raw counts")
    # the most of this job's test cells; at a tie the recorded job of the same name (a night that ran the same
    # circuits in two rounds answers each round from its own job), then the first in name order
    tag = os.path.basename(os.path.abspath(folder))
    primary = max(srcs, key=lambda s: (sum(1 for k in test if k in s[1]), s[0] == tag))
    order = [primary] + [s for s in srcs if s is not primary]
    missing, rows, source = [], [], []
    for k, (n, role, pt, _) in zip(keys, new):
        got = next(((nm, idx[k]) for nm, idx in order if k in idx), None)
        if got is None:
            missing.append(f"{n}" + (f"[{pt}]" if pt else ""))
            continue
        rows.append(dict(pub=len(rows), counts=got[1]))
        source.append(dict(cell=n, point=pt, role=role, from_job=got[0]))
    if missing:
        raise NotInRecord(f"記録にない: {len(missing)} of {len(new)} circuits of {os.path.basename(folder)} are in no "
                          f"recorded job of {os.path.basename(os.path.abspath(record))} (name and gates): "
                          + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    man = json.load(open(os.path.join(job, "manifest.json")))
    from_jobs = sorted({s["from_job"] for s in source if s["role"] != "sentinel"})
    json.dump(dict(backend=man["fetch"]["backend"], job_id=f"replay:{primary[0]}", local=True,
                   replay_from=os.path.abspath(record), submitted=now()),
              open(os.path.join(job, "submission.json"), "w"), indent=1)
    json.dump(dict(fetched=now(), job_id=f"replay:{primary[0]}", status="DONE", metrics=None, rows=rows,
                   replay_from=os.path.abspath(record)),
              open(os.path.join(job, "raw_counts.json"), "w"), indent=1)
    rep_ = dict(record=os.path.abspath(record), primary=primary[0], test_cells_from=from_jobs,
                one_recorded_job=len(from_jobs) <= 1, cells=source)
    json.dump(rep_, open(os.path.join(folder, "replay.json"), "w"), indent=1)
    return rep_


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

    def build_dry(self):
        """The experiment's build flag: dry, except on a replayed night (its circuits are the device night's)."""
        c = self.cfg()
        return bool(c["dry"]) and not c.get("replay_from")

    # ---------------------------------------------------------------------------------------------- IBM or local
    def submit(self, folder):
        job = os.path.join(folder, "job")
        tag = os.path.basename(folder)
        if os.path.exists(os.path.join(job, "submission.json")):
            self.log(f"{tag}: already submitted (submission.json), not sent again")
            return True
        if self.cfg().get("replay_from"):
            try:
                r = replay_job(self.cfg()["replay_from"], folder)
            except NotInRecord as exc:
                self.replay_stop = str(exc)
                return False
            self.log(f"{tag}: replayed from the record (primary {r['primary']}; test cells from "
                     f"{', '.join(r['test_cells_from'])})")
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
                if getattr(self, "replay_stop", None):
                    return self.stop(f"{name}: {self.replay_stop}", replay_missing=True)
                return self.stop(f"{name}: submission refused or failed (see run_log.txt)")
        st = self.collect(folder)
        if st == "WAIT":
            if self.cfg()["dry"]:
                return self.stop(f"{name}: the dry run has no result for this job (simulation output missing); run "
                                 "resume after it is written", resumable=True)
            return self.stop(f"{name}: not finished after {COLLECT_WAIT_MIN} min; run resume", resumable=True)
        if st != "DONE":
            # 2026-11-19 (the workplace VS): only a failed job ends the night; any other state (UNKNOWN, a result not
            # fetched, a dropped connection) stops it resumably: a job holding a submission.json is never sent again
            if st.startswith("FAILED"):
                return self.stop(f"{name}: the job ended {st}")
            return self.stop(f"{name}: the job's state is {st}; run resume after checking it", resumable=True)
        _quiet(score, os.path.join(folder, "score.txt"), folder)
        self.log(f"{name}: scored")
        return True

    def diagnostics(self):
        for tag, build, check, score, scores_file in self.exp.diagnostics():
            folder = os.path.join(self.out, tag)
            if not os.path.exists(folder):
                try:
                    _quiet(build, os.path.join(self.out, f"build_{tag}.txt"), self.device_dir(), folder,
                           self.build_dry())
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
        if c.get("paired"):
            R = paired_rule1(R, stacks, dec)
            R.update(ledger=dec, ledger_last=led[-1] if led else None, calibration_time=cal, rungs=c["rungs"])
            json.dump(R, open(p, "w"), indent=1)
            self.log("RULE1.json (paired): incumbents " + json.dumps(R["stacks"]) + ", candidates " +
                     json.dumps(R["candidates"]) + f"; ledger: {dec['why']}")
            return
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
            ok = _quiet(self.exp.build, os.path.join(tmp, "x.txt"), self.device_dir(), d, rule_path, self.build_dry())
            if ok is False or not os.path.exists(os.path.join(d, "build.json")):
                raise SystemExit("the round's build was refused: " + open(os.path.join(tmp, "x.txt")).read()[-300:])
            return float(json.load(open(os.path.join(d, "build.json")))["qpu_seconds_model"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def next_rule(self, k):
        c = self.cfg()
        rounds = [json.load(open(os.path.join(self.out, f"r{j}", "round_scores.json"))) for j in range(1, k + 1)]
        rf = os.path.join(self.out, f"r{k}")
        if c.get("paired"):
            return next_stacks_paired(rounds, c["rungs"], q_choice=lambda key, info, dr: self.exp.q_choice(
                rf, key, info, dr), k_max=c["k_max"], budget=c["budget"], climb_on_kept=c.get("climb_on_kept", False))
        return next_stacks(rounds, c["rungs"], q_choice=lambda key, row, dr: self.exp.q_choice(rf, key, row, dr),
                           k_max=c["k_max"], budget=c["budget"], between_job_sigma=c.get("between_job_sigma"))

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
                                rule_k, self.build_dry())
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
                if Rn.get("paired"):
                    self.log(f"RULE{k + 1}.json candidates: {json.dumps(Rn['candidates'])}")
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
            # the night's name carries its date (after 2026-11-19, the workplace VS: OUT folders named alike, e.g.
            # ".../loop" every night, gave lines no one could tell apart)
            e = ledger_entry(f"{os.path.basename(self.out)}@{now()[:10]}", c["device"], R1.get("calibration_time"), c["rungs"], rounds,
                             R1.get("ledger", {}))
            if c.get("paired") and rounds:
                # the paired night's final stacks: the incumbents after the last round's decisions (the next rule)
                last_rule = os.path.join(self.out, f"RULE{rounds[-1]['round'] + 1}.json")
                if os.path.exists(last_rule):
                    e["last_stacks"] = json.load(open(last_rule))["stacks"]
                e["paired"] = True
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
        """The night under its own park setting (park-default-2026-11-21): run.json's "park", False when absent (a
        night started before parking was the default), so a resumed old night builds as it began."""
        from psf_os import feedforward as FF
        with FF.park_default(bool(self.cfg().get("park", False))):
            return self._go()

    def _go(self):
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


def _park_now():
    from psf_os import feedforward as FF
    return bool(FF.PARK_DEFAULT)


def _park_for(replay_from):
    """The park setting of a new night (release-0.3.0): the default, except for a replay, which takes the recorded
    night's own setting (its run.json's "park", False when absent), so that its jobs are the recorded circuits."""
    if not replay_from:
        return _park_now()
    p = os.path.join(replay_from, "run.json")
    return bool(json.load(open(p)).get("park", False)) if os.path.exists(p) else False


def start(out, exp, dry_devdir=None, noise=None, k_max=None, budget=None, rungs=RUNGS, ledger=None, zip_path=None,
          extra=None, paired=False, replay_from=None):
    """A new night in OUT (refused if OUT exists). Device runs need zip_path; noise / k_max / budget are for dry runs.
    paired (loop-paired-2026-11-21; default False: the cfg and the night as before): the paired rule; the experiment's
    build reads the rule's candidates.
    replay_from (ledger-replay-2026-11-21; default None: the cfg and the night as before): a recorded night's folder
    whose raw counts answer this dry night's jobs (replay_job); dry runs only, and never with noise."""
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: use resume")
    if replay_from and not dry_devdir:
        raise SystemExit("replay_from is for dry runs only (give dry_devdir, e.g. the record's fetch_in device)")
    if replay_from and noise:
        raise SystemExit("replay_from answers the jobs from the record: no noise")
    if replay_from and not os.path.isdir(replay_from):
        raise SystemExit(f"replay_from: {replay_from} is not a folder")
    if not dry_devdir and (noise or k_max or budget):
        raise SystemExit("noise, k_max and budget are for dry runs only")
    if not dry_devdir and not zip_path:
        raise SystemExit("a zip file is required for a device run")
    os.makedirs(out)
    from psf_os import job as J
    cfg = dict(build_env=J.build_env(), dry=bool(dry_devdir), dry_devdir=os.path.abspath(dry_devdir) if dry_devdir else None, noise=noise,
               k_max=k_max or exp.k_max, budget=budget or exp.budget_s, rungs=list(rungs),
               ledger=os.path.abspath(ledger) if ledger else None, zip=os.path.abspath(zip_path) if zip_path else None,
               started=now(), device=None, experiment=exp.name, **(extra or {}),
               **({"paired": True} if paired else {}),
               **({"replay_from": os.path.abspath(replay_from)} if replay_from else {}),
               **({"park": True} if _park_for(replay_from) else {}))
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
CELL_VERSION_KEYS = ("qpy_sha256", "run_sha256")


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
        try:
            ta, sa = _split_digests(rec_job)
            tb, sb = _split_digests(new_job)
        except Exception as exc:  # noqa: BLE001 - e.g. a QPY format newer than this qiskit reads (2026-11-19)
            return f"cannot be read here ({type(exc).__name__}: {str(exc)[:160]})", "unreadable"
        if ta != tb:
            return "test circuits differ (gates)", "gates"
        how = "test circuits' gates (the bytes differ: other versions)" + (
            "" if sa == sb else "; the ring sentinel differs (beside, not judged)")
    ma, mb = (json.load(open(os.path.join(x, "manifest.json"))) for x in (rec_job, new_job))
    drop = VERSION_KEYS if how != "bytes" else ("written", "env")
    for m in (ma, mb):
        for k in drop:
            m.pop(k, None)
        if how != "bytes":
            # the per-cell byte hashes follow the versions too (2026-11-19, the workplace VS: a night built on one
            # machine and verified on another read DIFFERENT on /cells[*]/run_sha256 with the same gates)
            for cell in m.get("cells", []):
                for k in CELL_VERSION_KEYS:
                    cell.pop(k, None)
    return first_difference(ma, mb), how


def verify(out, exp, resim=False):
    """Rebuild, re-score and re-rule every step from the records: a line SAME / DIFFERENT per item. The rebuilds use
    the night's park setting (park-default-2026-11-21: run.json's "park", False when absent)."""
    from psf_os import feedforward as FF
    cfg = json.load(open(os.path.join(os.path.abspath(out), "run.json")))
    with FF.park_default(bool(cfg.get("park", False))):
        return _verify(out, exp, resim)


def _verify(out, exp, resim=False):
    out = os.path.abspath(out)
    cfg = json.load(open(os.path.join(out, "run.json")))
    devdir = os.path.join(out, "fetch_in", cfg["device"], "dev")
    bdry = bool(cfg["dry"]) and not cfg.get("replay_from")      # a replayed night's builds are the device night's
    lines, bad = [], []

    unchecked = []

    def note(what, diff):
        how = None
        if isinstance(diff, tuple):
            diff, how = diff
        if how == "unreadable":
            # not a difference and not the same: this machine cannot read the record (verify with a qiskit at least
            # as new as the one that wrote it)
            lines.append(f"NOT CHECKED {what}: {diff}")
            unchecked.append(what)
            return
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
            _quiet(build, os.path.join(tmp, "x.txt"), devdir, new, bdry)
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
            if cfg.get("paired"):
                P = paired_rule1(R, stacks, dec)
                keys = ("stacks", "candidates", "steps", "tried", "pick")
                note("round 1 (paired): the ledger and the candidates re-applied", first_difference(
                    dict({k: rec1.get(k) for k in keys}, ledger=rec1["ledger"]),
                    dict({k: P.get(k) for k in keys}, ledger=dec)))
            else:
                note("round 1: the ledger re-applied", first_difference(dict(stacks=rec1["stacks"],
                                                                             ledger=rec1["ledger"]),
                                                                        dict(stacks=stacks, ledger=dec)))
        k = 1
        while os.path.exists(os.path.join(out, f"RULE{k}.json")):
            rec = os.path.join(out, f"r{k}")
            if not os.path.isdir(rec):
                break
            new = os.path.join(tmp, f"r{k}")
            _quiet(exp.build, os.path.join(tmp, "x.txt"), devdir, new, os.path.join(out, f"RULE{k}.json"), bdry)
            j = job_of(rec)
            if j:
                note(f"r{k} job rebuilt", same_job(j, os.path.join(new, "job")))
            if os.path.exists(os.path.join(rec, "round_scores.json")):
                note(f"r{k} re-scored", rescore(rec, exp.score, "round_scores.json", tmp, extra=("rule.json",)))
                nxt = os.path.join(out, f"RULE{k + 1}.json")
                if os.path.exists(nxt):
                    rounds = [json.load(open(os.path.join(out, f"r{i}", "round_scores.json"))) for i in range(1, k + 1)]
                    if cfg.get("paired"):
                        Rn = next_stacks_paired(rounds, cfg["rungs"], q_choice=lambda key, info, dr, rf=rec:
                                                exp.q_choice(rf, key, info, dr), k_max=cfg["k_max"],
                                                budget=cfg["budget"], climb_on_kept=cfg.get("climb_on_kept", False))
                    else:
                        Rn = next_stacks(rounds, cfg["rungs"], q_choice=lambda key, row, dr, rf=rec: exp.q_choice(
                            rf, key, row, dr), k_max=cfg["k_max"], budget=cfg["budget"],
                            between_job_sigma=cfg.get("between_job_sigma"))
                    note(f"RULE{k + 1}.json re-ruled", first_difference(json.load(open(nxt)), Rn))
            k += 1
        if cfg.get("replay_from"):
            # a replayed night: every job's counts are answered again from the record and compared
            for d in sorted(os.listdir(out)):
                rec = os.path.join(out, d)
                if not os.path.exists(os.path.join(rec, "replay.json")):
                    continue
                cp = os.path.join(tmp, "replay", d)            # the same folder name: replay_job's tie rule
                shutil.copytree(os.path.join(rec, "job"), os.path.join(cp, "job"),
                                ignore=shutil.ignore_patterns("raw_counts.json", "submission.json"))
                try:
                    replay_job(cfg["replay_from"], cp)
                    diff = first_difference(json.load(open(os.path.join(rec, "job", "raw_counts.json")))["rows"],
                                            json.load(open(os.path.join(cp, "job", "raw_counts.json")))["rows"])
                except NotInRecord as exc:
                    diff = str(exc)
                note(f"{d} replayed again from the record", diff)
        elif resim and cfg["dry"]:
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
    if bad:
        print(f"DIFFERENT ({len(bad)}: {', '.join(bad)})")
    elif unchecked:
        print(f"NOT CHECKED ({len(unchecked)}: {', '.join(unchecked)}); the rest SAME")
    else:
        print("SAME")
    return not bad and not unchecked
