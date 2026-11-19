"""psf_os.loop.next_stacks on hand-made rounds (workplace VS Claude, 2026-11-19): the order of the rules (R1, R2
before the rungs; +M before +F), R3 taking F off with the base switched back, and the stops. Each expectation is
written from the module text, not from the code.

  python vs_rule.py
"""
from psf_os.loop import next_stacks

RES = []


def rnd(k, rows, dr=None, spent=0.0):
    return dict(round=k, device="d", rows=rows, dr=dr or {}, model_seconds_so_far=spent)


def row(stack, verdict, **kw):
    return dict(stack=stack, verdict=verdict, overshoot=verdict == "OVERSHOOT", output_qubit=3, **kw)


GAP = {"3": dict(gap=0.19, two_se=0.012, sig=True)}           # q3 reads |1> badly: against P1's frame
REV = {"3": dict(gap=-0.05, two_se=0.012, sig=True)}           # reversed: against P2's frame


def case(name, rounds, key, want_stack, want_reason, **kw):
    R = next_stacks(rounds, **kw)
    ok = R["stacks"][key] == want_stack and R["why"][key].startswith(want_reason)
    RES.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {'+'.join(R['stacks'][key])} ({R['why'][key][:60]})")
    return R


case("R1 first: an overshoot with a gap against the frame gets P2+B, not F", [rnd(1, {"2": row(["P1"], "OVERSHOOT")}, GAP)],
     "2", ["P2", "B"], "R1")
case("R2 for a balanced stack", [rnd(1, {"1": row(["P2", "B"], "PRESCRIPTION HELPS", share_gap=0.1, share_gap_two_se=0.02,
                                                     output_qubit_nf=3)}, GAP)], "1", ["P2", "B", "Q2"], "R2",
     q_choice=lambda k, r, d: 2)
case("rungs after: HELPS with a gap gives +M first", [rnd(1, {"2": row(["P1"], "PRESCRIPTION HELPS")}, GAP)],
     "2", ["P1", "M"], "RM")
case("then +F (base switched)", [rnd(1, {"2": row(["P1", "M"], "PRESCRIPTION HELPS")}, GAP)],
     "2", ["P2", "M", "F"], "RF")
case("rungs in the other order: +F first", [rnd(1, {"2": row(["P1"], "PRESCRIPTION HELPS")}, GAP)],
     "2", ["P2", "F"], "RF", rungs=("F", "M"))
case("F for P2 needs the reversed gap", [rnd(1, {"2": row(["P2"], "NOT SHOWN")}, REV)], "2", ["P1", "F"], "RF")
case("no F without a gap", [rnd(1, {"2": row(["P1", "M"], "PRESCRIPTION HELPS")}, {})], "2", ["P1", "M"], "R4")
case("no rung with B", [rnd(1, {"2": row(["P2", "B"], "PRESCRIPTION HELPS", share_gap=0.0, share_gap_two_se=0.02,
                                                  output_qubit_nf=3)}, GAP)], "2", ["P2", "B"], "R4")
r1 = rnd(1, {"2": row(["P2", "M", "F"], "NOT SHOWN")}, {})
r2 = rnd(2, {"2": row(["P2", "M", "F"], "NOT SHOWN")}, {})
case("R3 takes F off and switches the base back", [r1, r2], "2", ["P1", "M"], "R3")
r1 = rnd(1, {"2": row(["P1", "M"], "NOT SHOWN")}, {})
r2 = rnd(2, {"2": row(["P1", "M"], "NOT SHOWN")}, {})
case("R3 takes M off", [r1, r2], "2", ["P1"], "R3")
R = next_stacks([rnd(1, {"2": row(["P1"], "NOT SHOWN")}, {})])
ok = not R["stop"]
RES.append(ok)
print(f"{'PASS' if ok else 'FAIL'}  a first NOT SHOWN does not stop the night ({R['stop_reason']})")
R = next_stacks([rnd(1, {"2": row(["P1"], "PRESCRIPTION HELPS")}, {})], rungs=())
ok = R["stop"] and R["stop_reason"].startswith("fixed point")
RES.append(ok)
print(f"{'PASS' if ok else 'FAIL'}  no change, no overshoot: fixed point")
R = next_stacks([rnd(4, {"2": row(["P1"], "PRESCRIPTION HELPS")}, {})], k_max=4)
ok = R["stop"] and "round limit" in R["stop_reason"]
RES.append(ok)
print(f"{'PASS' if ok else 'FAIL'}  round 4 of K 4 with a change pending: round limit ({R['stop_reason']})")
R = next_stacks([rnd(1, {"2": row(["P1"], "PRESCRIPTION HELPS")}, {}, spent=200.0)], next_cost=50.0, budget=240.0)
ok = R["stop"] and R["stop_reason"].startswith("budget")
RES.append(ok)
print(f"{'PASS' if ok else 'FAIL'}  200 + 50 > 240: budget ({R['stop_reason']})")
print(f"{len(RES)} checks, {RES.count(False)} FAIL")
