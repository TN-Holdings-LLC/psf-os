"""psf_os.quirk.rank on hand-made ledgers (made-up values; no device record needed)."""
import copy
import json

import pytest

from psf_os import quirk


def ledger(days):
    """days: {day: {qubit: floor}} for a made-up device 'dev'."""
    return {"format": quirk.FORMAT,
            "devices": {"dev": {d: {str(q): {"floor": f, "mirror": 0.0, "n": 1} for q, f in rows.items()}
                                for d, rows in days.items()}}}


# day A: 20 qubits 0..19, qubits 3 and 7 the worst (weakest fifth of 20 = 4 qubits: 3, 7, 11, 15)
DAY_A = {q: 0.01 for q in range(20)} | {3: 0.30, 7: 0.25, 11: 0.20, 15: 0.15}
CANDS = [{"name": "c0", "qubits": [0, 1, 2, 3]},        # 1 marked (3)
         {"name": "c1", "qubits": [4, 5, 6, 8]},        # 0 marked
         {"name": "c2", "qubits": [3, 7, 11, 12]},      # 3 marked
         {"name": "c3", "qubits": [9, 10, 12, 13]}]     # 0 marked


def names(c):
    return [x["name"] for x in c]


def test_orders_by_marked_count_and_keeps_ties_in_order():
    out, rep = quirk.rank("dev", CANDS, ledger({"2026-01-10": DAY_A}), before="2026-01-11")
    assert names(out) == ["c1", "c3", "c0", "c2"]          # c1 before c3 (tie at 0, caller's order)
    assert rep["marked"] == [3, 7, 11, 15]
    assert rep["ledger_day"] == "2026-01-10"
    assert rep["counts"] == {"c0": 1, "c1": 0, "c2": 3, "c3": 0}
    assert rep["chosen"] == "c1" and rep["default"] == "c0" and rep["moved"]
    assert (rep["rule"], rep["version"]) == ("floor-fifth", 1)


def test_all_ties_keep_the_callers_order():
    cands = [{"name": f"t{i}", "qubits": [0, 1, 2]} for i in range(5)]   # all with 0 marks
    out, rep = quirk.rank("dev", cands, ledger({"2026-01-10": DAY_A}), before="2026-01-11")
    assert names(out) == names(cands) and not rep["moved"]


def test_does_not_look_at_the_future():
    past = ledger({"2026-01-10": DAY_A})
    # a later day that would mark completely different qubits, and a day equal to `before`
    fut = copy.deepcopy(past)
    bad = {q: 0.01 for q in range(20)} | {4: 0.9, 5: 0.9, 6: 0.9, 8: 0.9}
    for d in ("2026-01-11", "2026-01-12"):
        fut["devices"]["dev"][d] = ledger({d: bad})["devices"]["dev"][d]
    a = quirk.rank("dev", CANDS, past, before="2026-01-11")
    b = quirk.rank("dev", CANDS, fut, before="2026-01-11")
    assert names(a[0]) == names(b[0])
    assert a[1]["marked"] == b[1]["marked"] and b[1]["ledger_day"] == "2026-01-10"
    assert "2026-01-11" not in b[1]["days_read"]
    # with `before` after the later day, the later day is the one used
    c = quirk.rank("dev", CANDS, fut, before="2026-01-13")
    assert c[1]["ledger_day"] == "2026-01-12" and c[1]["marked"] == [4, 5, 6, 8]


def test_uses_only_the_latest_earlier_day():
    old = {q: 0.01 for q in range(20)} | {0: 0.9, 1: 0.9, 2: 0.9, 9: 0.9}
    out, rep = quirk.rank("dev", CANDS, ledger({"2026-01-05": old, "2026-01-10": DAY_A}), before="2026-01-11")
    assert rep["ledger_day"] == "2026-01-10" and rep["marked"] == [3, 7, 11, 15]


def test_no_marks_no_reordering():
    # no day before, another device only, and a day with too few qubits: the caller's order every time
    cases = [ledger({"2026-01-12": DAY_A}),
             {"format": quirk.FORMAT, "devices": {"other": ledger({"2026-01-10": DAY_A})["devices"]["dev"]}},
             ledger({"2026-01-10": {q: 0.5 if q == 3 else 0.01 for q in range(quirk.MIN_QUBITS - 1)}})]
    for L in cases:
        out, rep = quirk.rank("dev", CANDS, L, before="2026-01-11")
        assert names(out) == names(CANDS) and rep["marked"] == [] and not rep["moved"] and "note" in rep


def test_reads_a_path_and_string_numbers(tmp_path):
    L = ledger({"2026-01-10": DAY_A})
    for rows in L["devices"]["dev"].values():
        for v in rows.values():
            v["floor"] = f"{v['floor']:.3e}"
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(L))
    out, _ = quirk.rank("dev", CANDS, str(p), before="2026-01-11")
    assert names(out) == ["c1", "c3", "c0", "c2"]


def test_refuses_other_forms():
    with pytest.raises(ValueError):
        quirk.rank("dev", CANDS, {"devices": {}}, before="2026-01-11")
    with pytest.raises(ValueError):
        quirk.rank("dev", CANDS, ledger({"2026-01-10": DAY_A}), before="11 Jan")
