"""Order placement candidates by the user's own quirk ledger (OS item 4; owner 2026-11-13 11:25 JST: allowed as a
SPECIAL CONDITION of the placement policy, and every use must be written in the record).

The ledger holds, per device, day label and physical qubit, the user's own measurements from earlier jobs: "floor"
(1 - <Z> right after reset, i.e. readout loss; 0 is perfect) and "mirror" (the mirror circuit's <Z> standardised
within its circuit). It is built from the user's job records only (notebook gpu/quirk_ledger.py). IBM's calibration
numbers are never read here.

Rule "floor-fifth", version 1 (fixed; no fitted weight, no search):
  1. Of the device's ledger days strictly before `before`, take the latest one (the "previous day"). Nothing dated
     `before` or later is read.
  2. Mark the weakest fifth of that day's qubits by floor loss: the k = max(1, n // 5) qubits with the largest floor,
     ties broken by the smaller label. If the day has fewer than MIN_QUBITS qubits, nothing is marked.
  3. For every candidate count how many of its qubits are marked, and sort the candidates by that count, ascending,
     stably (equal counts keep the caller's order). No marks -> the caller's order unchanged.
  The mirror value is not used by this version (on the records of 11-04 to 11-12 its day-to-day rank correlation was
  about +0.3, against about +0.8 for the floor; notebook notes/quirk_ledger_2026-11-13).

    from psf_os import quirk
    ordered, report = quirk.rank("ibm_fez", [{"name": "start (1,0,0)", "qubits": [...]}, ...], "ledger.json",
                                 before="2026-11-13")
    # report: rule, version, special_condition, ledger_day, marked, counts, moved, ...

Ledger form (dict or path of a JSON file):
  {"format": "psf-os quirk ledger 1",
   "devices": {"ibm_fez": {"2026-11-12": {"17": {"floor": 0.03, "mirror": -0.4, "n": 2}, ...}, ...}, ...}}
Numbers may be given as strings (e.g. "3.1e-02"); they are read with float().
"""
from __future__ import annotations

import json
import re

RULE, VERSION = "floor-fifth", 1
FORMAT = "psf-os quirk ledger 1"
MIN_QUBITS = 10
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load(ledger):
    """The ledger as a dict (a dict is returned as it is; a path is read as JSON)."""
    if isinstance(ledger, dict):
        L = ledger
    else:
        with open(ledger, encoding="utf-8") as f:
            L = json.load(f)
    if L.get("format") != FORMAT:
        raise ValueError(f"not a quirk ledger (format {L.get('format')!r}, expected {FORMAT!r})")
    return L


def previous_day(L, dev_name, before):
    """The latest day label of the device strictly before `before`, or None. Later days are not looked at."""
    if not _DAY.match(str(before)):
        raise ValueError(f"before must be a day label YYYY-MM-DD, got {before!r}")
    days = [d for d in L.get("devices", {}).get(dev_name, {}) if _DAY.match(d) and d < before]
    return max(days) if days else None


def marks(L, dev_name, before):
    """(day, marked qubits as a sorted list, number of qubits on that day). Day None if there is none before."""
    day = previous_day(L, dev_name, before)
    if day is None:
        return None, [], 0
    rows = L["devices"][dev_name][day]
    fl = sorted(((float(v["floor"]), int(q)) for q, v in rows.items() if v.get("floor") is not None),
                key=lambda t: (-t[0], t[1]))
    if len(fl) < MIN_QUBITS:
        return day, [], len(fl)
    k = max(1, len(fl) // 5)
    return day, sorted(q for _, q in fl[:k]), len(fl)


def rank(dev_name, candidates, ledger, before):
    """candidates: list of dicts {"name": str, "qubits": [physical...]} in the caller's fixed order.
    ledger: path or dict of the quirk ledger, built only from the user's own job records dated before `before`.
    Returns (ordered_candidates, report)."""
    L = load(ledger)
    day, marked, n_day = marks(L, dev_name, before)
    mset = set(marked)
    counts = [sum(1 for q in c["qubits"] if int(q) in mset) for c in candidates]
    order = sorted(range(len(candidates)), key=lambda i: counts[i]) if mset else list(range(len(candidates)))
    ordered = [candidates[i] for i in order]
    report = dict(
        rule=RULE, version=VERSION,
        special_condition="placement ordered by the user's own quirk ledger (owner 2026-11-13 11:25 JST); "
                          "no IBM calibration number read",
        device=dev_name, before=str(before), ledger_day=day, ledger_day_qubits=n_day,
        days_read=sorted(d for d in L.get("devices", {}).get(dev_name, {}) if _DAY.match(d) and d < str(before)),
        marked=marked,
        counts={c["name"]: k for c, k in zip(candidates, counts)},
        order=[c["name"] for c in ordered],
        chosen=ordered[0]["name"] if ordered else None,
        default=candidates[0]["name"] if candidates else None,
        moved=bool(order) and order[0] != 0,
    )
    if day is None:
        report["note"] = "no ledger day before `before` for this device: order unchanged"
    elif not mset:
        report["note"] = f"previous day has {n_day} qubits (< {MIN_QUBITS}): nothing marked, order unchanged"
    return ordered, report
