"""T1: the registry made by psf-os from TOQB's device records equals TOQB's own list files, byte for byte.

Needs a TOQB checkout with its results (TOQB_DIR); skipped otherwise. The base list and the labels are the inputs
TOQB used (recorded in each folder's times.txt / list_after.json); nothing is fitted.
"""
import json
import os

import pytest

from psf_os import registry as R

TQ = os.environ.get("TOQB_DIR", "")
RES = os.path.join(TQ, "results")
pytestmark = pytest.mark.skipif(not os.path.isdir(RES), reason="TOQB_DIR with results/ not set")

CASES = [
    # (folder, device, base list, [(job, runs, label)])
    ("two-devices-2026-11-01/fez/scan", "ibm_fez", "defect-list-replay-2026-10-31/observations.json",
     [("job_A", "runs_A", "fezSA")]),
    ("two-devices-2026-11-01/marrakesh/scan", "ibm_marrakesh", "defect-list-replay-2026-10-31/observations.json",
     [("job_A", "runs_A", "marSA")]),
    ("coverage-days-2026-11-01-day2", "ibm_kingston", "coverage-days-2026-10-31-day1/list_after.json",
     [("job_A", "runs_A", "day2A"), ("job_B", "runs_B", "day2B")]),
]


@pytest.mark.parametrize("folder,device,base,jobs", CASES, ids=[c[0] for c in CASES])
def test_registry_equals_toqb_list(folder, device, base, jobs, tmp_path):
    d = os.path.join(RES, folder)
    # the list TOQB read as its base is the file recorded in list_after.json's "base"
    want = json.load(open(os.path.join(d, "list_after.json")))
    basef = os.path.join(RES, base)
    assert R.sha256_file(basef) == want["base"]["sha256"]
    # TOQB named the base list_before.json in that folder; use the same name so the recorded file name matches
    local = tmp_path / "list_before.json"
    local.write_bytes(open(basef, "rb").read())
    reg, _ = R.ingest([(device, os.path.join(d, j), os.path.join(d, r), lab) for j, r, lab in jobs], base=str(local))
    assert reg == want
    assert json.dumps(reg, indent=1) == open(os.path.join(d, "list_after.json")).read()


@pytest.mark.parametrize("folder,device,base,jobs", CASES, ids=[c[0] for c in CASES])
def test_status_equals_toqb_status(folder, device, base, jobs):
    d = os.path.join(RES, folder)
    want = [ln.rstrip("\n") for ln in open(os.path.join(d, "list_after_status.txt")) if ln.strip()]
    got = R.status_lines(json.load(open(os.path.join(d, "list_after.json"))), device)
    assert got == want
