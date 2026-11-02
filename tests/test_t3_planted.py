"""T3: planted defects are found, listed after two scans, and avoided; the usable part of the ring grows.

The simulator (Aer, stabilizer method) runs the Clifford-point echo (h = pi/4) on FakeFez with a ZZ Pauli error of
probability 0.15 after every CZ on three planted couplers: one on the standard n = 40 ring, one on the standard
n = 72 ring, one off both. The planted couplers are known to this test only; psf-os sees counts. The chain is psf-os's
own: scan job -> scan-ingest (twice, so the dead paths are LISTED) -> plan -> avoid -> assemble -> run -> score.
The scan at h = 0.6 cannot be simulated at n = 72; the Clifford point keeps the same ring, schedule and acceptance.
"""
import json
import os

import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from psf_os import device as Dv  # noqa: E402
from psf_os import job as J  # noqa: E402
from psf_os import plan as P  # noqa: E402
from psf_os import registry as R  # noqa: E402
from psf_os import sim  # noqa: E402

SPEC = dict(kind="echo_clifford", D=24, shots=1000)
P_ERR = 0.15


def _rings(runs):
    out = {}
    for f in os.listdir(runs):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(runs, f)))
            n = r["task"]["n"]
            out[n] = [int(r["echo"]["measured_qubits"][str(j)]) for j in range(n)]
    return out


def _on_ring(ring, e):
    n = len(ring)
    return any({ring[j], ring[(j + 1) % n]} == set(e) for j in range(n))


def test_planted_defects_are_listed_and_avoided(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    now = P.now_cet()
    s1 = P.scan_job(dev, str(tmp_path / "s1"), spec=SPEC)
    rings = _rings(s1["runs"])
    planted = [(rings[40][12], rings[40][13]), (rings[72][50], rings[72][51]), (100, 101)]
    assert not any(_on_ring(rings[n], planted[2]) for n in rings)
    reg_files = []
    base = None
    for k, s in enumerate([s1, P.scan_job(dev, str(tmp_path / "s2"), spec=SPEC)]):
        sim.submit(s["job"], "pauli", seed=1000 + k, planted=planted, p=P_ERR)
        reg, obs = P.scan_ingest(dev.name, s["job"], s["runs"], base, f"scan{k + 1}")
        f = str(tmp_path / f"reg{k + 1}.json")
        R.save(reg, f)
        reg_files.append(f)
        base = f
    reg = R.load(reg_files[-1])
    listed = [e for e in R.entries(reg["observations"], now=now) if e["status"] == "LISTED"]
    assert len(listed) == 2  # one per planted coupler on a standard ring; the third coupler was never scanned
    first = {o["n"]: o for o in reg["observations"][:3]}
    for n, e in ((40, planted[0]), (72, planted[1])):
        rep = P.plan(dev, dict(SPEC, n=n), reg, str(tmp_path / f"p{n}"), now)
        assert rep["category"] == "self-checked" and rep["avoided"] and rep["accepted"]
        assert not _on_ring(rep["ring"], e)
        sim.submit(rep["user_job"]["job"], "pauli", seed=2000 + n, planted=planted, p=P_ERR)
        (r,) = J.score(rep["user_job"]["job"], rep["user_job"]["runs"]).values()
        std_usable = n - sum(d["length"] for d in first[n]["dead"])
        print(f"T3 n = {n}: listed {[x['centre'] for x in listed]}; usable standard {std_usable} -> avoiding "
              f"{r['usable']}; masked {rep['masked_couplers']}")
        assert r["usable"] > std_usable, (n, r["usable"], std_usable)
