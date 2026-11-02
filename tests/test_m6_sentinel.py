"""M6 prototype: sentinels (S1-S6). A sentinel is the rule's echo on the user's ring, in the user's job; its verdict
uses the registry's rules unchanged, and what it finds enters a new registry file as gray entries of that day."""
import json
import os

import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from psf_os import device as Dv  # noqa: E402
from psf_os import plan as P  # noqa: E402
from psf_os import registry as R  # noqa: E402
from psf_os import sentinel as Se  # noqa: E402
from psf_os import sim  # noqa: E402

CLIFF = dict(kind="echo_clifford", D=24)


def fresh(dev, now):
    """A registry with a clean scan observation of the day (so plan does not ask for a scan first)."""
    reg = R.empty()
    reg["observations"] = [dict(device=dev.name, job="morning", label="morning", submitted=now.isoformat(sep=" "),
                                n=20, ring=[], median=0.4, dead=[])]
    return reg


def test_s1_sentinel_leaves_the_user_circuits_unchanged(tmp_path):
    dev, now = Dv.fetch_fake("FakeFez"), P.now_cet()
    job = dict(kind="sh6", n=40, D=12, shots=100)
    a = P.plan(dev, job, fresh(dev, now), str(tmp_path / "with"), now)
    b = P.plan(dev, job, fresh(dev, now), str(tmp_path / "without"), now, sentinels=False)
    ra = json.load(open(os.path.join(a["user_job"]["runs"], "kqsh_n40_D12-fold_rule.json")))
    rb = json.load(open(os.path.join(b["user_job"]["runs"], "kqsh_n40_D12-fold_rule.json")))
    assert ra["export"]["sha256"] == rb["export"]["sha256"]
    ma = json.load(open(os.path.join(a["user_job"]["job"], "manifest.json")))
    mb = json.load(open(os.path.join(b["user_job"]["job"], "manifest.json")))
    assert ma["user_pubs"] == mb["user_pubs"] == list(range(8))
    assert ma["roles"] == {"kqsh_n40_D12-fold_rule": "user", "ksent_n40_D24-fold_rule": "sentinel"}
    assert a["sentinels"]["circuits"] == 4 and b["sentinels"] is None
    # the user's circuits are the first 8 of the job, byte for byte
    from qiskit import qpy
    ca = qpy.load(open(os.path.join(a["user_job"]["job"], "circuits.qpy"), "rb"))
    cb = qpy.load(open(os.path.join(b["user_job"]["job"], "circuits.qpy"), "rb"))
    assert len(ca) == 12 and len(cb) == 8 and all(x == y for x, y in zip(ca[:8], cb))


def test_s2_clean_without_noise(tmp_path):
    dev, now = Dv.fetch_fake("FakeFez"), P.now_cet()
    rep = P.plan(dev, dict(kind="sh6", n=20, D=6, shots=200), fresh(dev, now), str(tmp_path / "p"), now)
    sim.submit(rep["user_job"]["job"], "noiseless")
    v = Se.verdict(rep["user_job"]["job"], rep["user_job"]["runs"])
    assert v["status"] == "CLEAN"


def _planted_run(tmp_path, now):
    dev = Dv.fetch_fake("FakeFez")
    rep = P.plan(dev, dict(kind="sh6", n=40, D=12, shots=1000), fresh(dev, now), str(tmp_path / "p"), now,
                 sentinel_spec=CLIFF)
    ring = rep["ring"]
    planted = [(ring[20], ring[21])]  # planted after placement; psf-os is not told
    sim.submit(rep["user_job"]["job"], "pauli", planted=planted, p=0.15)
    return dev, rep, planted


def test_s3_planted_defect_is_flagged_where_it_was_planted(tmp_path):
    now = P.now_cet()
    dev, rep, planted = _planted_run(tmp_path, now)
    v = Se.verdict(rep["user_job"]["job"], rep["user_job"]["runs"])
    assert v["status"] in ("WEAK", "DEAD")
    (c,) = v["cells"].values()
    runs = c["dead"] + c["weak"]
    assert any(set(planted[0]) <= set(r["qubits"]) for r in runs)
    print(f"S3: {v['status']}; runs {[(r['positions'][0], len(r['positions'])) for r in runs]}; planted {planted}")


def test_s4_sentinel_findings_are_gray_for_the_day_only(tmp_path):
    import datetime as dt
    now = P.now_cet()
    dev, rep, planted = _planted_run(tmp_path, now)
    reg = fresh(dev, now)
    new = Se.to_registry(reg, dev.name, rep["user_job"]["job"], rep["user_job"]["runs"])
    f = str(tmp_path / "reg_after.json")
    R.save(new, f)
    new = R.load(f)
    listed = lambda r: [e for e in R.entries(r["observations"], now=now) if e["status"] == "LISTED"]  # noqa: E731
    assert listed(new) == listed(reg) == []
    assert new["observations"] == reg["observations"]
    gray = R.avoid_paths(new, dev.name, now)
    assert gray and all(g["kind"] == "gray" for g in gray)
    assert R.avoid_paths(new, dev.name, now + dt.timedelta(days=1)) == []


def test_s5_time_model_cap_stops_the_assembly(tmp_path):
    dev, now = Dv.fetch_fake("FakeFez"), P.now_cet()
    job = dict(kind="sh6", n=20, D=6, shots=20000)  # 8 user circuits: 46.5 s; with the 4 sentinel circuits: 68.2 s
    ok = P.plan(dev, job, fresh(dev, now), str(tmp_path / "a"), now, sentinels=False)
    assert ok["user_job"]["qpu_seconds_model"] <= 60
    with pytest.raises(RuntimeError, match="time model"):
        P.plan(dev, job, fresh(dev, now), str(tmp_path / "b"), now)


def test_s6_sentinel_chain_reads_no_map(tmp_path):
    from qiskit_ibm_runtime import fake_provider
    from test_t6_no_map import Guarded
    b = fake_provider.FakeFez()
    dev = Dv.from_target(Guarded(b.target), b.name, fake=True)
    now = P.now_cet()
    rep = P.plan(dev, dict(kind="sh6", n=40, D=12, shots=100), fresh(dev, now), str(tmp_path / "p"), now)
    assert rep["sentinels"]["circuits"] == 4 and rep["accepted"]
