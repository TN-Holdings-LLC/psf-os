"""Practice problems (Q1-Q4): computations whose answers are not trivial, each answer exact and recorded.
P1 clifford_stab (fixed on 2 November: the read stabilizers of the random Clifford circuit; clifford_random, read
site by site, is kept but cannot see a defect), P2 cluster_parity (the owner's choice in place of GHZ), P3 kising_forward; P4 sh6 and P5 echo are
the existing kinds (tests T4)."""
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

IN = ["zero", "one", "neel", "antineel"]


def _max_diff(a, b):
    return max(abs(x - y) for i in a for x, y in zip(a[i], b[i]))


def test_q1_answers_agree_by_two_independent_methods():
    for k, seed in ((4, 1), (8, 2)):
        j = dict(kind="clifford_random", n=12, k=k, seed=seed)
        assert _max_diff(J.exact_answers(j, IN), J.exact_answers(j, IN, method="statevector")) < 1e-9
    for k, seed in ((4, 1), (8, 2), (6, 7)):
        j = dict(kind="clifford_stab", n=12, k=k, seed=seed)
        an = J.exact_answers(j, IN, method="analytic")  # sign of U Z_j U^dagger x (-1)^(input bit j), from the tableau
        assert all(abs(abs(x) - 1) < 1e-12 for i in an for x in an[i])
        assert _max_diff(an, J.exact_answers(j, IN)) < 1e-12
        assert _max_diff(an, J.exact_answers(j, IN, method="statevector")) < 1e-9
    j = dict(kind="cluster_parity", n=12)
    assert _max_diff(J.exact_answers(j, IN, method="analytic"), J.exact_answers(j, IN, method="statevector")) < 1e-9
    assert _max_diff(J.exact_answers(j, IN, method="analytic"), J.exact_answers(j, IN)) < 1e-12
    for L in (4, 8):
        j = dict(kind="kising_forward", n=8, L=L)
        assert _max_diff(J.exact_answers(j, IN), J.exact_answers(j, IN, method="density")) < 1e-9


def _solve(dev, job, out, standard=True, reg=None, now=None, mode="noiseless", **kw):
    now = now or P.now_cet()
    rep = P.plan(dev, job, reg or R.empty(), out, now, standard=standard,
                 sentinel_spec=dict(kind="echo_clifford", D=24))
    sim.submit(rep["user_job"]["job"], mode, **kw)
    (r,) = J.score(rep["user_job"]["job"], rep["user_job"]["runs"]).values()
    return rep, r


@pytest.mark.parametrize("job", [dict(kind="clifford_stab", n=20, k=4, seed=1),
                                 dict(kind="clifford_stab", n=40, k=8, seed=2),
                                 dict(kind="clifford_random", n=20, k=4, seed=1),
                                 dict(kind="clifford_random", n=40, k=8, seed=2),
                                 dict(kind="cluster_parity", n=20),
                                 dict(kind="kising_forward", n=20, L=4),
                                 dict(kind="kising_forward", n=20, L=8)], ids=lambda j: f"{j['kind']}-{j['n']}")
def test_q2_full_score_without_noise(tmp_path, job):
    dev = Dv.fetch_fake("FakeFez")
    rep, r = _solve(dev, dict(job, shots=2000), str(tmp_path / "p"))
    assert rep["accepted"]
    if job["kind"] == "clifford_stab":
        assert r["signal"] == 1.0 and r["right"] == 1.0 and len(r["observables"]) >= 2
    elif job["kind"] == "clifford_random":
        # most sites have answer 0 (an entangled output); the few determined ones must all be right
        assert r["right_on_determined"] in (None, 1.0) and r["size_on_undetermined"] < 0.1
    elif job["kind"] == "cluster_parity":
        assert r["right"] == 1.0
    else:
        assert r["mse"] < 0.005  # shot noise only: about 1 / shots per site


def test_q3_planted_defects_standard_against_avoiding(tmp_path):
    """As T3, with one change: the planted error is two-qubit depolarizing (probability 0.15 after each CZ on three
    planted couplers) instead of ZZ, because a ZZ error keeps the parity that P2 reads and so cannot test it. Two
    Clifford-point scans list the planted couplers on standard rings; each problem is then solved on the standard ring
    (standard category) and on the ring around the registry, with the same planted error."""
    dev, now = Dv.fetch_fake("FakeFez"), P.now_cet()
    spec = dict(kind="echo_clifford", D=24, shots=1000)
    s1 = P.scan_job(dev, str(tmp_path / "s1"), spec=spec)
    rings = {}
    for f in sorted(x for x in os.listdir(s1["runs"]) if x.endswith(".json")):
        r = json.load(open(os.path.join(s1["runs"], f)))
        rings[r["task"]["n"]] = [int(r["echo"]["measured_qubits"][str(j)]) for j in range(r["task"]["n"])]
    planted = [(rings[40][12], rings[40][13]), (rings[72][50], rings[72][51]), (100, 101)]
    base = None
    for k, s in enumerate([s1, P.scan_job(dev, str(tmp_path / "s2"), spec=spec)]):
        sim.submit(s["job"], "pauli", seed=1000 + k, planted=planted, p=0.15, kind="depol2")
        reg, _ = P.scan_ingest(dev.name, s["job"], s["runs"], base, f"scan{k + 1}")
        base = str(tmp_path / f"reg{k + 1}.json")
        R.save(reg, base)
    reg = R.load(base)
    rows = {}
    for job in (dict(kind="clifford_random", n=40, k=4, seed=1, shots=1000), dict(kind="cluster_parity", n=40, shots=1000),
                dict(kind="clifford_stab", n=40, k=4, seed=1, shots=1000)):
        _, std = _solve(dev, job, str(tmp_path / f"std_{job['kind']}"), mode="pauli", planted=planted, p=0.15,
                        kind="depol2")
        _, av = _solve(dev, job, str(tmp_path / f"av_{job['kind']}"), standard=False, reg=reg, now=now, mode="pauli",
                       planted=planted, p=0.15, kind="depol2")
        if job["kind"] == "clifford_stab":
            rows[job["kind"]] = dict(signal=(std["signal"], av["signal"]), weights=std["weights"])
        elif job["kind"] == "clifford_random":
            rows[job["kind"]] = dict(right_on_determined=(std["right_on_determined"], av["right_on_determined"]),
                                     n_determined=std["n_determined"],
                                     size_on_undetermined=(std["size_on_undetermined"], av["size_on_undetermined"]),
                                     mse=(std["mse"], av["mse"]))
        else:
            rows[job["kind"]] = dict(right=(std["right"], av["right"]))
    print("Q3:", json.dumps(rows))
    # P2 must improve. P1 is recorded, not asserted: with 3 determined sites of 40 (k = 4, seed 1) and every other
    # answer 0, its score cannot see a defect (noise also gives <Z> = 0 on undetermined sites); see the report.
    assert rows["cluster_parity"]["right"][1] > rows["cluster_parity"]["right"][0]
    # P1 fixed (read stabilizers, every answer +-1) sees the defect: 0.964 -> 1.000 at home on 2 November
    assert rows["clifford_stab"]["signal"][1] > rows["clifford_stab"]["signal"][0]


def test_q4_problems_read_no_map(tmp_path):
    from qiskit_ibm_runtime import fake_provider
    from test_t6_no_map import Guarded
    b = fake_provider.FakeFez()
    dev = Dv.from_target(Guarded(b.target), b.name, fake=True)
    now = P.now_cet()
    for k, job in enumerate((dict(kind="clifford_random", n=40, k=4, seed=1), dict(kind="clifford_stab", n=40, k=4, seed=1),
                             dict(kind="cluster_parity", n=72),
                             dict(kind="kising_forward", n=20, L=4))):
        rep = P.plan(dev, dict(job, shots=100), R.empty(), str(tmp_path / f"p{k}"), now, standard=True)
        assert rep["accepted"]
