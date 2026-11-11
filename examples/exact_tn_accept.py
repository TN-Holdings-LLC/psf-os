"""Acceptance of psf_os.exact_tn (2026-11-11, home's instruction of 11-11; criteria fixed before running, all 1e-10):
  1. cards of at most 20 sites (kicked Ising 4x2x2, Heisenberg 4x2x2 textbook and heisE line, Schwinger N 6; merge on
     and off; extra X / Y / two-site observables added): method "tn" against "statevector";
  2. IBM 2023 utility observables on the Eagle graph (Z106 4 steps, weight-10 5 steps, 17 points each; weight-17 5
     steps optional) against TOQB's released answers (results/utility-2026-11-10/answers/answers_ibm_fez.json);
  3. kicked Ising 4x6x3 (72 sites), t 4, theta 0 and pi/2, every site: exact_tn against Aer's stabilizer method;
  plus reach probes (not criteria): 72 sites at theta 0.8, t 3-6, one interior site; each line records the cone, the
  largest intermediate and the time, or why it was refused (memory guard 2^27 elements, cost guard 10^12 flops).

usage: python examples/exact_tn_accept.py TOQB_DIR OUT.jsonl [1 2 3 reach w17]
(workplace: run under `ulimit -v 6500000` on an 8 GB machine, so that memory runs out as MemoryError, not a kill)
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from psf_os import card as C, exact_tn as E, grid3d  # noqa: E402

TOL = 1e-10
HERE = os.path.dirname(os.path.abspath(__file__))


def log(out, **row):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    row["time_cet"] = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%H:%M:%S")
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(json.dumps(row), flush=True)


def check1(out):
    cards = {}
    for f in ("kicked_ising_4x2x2.json", "heisenberg_4x2x2_textbook.json",
              "heisenberg_4x2x2_heisE_line_kingston_c0_t4.json", "schwinger_N6_M1.json"):
        cards[f] = C.load(os.path.join(HERE, "cards", f))
    k3 = copy.deepcopy(cards["kicked_ising_4x2x2.json"])
    k3["time"]["steps"] = 3
    cards["kicked_ising_4x2x2 (steps 3)"] = k3
    worst_all = 0.0
    for name, cd in cards.items():
        cd = copy.deepcopy(cd)
        cd["measure"] = cd["measure"] + [{"op": "XX", "sites": [0, 1]}, {"op": "Y", "sites": [2]},
                                         {"op": "ZY", "sites": [1, 3]}]
        for merge in (False, True):
            t0 = time.time()
            a = C.exact(cd, method="statevector", merge=merge)
            t1 = time.time()
            info = []
            b = C.exact(cd, method="tn", merge=merge, info=info)
            t2 = time.time()
            w = max(abs(a[k] - b[k]) for k in a)
            worst_all = max(worst_all, w)
            log(out, check=1, card=name, merge=merge, observables=len(a), worst=w, ok=w <= TOL,
                statevector_s=round(t1 - t0, 2), tn_s=round(t2 - t1, 2),
                max_log2_intermediate=max(r.get("log2_max_intermediate", 0) for r in info))
    log(out, check=1, summary=True, worst=worst_all, ok=worst_all <= TOL)


def eagle():
    from qiskit_ibm_runtime import fake_provider as fp
    return sorted({tuple(sorted(e)) for e in fp.FakeSherbrooke().configuration().coupling_map})


def utility_circuit(edges, theta, t, extra_rx=False):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(127)
    for _ in range(t):
        for q in range(127):
            qc.rx(theta, q)
        for a, b in edges:
            qc.rzz(-math.pi / 2, a, b)
    if extra_rx:
        for q in range(127):
            qc.rx(theta, q)
    return qc


W10 = {13: "X", 29: "X", 31: "X", 9: "Y", 30: "Y", 8: "Z", 12: "Z", 17: "Z", 28: "Z", 32: "Z"}
W17 = {**{q: "X" for q in (37, 41, 52, 56, 57, 58, 62, 79)}, 75: "Y",
       **{q: "Z" for q in (38, 40, 42, 63, 72, 80, 90, 91)}}


def check2(out, toqb, names=("z106_t4", "w10_t5"), search_time=20.0):
    ans = json.load(open(os.path.join(toqb, "results/utility-2026-11-10/answers/answers_ibm_fez.json")))
    obs = {"z106_t4": ({106: "Z"}, 4), "w10_t5": (W10, 5), "w17_t5": (W17, 5)}
    edges = eagle()
    for name in names:
        o, t = obs[name]
        worst = 0.0
        for k in range(17):
            theta = k * math.pi / 32
            try:
                v, r = E.expectation(utility_circuit(edges, theta, t), o, search_time=search_time, max_cost=1e12)
            except E.TooLarge as err:
                log(out, check=2, obs=name, k=k, refused=err.reason, **err.report)
                worst = float("inf")
                continue
            except MemoryError:
                log(out, check=2, obs=name, k=k, refused="MemoryError (process limit)")
                worst = float("inf")
                continue
            d = abs(v - ans[f"{name}_k{k:02d}"])
            worst = max(worst, d)
            log(out, check=2, obs=name, k=k, diff=d, cone_qubits=r["cone_qubits"],
                tensors=r["tensors"], tensors_simplified=r["tensors_simplified"],
                log2_max_intermediate=r["log2_max_intermediate"], slices=r["slices"], log10_cost=r["log10_cost"],
                search_s=r["search_s"], contract_s=r["contract_s"])
        log(out, check=2, obs=name, summary=True, worst=worst, ok=worst <= TOL)


def check3(out, L=4, M=6, N=3, t=4):
    from qiskit.quantum_info import Clifford, Pauli
    from qiskit_aer import AerSimulator
    n = L * M * N
    for theta in (0.0, math.pi / 2):
        qc = grid3d.logical(L, M, N, t, theta)
        cl = Clifford(qc).to_circuit()
        for q in range(n):
            cl.save_expectation_value(Pauli("I" * (n - 1 - q) + "Z" + "I" * q), list(range(n)), label=f"z{q}")
        t0 = time.time()
        data = AerSimulator(method="stabilizer").run(cl).result().data()
        ref = [float(np.real(data[f"z{q}"])) for q in range(n)]
        t1 = time.time()
        vals, reps = E.z_sites(qc, range(n), info=True)
        t2 = time.time()
        w = max(abs(a - b) for a, b in zip(vals, ref))
        log(out, check=3, lattice=f"{L}x{M}x{N}", t=t, theta=theta, sites=n, worst=w, ok=w <= TOL,
            stabilizer_s=round(t1 - t0, 2), tn_s=round(t2 - t1, 2),
            max_cone=max(r.get("cone_qubits", 0) for r in reps),
            max_log2_intermediate=max(r.get("log2_max_intermediate", 0) for r in reps),
            nonzero_answers=sum(abs(x) > 1e-9 for x in ref))


def check3b(out, L=4, M=6, N=3, t=4, count=6):
    """Beyond the criterion: at theta = pi/2 every <Z> of check 3 is 0, so also take stabilizers of the state (from
    qiskit's Clifford, exact +-1) as Pauli products with large support and compare exact_tn with them."""
    from qiskit.quantum_info import Clifford
    n = L * M * N
    qc = grid3d.logical(L, M, N, t, math.pi / 2)
    labels = Clifford(qc).to_labels(mode="S")
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(labels), count, replace=False).tolist())
    worst = 0.0
    for i in pick:
        lab = labels[i]
        sign = -1.0 if lab[0] == "-" else 1.0
        body = lab.lstrip("+-")
        obs = {n - 1 - k: c for k, c in enumerate(body) if c != "I"}
        t0 = time.time()
        v, r = E.expectation(qc, obs)
        d = abs(v - sign)
        worst = max(worst, d)
        log(out, check="3b", lattice=f"{L}x{M}x{N}", t=t, stabilizer=i, weight=len(obs), expected=sign, diff=d,
            seconds=round(time.time() - t0, 1), log2_max_intermediate=r.get("log2_max_intermediate"))
    log(out, check="3b", summary=True, worst=worst, ok=worst <= TOL)


def reach(out, L=4, M=6, N=3, site=None, steps=(3, 4, 5, 6), theta=0.8):
    site = site if site is not None else ((L // 2) * M + M // 2) * N + N // 2
    for t in steps:
        qc = grid3d.logical(L, M, N, t, theta)
        t0 = time.time()
        try:
            v, r = E.expectation(qc, {site: "Z"}, search_time=30.0, max_cost=1e12)
            log(out, check="reach", lattice=f"{L}x{M}x{N}", t=t, theta=theta, site=site, ok=True, seconds=round(time.time() - t0, 1),
                **{k: r[k] for k in ("cone_qubits", "cone_gates", "tensors", "tensors_simplified",
                                     "log2_max_intermediate", "slices", "log10_cost", "search_s", "contract_s")})
        except E.TooLarge as err:
            log(out, check="reach", lattice=f"{L}x{M}x{N}", t=t, theta=theta, site=site, ok=False, refused=err.reason,
                seconds=round(time.time() - t0, 1), **err.report)
        except MemoryError:
            log(out, check="reach", lattice=f"{L}x{M}x{N}", t=t, theta=theta, site=site, ok=False,
                refused="MemoryError (process limit)", seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    toqb, path = sys.argv[1], sys.argv[2]
    which = sys.argv[3:] or ["1", "2", "3", "reach"]
    if os.path.exists(path):
        raise SystemExit(f"STOP: {path} exists")
    with open(path, "w") as out:
        if "1" in which:
            check1(out)
        if "2" in which:
            check2(out, toqb)
        if "3" in which:
            check3(out)
        if "3b" in which:
            check3b(out)
        if "reach" in which:
            reach(out)
        if "w17" in which:
            check2(out, toqb, names=("w17_t5",), search_time=120.0)
