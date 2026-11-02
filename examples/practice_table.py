"""The practice problems on a fake device: size, two-qubit gates, depth, time model and the noiseless score.
Usage: TOQB not needed.  python examples/practice_table.py OUT_DIR   (writes practice_table.json and .md)"""
import json
import os
import sys

from psf_os import circuits as C
from psf_os import device as Dv
from psf_os import job as J
from psf_os import plan as P
from psf_os import registry as R
from psf_os import sim

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
dev = Dv.fetch_fake("FakeFez")
now = P.now_cet()
PROBS = [("P1", dict(kind="clifford_random", n=20, k=4, seed=1)), ("P1", dict(kind="clifford_random", n=40, k=8, seed=2)),
         ("P1", dict(kind="clifford_random", n=72, k=8, seed=3)), ("P2", dict(kind="cluster_parity", n=20)),
         ("P2", dict(kind="cluster_parity", n=40)), ("P2", dict(kind="cluster_parity", n=72)),
         ("P3", dict(kind="kising_forward", n=20, L=4)), ("P3", dict(kind="kising_forward", n=20, L=8)),
         ("P4", dict(kind="sh6", n=40, D=12)), ("P5", dict(kind="echo", n=20, D=24))]
rows = []
for k, (pid, job) in enumerate(PROBS):
    job = dict(job, shots=1000)
    rep = P.plan(dev, job, R.empty(), os.path.join(out, f"p{k}"), now, standard=True)
    runs = rep["user_job"]["runs"]
    rec = json.load(open(os.path.join(runs, [f for f in os.listdir(runs) if f.endswith(".json")][0])))
    from qiskit import qpy
    circ = qpy.load(open(os.path.join(runs, rec["export"]["qpy"]), "rb"))[0]
    row = dict(problem=pid, job={k_: v for k_, v in job.items() if k_ != "shots"}, q2=rec["echo"]["q2_circuits"][0],
               depth=circ.depth(), circuits=rep["user_job"]["circuits"], qpu_s=rep["user_job"]["qpu_seconds_model"])
    if rec.get("answers") and job["kind"] == "clifford_random":
        row["determined_sites"] = sum(1 for v in rec["answers"].values() for x in v if abs(abs(x) - 1) < 1e-9)
    try:
        sim.submit(rep["user_job"]["job"], "noiseless")
        (r,) = J.score(rep["user_job"]["job"], runs).values()
        row["noiseless"] = {k_: r[k_] for k_ in ("right_on_determined", "size_on_undetermined", "mse", "right", "per_bit",
                                                  "rho4") if k_ in r}
    except ValueError as e:
        row["noiseless"] = f"not simulated: {str(e)[:80]}"
    rows.append(row)
    print(row)
json.dump(rows, open(os.path.join(out, "practice_table.json"), "w"), indent=1)
