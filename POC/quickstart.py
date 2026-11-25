"""Five-minute tour of psf-os 0.3.3 on a fake device (no IBM account needed).

SIMULATION ONLY. The defects below are planted by this script into an Aer simulation of IBM's FakeFez; psf-os is
not told where they are. Nothing printed here is a device result. The device results are in Papers 5 to 11 and in the
psf-os-lab-notebook records (see POC/README.md).

Steps 1 to 4 are psf-zero's POC/quickstart_os.py of 2026-11-02, unchanged in what they compute; steps 5 and 6 show two
defaults that arrived later (0.3.0 and 0.3.3). It uses psf-os's own functions only (no new rule or threshold):
  1. reads FakeFez's coupling graph and failed elements (no quality number);
  2. plants a two-qubit depolarizing error (probability 0.15 after every CZ) on two couplers, one on the standard
     n = 40 ring and one on the standard n = 72 ring;
  3. scans twice (the mirror echo per ring site; at its Clifford point h = pi/4 so that a laptop can simulate 72
     qubits; the device scan uses h = 0.6), and ingests both scans into the registry: the planted paths become LISTED;
  4. solves two practice problems with exact answers, P2 (cluster-state parity) and P1 (random Clifford circuit read
     through output stabilizers), at n = 40 on the standard ring and on the ring placed around the registry, with the
     same planted errors, plus the sentinel on the avoiding ring, and prints the scores side by side;
  5. parking (default since 0.3.0 for a qubit in |1> on every branch): a three-qubit circuit whose third qubit holds
     |1> through a mid-circuit read and a feed-forward; footprint.park_waits puts it in |0> for the wait (an X pair
     behind fences) and checks that the exact distribution of the classical bits is unchanged. No noise is simulated
     here: what parking keeps on a device is in Paper 11;
  6. pairing (default on the device since 0.3.3): loop.start refuses a device night that would decide across jobs
     unless the caller says so (cross_job=True). Nothing is built or sent;
  7. writes OUT_DIR/summary.json and prints the time taken.

Usage:  pip install -e "psf-os[sim]"   then   python psf-os/POC/quickstart.py [OUT_DIR]
"""
import json
import os
import sys
import time

from qiskit import QuantumCircuit

import psf_os
from psf_os import device as Dv
from psf_os import footprint as F
from psf_os import job as J
from psf_os import loop as L
from psf_os import plan as P
from psf_os import registry as R
from psf_os import sentinel as Se
from psf_os import sim

BANNER = "SIMULATION with planted defects on FakeFez -- not a device result"
SCAN = dict(kind="echo_clifford", D=24, shots=1000)        # the scan at its Clifford point (simulable at n = 72)
SENTINEL = dict(kind="echo_clifford", D=24)
PLANT = dict(p=0.15, kind="depol2")
PROBLEMS = [("P2 cluster parity, n = 40", dict(kind="cluster_parity", n=40, shots=1000), "right"),
            ("P1 random Clifford (k = 4), read stabilizers, n = 40", dict(kind="clifford_stab", n=40, k=4, seed=1,
                                                                            shots=1000), "right")]


def rings_of(runs):
    out = {}
    for f in sorted(x for x in os.listdir(runs) if x.endswith(".json")):
        r = json.load(open(os.path.join(runs, f)))
        out[r["task"]["n"]] = [int(r["echo"]["measured_qubits"][str(j)]) for j in range(r["task"]["n"])]
    return out


def main(out):
    t0 = time.time()
    os.makedirs(out, exist_ok=False)
    print(BANNER)
    dev = Dv.fetch_fake("FakeFez")
    now = P.now_cet()
    print(f"1. device record: {dev.name}, {len(dev.couplers)} couplers, {len(dev.failed_couplers)} failed, "
          f"{len(dev.failed_qubits)} failed qubits (no quality number read)")

    s1 = P.scan_job(dev, os.path.join(out, "scan1"), spec=SCAN)
    rings = rings_of(s1["runs"])
    planted = [(rings[40][12], rings[40][13]), (rings[72][50], rings[72][51])]
    print(f"2. planted (known to this script only): couplers {planted}, two-qubit depolarizing 0.15")

    base = None
    for k, s in enumerate([s1, P.scan_job(dev, os.path.join(out, "scan2"), spec=SCAN)]):
        sim.submit(s["job"], "pauli", seed=1000 + k, planted=planted, **PLANT)
        reg, obs = P.scan_ingest(dev.name, s["job"], s["runs"], base, f"scan{k + 1}")
        base = os.path.join(out, f"registry_{k + 1}.json")
        R.save(reg, base)
    reg = R.load(base)
    listed = [e["centre"] for e in R.entries(reg["observations"], now=now) if e["status"] == "LISTED"]
    print(f"3. two scans ({s1['circuits']} circuits each); LISTED paths: {listed}")

    print("4. practice problems (exact answers recorded at build time), standard ring against avoiding ring:")
    rows = []
    for label, job, key in PROBLEMS:
        std = P.plan(dev, job, R.empty(), os.path.join(out, f"std_{job['kind']}"), now, standard=True)
        sim.submit(std["user_job"]["job"], "pauli", seed=2000, planted=planted, **PLANT)
        (rs,) = J.score(std["user_job"]["job"], std["user_job"]["runs"]).values()
        av = P.plan(dev, job, reg, os.path.join(out, f"avoid_{job['kind']}"), now, sentinel_spec=SENTINEL)
        sim.submit(av["user_job"]["job"], "pauli", seed=2000, planted=planted, **PLANT)
        (ra,) = J.score(av["user_job"]["job"], av["user_job"]["runs"]).values()
        sv = Se.verdict(av["user_job"]["job"], av["user_job"]["runs"])
        rows.append(dict(problem=label, standard=rs[key], avoiding=ra[key], sentinel=sv["status"],
                         masked=av["masked_couplers"], qpu_seconds_model=av["user_job"]["qpu_seconds_model"]))
        print(f"   {label}: fraction right {rs[key]:.3f} (standard ring) -> {ra[key]:.3f} (avoiding ring); "
              f"sentinel on the avoiding ring: {sv['status']}; masked couplers {av['masked_couplers']}")
    park = parking_example()
    print(f"5. parking (0.3.0 and later): {park['waits']} wait, qubit(s) parked {park['parked']}; waiting qubits "
          f"in |1> (expected number) {park['excited_before']:.0f} -> {park['excited_after']:.0f}; exact distribution unchanged: "
          f"{park['unchanged']}")
    pair = pairing_example(out)
    print(f"6. pairing (0.3.3 default on the device): a device night deciding across jobs is {pair}")
    dt = time.time() - t0
    json.dump(dict(banner=BANNER, psf_os=psf_os.__version__, planted=planted, listed=listed, rows=rows,
                   parking=park, pairing=pair, seconds=round(dt, 1)),
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(f"7. done in {dt:.0f} s on this machine (psf-os {psf_os.__version__}). {BANNER}.")


def parking_example():
    """A qubit in |1> that waits through a mid-circuit read and a feed-forward, parked by psf-os's own park_waits."""
    qc = QuantumCircuit(3, 2)
    qc.x(2)                            # qubit 2 holds |1> through the wait
    qc.h(0)
    qc.measure(0, 0)                   # the mid-circuit read
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)                        # the feed-forward
    qc.measure(2, 1)
    parked, info = F.park_waits(qc)    # raises ValueError if parking changed the noiseless distribution
    d0, d1 = F.exact_distribution(qc), F.exact_distribution(parked)
    same = all(abs(d0.get(k, 0.0) - d1.get(k, 0.0)) < 1e-9 for k in set(d0) | set(d1))
    return dict(waits=info["waits"], parked=[q for w in info["parked"] for q in w["qubits"]],
                excited_before=info["excited_before"], excited_after=info["excited_after"], unchanged=same)


def pairing_example(out):
    """loop.start's own refusal of a device night that decides across jobs (nothing is built: it stops first)."""
    try:
        L.start(os.path.join(out, "night_not_built"), None, paired=False)
    except SystemExit as e:
        return f"REFUSED ({e})"
    return "NOT REFUSED"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "quickstart_out")
