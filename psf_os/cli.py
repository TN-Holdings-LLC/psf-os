"""psf-os: one command for the self-checking layer.

    psf-os fetch --backend ibm_kingston | fake:FakeFez --out DEV_DIR
                                                           coupling graph and failed elements only (IBM: the saved account)
    psf-os plan --device DEV_DIR --job LOGICAL.json --registry REG.json --out DIR [--now ISO] [--standard]
                                                           stale registry -> a scan job; fresh -> avoid, place, assemble
    psf-os submit --job JOB_DIR [--local]                  IBM: the job's own device only ("stay"), options off,
                                                           refused if a circuit touches a failed element; --local: Aer
    psf-os collect --job JOB_DIR [--wait-min M]            IBM: raw_counts.json once the job is done
    psf-os scan-ingest --device DEV_DIR --job JOB_DIR --runs RUNS --registry REG.json --out NEW_REG.json --label L
    psf-os status --registry REG.json [--device NAME] [--now ISO]
    psf-os result --job JOB_DIR --runs RUNS                score and logical counts
    psf-os run2d --device DEV_DIR --circuit FILE.qpy|.qasm [FILE ...] --out DIR [--registry REG.json] [--shots N]
                 [--max-seconds S]                        several files: one job, each with its own floor and mirror
                                                           a 2D circuit on Heron r2 labels: placed by translation,
                                                           floor, mirror and ring sentinel added (psf_os.user2d)
    psf-os result2d --job JOB_DIR --runs RUNS [--answers ANSWERS.json] [--extended]
                                                           the answer card (value +- 2 SE, floor, mirror flags, sentinel);
                                                           --extended: + report-only flags, retention, notes (report2d)
    psf-os ringledger --jobs JOBS.json --out LEDGER.json   the ring sentinel's ledger: which qubits were in DEAD runs,
                                                           job by job (report2d); result2d --extended --ring-ledger L
                                                           --date D also reads the sentinel with them set aside
    psf-os gatehash --qpy FILE.qpy [...]                   a hash of the gate list (name, qubits, angles to 1e-10) that
                                                           does not change with Qiskit's QPY format (psf_os.gatehash)
    psf-os repair2d --device DEV_DIR --ledger LEDGER.json --rule a|b --release expiry|early --out LOG.json
                    [--registry REG.json] [--now ISO]      the self-repair chain for the P6 patch (psf_os.repair2d):
                                                           confirmed exclusions, A's and B's copies, the log

A registry file is never overwritten: scan-ingest writes the next one (TOQB's list_after convention).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

from psf_os import device as Dv
from psf_os import job as J
from psf_os import plan as P
from psf_os import registry as R


def main(argv=None):
    p = argparse.ArgumentParser(prog="psf-os")
    sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch")
    f.add_argument("--backend", required=True, help="ibm_kingston etc. (the saved IBM account), or fake:FakeFez")
    f.add_argument("--out", required=True)
    pl = sp.add_parser("plan")
    pl.add_argument("--device", required=True)
    pl.add_argument("--job", required=True)
    pl.add_argument("--registry", default=None)
    pl.add_argument("--out", required=True)
    pl.add_argument("--now", default=None)
    pl.add_argument("--standard", action="store_true", help="no registry: the standard category")
    pl.add_argument("--no-sentinels", action="store_true", help="leave out the L5 sentinel")
    pl.add_argument("--scan-sizes", default="20,40,72")
    pl.add_argument("--scan-clifford", action="store_true",
                    help="dry runs only: scan with the echo at its Clifford point (h = pi/4) so a local run can simulate it")
    s = sp.add_parser("submit")
    s.add_argument("--job", required=True)
    s.add_argument("--local", action="store_true")
    s.add_argument("--seed", type=int, default=20261102)
    co = sp.add_parser("collect")
    co.add_argument("--job", required=True)
    co.add_argument("--wait-min", type=float, default=0.0)
    si = sp.add_parser("scan-ingest")
    si.add_argument("--device", required=True)
    si.add_argument("--job", required=True)
    si.add_argument("--runs", required=True)
    si.add_argument("--registry", default=None)
    si.add_argument("--out", required=True)
    si.add_argument("--label", required=True)
    st = sp.add_parser("status")
    st.add_argument("--registry", required=True)
    st.add_argument("--device", default=None)
    st.add_argument("--now", default=None)
    r = sp.add_parser("result")
    r.add_argument("--job", required=True)
    r.add_argument("--runs", required=True)
    r2 = sp.add_parser("run2d")
    r2.add_argument("--device", required=True)
    r2.add_argument("--circuit", required=True, nargs="+",
                    help="one or more QPY/QASM files; with several, one job and each file's stem is its tag")
    r2.add_argument("--max-seconds", type=float, default=60)
    r2.add_argument("--out", required=True)
    r2.add_argument("--registry", default=None)
    r2.add_argument("--shots", type=int, default=4000)
    r2.add_argument("--dry-clifford-sentinel", action="store_true",
                    help="local dry runs only: the ring sentinel at the echo's Clifford point, so Aer can run the job")
    g3 = sp.add_parser("run3d", help="prototype: kicked Ising on an L x M x N cubic lattice, fixed-rule placement")
    g3.add_argument("--device", required=True)
    for k in ("L", "M", "N", "t"):
        g3.add_argument(f"--{k}", type=int, required=True)
    g3.add_argument("--theta", type=float, required=True)
    g3.add_argument("--out", required=True)
    g3.add_argument("--shots", type=int, default=4000)
    g3.add_argument("--max-seconds", type=float, default=60)
    g3.add_argument("--answers-out", default=None, help="write the exact answers {g3d: {site: <Z>}} (<= 24 sites)")
    g3.add_argument("--layout", choices=("auto", "fold", "rows", "line"), default="auto",
                    help="auto: fold for L = 4, rows otherwise")
    c2 = sp.add_parser("result2d")
    c2.add_argument("--job", required=True)
    c2.add_argument("--runs", required=True)
    c2.add_argument("--answers", default=None,
                    help="JSON {label: exact <Z>}; for a batch job {tag: {label: exact <Z>}}")
    c2.add_argument("--extended", action="store_true",
                    help="add the report-only blocks of psf_os.report2d (flags, retention, notes); values unchanged")
    c2.add_argument("--ring-ledger", default=None, help="with --extended: a ledger of ringledger; the sentinel is also "
                    "read with the ledger's known-dead qubits set aside (report only)")
    c2.add_argument("--date", default=None, help="with --ring-ledger: the job's day YYYY-MM-DD (only earlier days count)")
    rl = sp.add_parser("ringledger")
    rl.add_argument("--jobs", required=True, help="JSON [[date, job_dir, runs_dir], ...]")
    rl.add_argument("--out", required=True)
    gh = sp.add_parser("gatehash")
    gh.add_argument("--qpy", nargs="+", required=True, help="QPY files (e.g. JOB_DIR/circuits.qpy, RUNS/*.qpy)")
    rp = sp.add_parser("repair2d")
    rp.add_argument("--device", required=True)
    rp.add_argument("--ledger", required=True)
    rp.add_argument("--rule", choices=("a", "b"), required=True)
    rp.add_argument("--release", choices=("expiry", "early"), required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--registry", default=None)
    rp.add_argument("--now", default=None)
    a = p.parse_args(argv)
    now = (dt.datetime.fromisoformat(a.now) if getattr(a, "now", None) else P.now_cet())

    if a.cmd == "fetch":
        kind, _, name = a.backend.partition(":")
        if kind == "fake":
            Dv.fetch_fake(name).save(a.out)
        else:
            from psf_os import ibm
            ibm.fetch(ibm.service(), a.backend, a.out)
        print(f"{a.out}/device.json")
        return 0
    if a.cmd == "plan":
        dev = Dv.Device.load(a.device)
        reg = R.load(a.registry) if a.registry and os.path.exists(a.registry) else R.empty()
        rep = P.plan(dev, json.load(open(a.job)), reg, a.out, now, standard=a.standard,
                     scan_sizes=tuple(int(x) for x in a.scan_sizes.split(",")),
                     scan_spec=dict(kind="echo_clifford", D=24, shots=1000) if a.scan_clifford else None,
                     sentinels=not a.no_sentinels)
        print(json.dumps({k: rep[k] for k in ("category", "why", "avoided", "ring", "next") if k in rep}, indent=1))
        return 0
    if a.cmd == "submit":
        if not a.local:
            from psf_os import ibm
            print(f"submitted job {ibm.submit(ibm.service(), a.job)} (stay); results: psf-os collect --job {a.job}")
            return 0
        man = json.load(open(os.path.join(a.job, "manifest.json")))
        if not man["fetch"].get("fake"):
            print("STOP: a local run is only for a fake device's job", file=sys.stderr)
            return 2
        from psf_os import sim
        try:
            print(sim.submit(a.job, "noiseless", seed=a.seed))
        except ValueError as e:
            print(f"STOP: {e}", file=sys.stderr)
            return 2
        return 0
    if a.cmd == "collect":
        from psf_os import ibm
        st = ibm.collect(ibm.service(), a.job, wait_s=60 * a.wait_min)
        print(f"status {st}" + ("; raw_counts.json written" if "DONE" in st.upper() else "; run collect again later"))
        return 0
    if a.cmd == "scan-ingest":
        dev = Dv.Device.load(a.device)
        reg, obs = P.scan_ingest(dev.name, a.job, a.runs, a.registry, a.label)
        R.save(reg, a.out)
        for o in obs:
            print(f"{o['device']} {o['label']} n {o['n']}: median {o['median']:.2f}; dead runs "
                  + ("; ".join(f"{d['length']} sites, centre {d['centre']}" for d in o["dead"]) or "none"))
        return 0
    if a.cmd == "status":
        print("\n".join(R.status_lines(R.load(a.registry), a.device, now)) or "(empty)")
        return 0
    if a.cmd == "result":
        from psf_os import sentinel
        v = sentinel.verdict(a.job, a.runs)
        print(json.dumps(dict(user=J.score(a.job, a.runs), sentinel=dict(
            status=v["status"], runs={k: c["dead"] + c["weak"] for k, c in v["cells"].items()})), indent=1))
        return 0
    if a.cmd == "run3d":
        from psf_os import grid3d as G
        dev = Dv.Device.load(a.device)
        try:
            rep = G.build(dev, a.L, a.M, a.N, a.t, a.theta, a.out, shots=a.shots, max_seconds=a.max_seconds,
                          layout=a.layout)
        except G.Refused as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        if a.answers_out:
            json.dump({"g3d": G.exact(a.L, a.M, a.N, a.t, a.theta)}, open(a.answers_out, "w"))
        print(json.dumps({k: rep[k] for k in ("device", "layout", "r0", "c0", "sites", "qubits_used", "cz",
                                              "cz_layers", "passes", "circuits", "qpu_seconds_model", "job", "runs")}, indent=1))
        return 0
    if a.cmd == "run2d":
        from psf_os import allocate as A
        from psf_os import user2d as U
        dev = Dv.Device.load(a.device)
        avoid = R.avoid_paths(R.load(a.registry), dev.name, now) if a.registry else []
        if avoid:
            dev, _ = A.mask(dev, avoid, reason=f"registry at {now.isoformat(sep=' ')}")
        spec = dict(kind="echo_clifford", D=24) if a.dry_clifford_sentinel else None
        # time model before anything is written (user2d writes its records before the job is assembled, so a refusal
        # there leaves a half-written --out): each user circuit adds three (itself, floor, mirror), the ring four
        est = J.qpu_seconds(3 * len(a.circuit) + 4, a.shots)
        if est > a.max_seconds + 1e-9:
            per = J.qpu_seconds(1, a.shots) - 3.0
            print(f"REFUSED: {len(a.circuit)} circuit(s) make {3 * len(a.circuit) + 4} circuits at {a.shots} shots: "
                  f"{est:.1f} s by the time model, over --max-seconds {a.max_seconds:g} ({per:.2f} s per circuit + 3 s "
                  f"per job). Use --max-seconds {int(est) + 1}, fewer circuits or fewer shots. Nothing was written.",
                  file=sys.stderr)
            return 2
        try:
            if len(a.circuit) == 1:
                rep = U.build(dev, U.load(a.circuit[0]), a.out, shots=a.shots, sentinel_spec=spec,
                              max_seconds=a.max_seconds)
                show = {k: rep[k] for k in ("device", "copy", "cz", "circuits", "qpu_seconds_model", "job", "runs")}
            else:
                items = [(os.path.splitext(os.path.basename(f))[0], U.load(f)) for f in a.circuit]
                rep = U.build_batch(dev, items, a.out, shots=a.shots, sentinel_spec=spec, max_seconds=a.max_seconds)
                show = dict(device=rep["device"], circuits=rep["circuits"], qpu_seconds_model=rep["qpu_seconds_model"],
                            job=rep["job"], runs=rep["runs"],
                            items={t: dict(copy=r["copy"], cz=r["cz"]) for t, r in rep["items"].items()})
        except U.Refused as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(json.dumps(show, indent=1))
        return 0
    if a.cmd == "repair2d":
        from psf_os import allocate as A
        from psf_os import ising2d as I
        from psf_os import ledger as Lg
        from psf_os import repair2d as Rp
        dev = Dv.Device.load(a.device)
        avoid = R.avoid_paths(R.load(a.registry), dev.name, now) if a.registry else []
        if avoid:
            dev, _ = A.mask(dev, avoid, reason=f"registry at {now.isoformat(sep=' ')}")
        log = Rp.chain(dev, Lg.load(a.ledger), now, I.EDGES, I.QUBITS, rule=a.rule, release=a.release,
                       offsets=I.OFFSETS)
        log["ledger_sha256"] = R.sha256_file(a.ledger)
        log["registry_sha256"] = R.sha256_file(a.registry) if a.registry else None
        Rp.save_log(log, a.out)
        print(json.dumps({k: log[k] for k in ("device", "rule", "release", "excluded", "status", "price")}, indent=1))
        return 0
    if a.cmd == "result2d":
        from psf_os import user2d as U
        ans = json.load(open(a.answers)) if a.answers else None
        if a.extended:
            from psf_os import report2d as X
            if len(U.tags_of(a.job, a.runs)) == 1:
                res = X.extended(a.job, a.runs, ans)
            else:
                res = X.extended_cards(a.job, a.runs, ans)
            if a.ring_ledger:
                if not a.date:
                    print("--ring-ledger needs --date (the job's day)", file=sys.stderr)
                    return 2
                side = X.sentinel_beside_ledger(a.job, a.runs, json.load(open(a.ring_ledger)), a.date)
                res = dict(cards=res, sentinel_beside_ledger=side) if len(U.tags_of(a.job, a.runs)) > 1 else \
                    dict(res, sentinel_beside_ledger=side)
            print(json.dumps(res, indent=1))
            return 0
        if len(U.tags_of(a.job, a.runs)) == 1:
            print(json.dumps(U.card(a.job, a.runs, ans), indent=1))
        else:
            print(json.dumps(U.cards(a.job, a.runs, ans), indent=1))
        return 0
    if a.cmd == "ringledger":
        from psf_os import report2d as X
        if os.path.exists(a.out):
            print(f"{a.out} exists: a ledger is never overwritten", file=sys.stderr)
            return 2
        led = X.ring_ledger([tuple(j) for j in json.load(open(a.jobs))])
        json.dump(led, open(a.out, "w"), indent=1)
        print(json.dumps({k: (len(v) if k != "_jobs" else {d: len(x) for d, x in v.items()}) for k, v in led.items()}))
        return 0
    if a.cmd == "gatehash":
        from psf_os import gatehash as G
        print(json.dumps({f: G.qpy_hash(f) for f in a.qpy}, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
