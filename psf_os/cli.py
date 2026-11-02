"""psf-os: one command for the self-checking layer.

    psf-os fetch --backend fake:FakeFez --out DEV_DIR      coupling graph and failed elements only
    psf-os plan --device DEV_DIR --job LOGICAL.json --registry REG.json --out DIR [--now ISO] [--standard]
                                                           stale registry -> a scan job; fresh -> avoid, place, assemble
    psf-os submit --job JOB_DIR --local [--noiseless]      fake devices only here (Aer); an IBM device is not wired in M5
    psf-os scan-ingest --device DEV_DIR --job JOB_DIR --runs RUNS --registry REG.json --out NEW_REG.json --label L
    psf-os status --registry REG.json [--device NAME] [--now ISO]
    psf-os result --job JOB_DIR --runs RUNS                score and logical counts

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
    f.add_argument("--backend", required=True, help="fake:FakeFez etc. (an IBM backend is fetched at home)")
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
    a = p.parse_args(argv)
    now = (dt.datetime.fromisoformat(a.now) if getattr(a, "now", None) else P.now_cet())

    if a.cmd == "fetch":
        kind, _, name = a.backend.partition(":")
        if kind != "fake":
            print("STOP: psf-os does not connect to IBM in M5; fetch a fake device (fake:FakeFez)", file=sys.stderr)
            return 2
        Dv.fetch_fake(name).save(a.out)
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
            print("STOP: only --local (Aer) in M5; IBM submission is not wired", file=sys.stderr)
            return 2
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
