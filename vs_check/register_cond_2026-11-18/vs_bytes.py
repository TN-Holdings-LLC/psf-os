"""Single-bit builds, byte for byte, under two psf-os checkouts (run once per checkout, then compare). Builds: the
notebook's phase-estimation parts A and B (one-bit conditionals: IPE; none: textbook), a feed-forward job of
teleport_test items, and a bundle of teleport_test items. Writes OUT/<name>/job/...

  python vs_bytes.py build NOTEBOOK DEVDIR OUT        (with PYTHONPATH on the psf-os checkout under test)
  python vs_bytes.py compare OUT_A OUT_B
"""
import hashlib
import json
import os
import subprocess
import sys


def build(nb, devdir, out):
    from psf_os import feedforward as FF
    from psf_os.device import Device
    os.makedirs(out, exist_ok=True)
    tool = os.path.join(nb, "gpu", "qpe_repro_night.py")
    for part in ("A", "B"):
        subprocess.run([sys.executable, tool, "build", devdir, os.path.join(out, f"qpe_{part}"), "--part", part],
                       check=True, capture_output=True)
    dev = Device.load(devdir)
    items = [(f"tp{k}", FF.teleport_test(k), dict(success={k: 0})) for k in (1, 2, 3)]
    FF.build(dev, items, os.path.join(out, "teleport"), shots=1000, sentinel_spec=dict(kind="echo_clifford", D=24))
    FF.build_bundle(dev, items[:2], os.path.join(out, "teleport_bundle"), shots=1000,
                    sentinel_spec=dict(kind="echo_clifford", D=24))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def compare(a, b):
    ok = True
    for name in sorted(os.listdir(a)):
        qa, qb = (os.path.join(x, name, "job", "circuits.qpy") for x in (a, b))
        ma, mb = (json.load(open(os.path.join(x, name, "job", "manifest.json"))) for x in (a, b))
        same_qpy = sha(qa) == sha(qb)
        extra = sorted(set(mb) - set(ma))
        drop = {"written"} | set(extra)
        same_man = {k: v for k, v in ma.items() if k not in drop} == {k: v for k, v in mb.items() if k not in drop}
        ok &= same_qpy and same_man
        print(f"{name:16s} circuits.qpy {'same bytes' if same_qpy else 'DIFFER'} ({sha(qa)[:12]}); manifest apart from "
              f"'written' {'same' if same_man else 'DIFFERS'}; fields only in the second: {extra or '-'}")
    print("ALL SAME" if ok else "SOME DIFFER")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:5])
    else:
        sys.exit(0 if compare(*sys.argv[2:4]) else 1)
