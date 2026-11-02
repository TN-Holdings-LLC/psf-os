"""T2: psf-os places the rule's ring on the same physical qubits as TOQB, with and without a mask.

For three fake devices and n = 20, 40, 72: TOQB's procedure (s2rc_device fetch --fake, then kising_echo build
--arm fold:rule; with a mask, defect_list's mask_target on the fetched Target first) against psf-os (device record,
allocate.mask, RuleCompiler). Compared: the measured qubit of every ring site of the D = 24 echo, and the two-qubit
gate count. The masked path is made up for the test (four consecutive sites of the unmasked ring).
Needs TOQB (TOQB_DIR) and qiskit-ibm-runtime; slow (one TOQB build per case).
"""
import json
import os
import pickle
import subprocess
import sys

import pytest

TQ = os.environ.get("TOQB_DIR", "")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(TQ, "toqb")), reason="TOQB_DIR not set")

FAKES = ["FakeKingston", "FakeMarrakesh", "FakeFez"]
SIZES = [20, 40, 72]


def _toqb(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=TQ, PYTHONWARNINGS="ignore")
    p = subprocess.run([sys.executable, "-m", *args], cwd=TQ, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def _toqb_ring(fetch, runs, n):
    code, out = _toqb("toqb.kising_echo", "build", fetch, runs, "--arm", "fold:rule", "--n", str(n), "--L", "12",
                      "--hz", "0", "--shots", "1000")
    name = f"kecho_n{n}_D24_hz0_h0.6-fold_rule.json"
    if code != 0 or not os.path.exists(os.path.join(runs, name)):
        return None, out
    r = json.load(open(os.path.join(runs, name)))
    mq = r["echo"]["measured_qubits"]
    return dict(ring=[int(mq[str(j)]) for j in range(n)], q2=r["echo"]["q2_circuits"][0], valid=r["valid"]), out


def _masked_fetch(src, dst, path):
    sys.path.insert(0, TQ)
    from toqb.defect_list import mask_target
    from toqb.kising_charge import sha256_file
    t = pickle.load(open(os.path.join(src, "target.pkl"), "rb"))
    t2, masked = mask_target(t, [tuple(path)])
    os.makedirs(dst)
    pickle.dump(t2, open(os.path.join(dst, "target.pkl"), "wb"))
    fj = json.load(open(os.path.join(src, "fetch.json")))
    fj["target_sha256"] = sha256_file(os.path.join(dst, "target.pkl"))
    json.dump(fj, open(os.path.join(dst, "fetch.json"), "w"))
    return masked


def _psfos_ring(dev, n):
    from psf_os import circuits as C
    from psf_os.compile import RuleCompiler, measured_qubits
    rc = RuleCompiler(dev)
    out = rc.compile(C.echo_circuit([0] * n, n, 12))
    mq = measured_qubits(out)
    return dict(ring=[mq[j] for j in range(n)], q2=C.q2_count(out))


@pytest.fixture(scope="module", params=FAKES)
def fake(request, tmp_path_factory):
    from psf_os.device import fetch_fake
    d = tmp_path_factory.mktemp(request.param)
    code, out = _toqb("toqb.s2rc_device", "fetch", "--fake", request.param, "--out", str(d / "fetch"))
    assert code == 0, out
    return request.param, str(d), fetch_fake(request.param)


@pytest.mark.parametrize("n", SIZES)
def test_same_ring_as_toqb(fake, n):
    from psf_os import allocate as A
    name, d, dev = fake
    want, log = _toqb_ring(os.path.join(d, "fetch"), os.path.join(d, f"runs{n}"), n)
    if want is None:
        with pytest.raises(Exception):
            _psfos_ring(dev, n)
        pytest.skip(f"TOQB places no ring at n = {n} on {name} (psf-os refuses too)")
    got = _psfos_ring(dev, n)
    assert got["ring"] == want["ring"] and got["q2"] == want["q2"]
    # with a mask: a made-up path, four consecutive sites of the unmasked ring
    path = want["ring"][5:9]
    _masked_fetch(os.path.join(d, "fetch"), os.path.join(d, f"fetch_m{n}"), path)
    want_m, log = _toqb_ring(os.path.join(d, f"fetch_m{n}"), os.path.join(d, f"runs_m{n}"), n)
    dev_m, masked = A.mask(dev, [path])
    if want_m is None:
        with pytest.raises(Exception):
            _psfos_ring(dev_m, n)
        return
    got_m = _psfos_ring(dev_m, n)
    assert got_m["ring"] == want_m["ring"] and got_m["q2"] == want_m["q2"]
    assert want_m["ring"] != want["ring"]
