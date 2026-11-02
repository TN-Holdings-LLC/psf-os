"""IBM submission goes through TOQB's `s2rc_device submit` for now (owner, 2026-11-02): a psf-os job folder must be
readable by it. Checked with its --local mode on a fake device's job (no IBM connection), and psf-os reads the
counts it writes."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")
TQ = os.environ.get("TOQB_DIR", "")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(TQ, "toqb")), reason="TOQB_DIR not set")

from psf_os import device as Dv  # noqa: E402
from psf_os import job as J  # noqa: E402
from psf_os import plan as P  # noqa: E402
from psf_os import registry as R  # noqa: E402


def test_toqb_submit_local_runs_a_psfos_job(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    rep = P.plan(dev, dict(kind="sh6", n=20, D=6, shots=100), R.empty(), str(tmp_path / "p"), P.now_cet(),
                 standard=True)
    env = dict(os.environ, PYTHONPATH=TQ, PYTHONWARNINGS="ignore")
    p = subprocess.run([sys.executable, "-m", "toqb.s2rc_device", "submit", "--job", rep["user_job"]["job"],
                        "--local", "--noiseless"], cwd=TQ, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    (r,) = J.score(rep["user_job"]["job"], rep["user_job"]["runs"]).values()
    assert r["per_bit"] == 1.0


def test_psfos_manifest_makes_toqb_submit_stay(tmp_path):
    """Every psf-os job says "stay": true, and TOQB's submit reads it as --stay (owner, 2026-11-02): a job placed
    around one device's registry is never moved to another device, even if the flag is forgotten."""
    import json
    dev = Dv.fetch_fake("FakeFez")
    rep = P.plan(dev, dict(kind="sh6", n=20, D=6, shots=100), R.empty(), str(tmp_path / "p"), P.now_cet(),
                 standard=True)
    man = json.load(open(os.path.join(rep["user_job"]["job"], "manifest.json")))
    assert man["stay"] is True
    sys.path.insert(0, TQ)
    try:
        from toqb import s2rc_device as SD
        assert SD.job_stays(False, man)
    finally:
        sys.path.remove(TQ)
