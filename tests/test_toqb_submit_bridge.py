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
