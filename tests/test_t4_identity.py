"""T4: the forward identity computation of OS plan M3's second form (sh6: (S, CZ ring)(H, CZ ring) repeated, the
identity every 6 layers), entered through the L6 job interface, returns every input bit without noise; so does the
mirror echo (output = input flipped)."""
import json

import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from psf_os import device as Dv  # noqa: E402
from psf_os import job as J  # noqa: E402
from psf_os import plan as P  # noqa: E402
from psf_os import registry as R  # noqa: E402
from psf_os import sim  # noqa: E402

NOW = P.now_cet()


@pytest.mark.parametrize("fake,n,D", [("FakeFez", 40, 12), ("FakeMarrakesh", 72, 6), ("FakeKingston", 20, 24)])
def test_sh6_returns_every_bit(tmp_path, fake, n, D):
    dev = Dv.fetch_fake(fake)
    rep = P.plan(dev, dict(kind="sh6", n=n, D=D, shots=200), R.empty(), str(tmp_path / "p"), NOW, standard=True)
    assert rep["category"] == "standard" and rep["accepted"]
    sim.submit(rep["user_job"]["job"], "noiseless")
    res = J.score(rep["user_job"]["job"], rep["user_job"]["runs"])
    (r,) = res.values()
    assert r["per_bit"] == 1.0 and r["whole"] == 1.0 and r["usable"] == n
    # the logical counts of one input are that input only
    out = J.read(rep["user_job"]["job"], rep["user_job"]["runs"])
    (rec, cc), = out.values()
    lg = J.logical_counts(cc[("neel", "echo")], n)
    assert list(lg) == ["".join(str(q % 2) for q in range(n))]


def test_echo_returns_the_flipped_input(tmp_path):
    dev = Dv.fetch_fake("FakeFez")
    rep = P.plan(dev, dict(kind="echo", n=20, D=4, shots=200), R.empty(), str(tmp_path / "p"), NOW, standard=True)
    sim.submit(rep["user_job"]["job"], "noiseless")
    (r,) = J.score(rep["user_job"]["job"], rep["user_job"]["runs"]).values()
    assert abs(r["rho4"] - 1) < 1e-12 and r["usable"] == 20
