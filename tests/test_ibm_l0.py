"""L0 on an IBM device (psf_os.ibm), checked with a fake backend of qiskit-ibm-runtime run locally through SamplerV2:
submit sends only to the job's own device with the options off (a SamplerV2 stand-in runs the circuits on Aer), refuses when a circuit touches an element the device
marks as failed now, and collect writes counts that psf-os scores. No IBM connection; no key."""
import copy
import json
import os

import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from psf_os import device as Dv  # noqa: E402
from psf_os import ibm  # noqa: E402
from psf_os import job as J  # noqa: E402
from psf_os import plan as P  # noqa: E402
from psf_os import registry as R  # noqa: E402


class FakeService:
    """Stands in for QiskitRuntimeService: backend(name) and job(id)."""
    allows_fake = True

    def __init__(self, backend):
        self.b, self.jobs = backend, {}

    def backend(self, name):
        return self.b

    def job(self, job_id):
        return self.jobs[job_id]


def _job(tmp_path, kind="sh6"):
    from qiskit_ibm_runtime import fake_provider
    b = fake_provider.FakeFez()
    dev = Dv.fetch_fake("FakeFez")
    spec = dict(kind="sh6", n=20, D=6, shots=200) if kind == "sh6" else dict(kind="cluster_parity", n=20, shots=200)
    rep = P.plan(dev, spec, R.empty(), str(tmp_path / "p"), P.now_cet(), standard=True)
    return b, rep


class _Opts:
    """Records the options as SamplerV2's nested options would hold them."""
    def __init__(self):
        self.dynamical_decoupling = type("DD", (), {})()
        self.twirling = type("TW", (), {})()


class _Result:
    def __init__(self, counts):
        self.c = counts

    def join_data(self):
        return self

    def get_counts(self):
        return self.c


class _Job:
    def __init__(self, rows):
        self.rows = rows

    def job_id(self):
        return "stub-job-1"

    def status(self):
        return "DONE"

    def result(self):
        return [_Result(r["counts"]) for r in self.rows]

    def metrics(self):
        return {"usage": {"quantum_seconds": 0}}


def _stub_sampler(svc):
    """A SamplerV2 stand-in: the job's circuits run noiselessly on Aer (psf-os's sim), the options are recorded."""
    from psf_os import sim

    class S:
        def __init__(self, mode):
            self.mode, self.options = mode, _Opts()

        def run(self, pubs):
            svc.options = dict(dd=self.options.dynamical_decoupling.enable, tw=self.options.twirling.enable_gates,
                               twm=self.options.twirling.enable_measure)
            rows = sim.run_circuits([p[0] for p in pubs], pubs[0][2], 7, None, 0.0, pauli=None)
            job = _Job(rows)
            svc.jobs[job.job_id()] = job
            return job
    return S


def test_submit_collect_score(tmp_path):
    b, rep = _job(tmp_path)
    svc = FakeService(b)
    jid = ibm.submit(svc, rep["user_job"]["job"], sampler_cls=_stub_sampler(svc))
    assert svc.options == dict(dd=False, tw=False, twm=False)
    sub = json.load(open(os.path.join(rep["user_job"]["job"], "submission.json")))
    assert sub["job_id"] == jid and sub["stay"] is True and sub["sampler_options"] == ibm.OPTIONS
    assert "DONE" in ibm.collect(svc, rep["user_job"]["job"]).upper()
    (r,) = J.score(rep["user_job"]["job"], rep["user_job"]["runs"]).values()
    assert r["per_bit"] == 1.0
    with pytest.raises(SystemExit):
        ibm.submit(svc, rep["user_job"]["job"], sampler_cls=_stub_sampler(svc))  # never twice


def test_refused_on_a_failed_element(tmp_path):
    from qiskit.transpiler import InstructionProperties
    b, rep = _job(tmp_path)
    man = json.load(open(os.path.join(rep["user_job"]["job"], "manifest.json")))
    ring = rep["ring"]
    t = copy.deepcopy(b.target)
    e = (ring[0], ring[1]) if (ring[0], ring[1]) in t["cz"] else (ring[1], ring[0])
    p = t["cz"][e]
    t.update_instruction_properties("cz", e, InstructionProperties(duration=p.duration, error=1.0))

    class B:
        name, target = b.name, t
    with pytest.raises(SystemExit):
        ibm.submit(FakeService(B()), rep["user_job"]["job"])
    ref = json.load(open(os.path.join(rep["user_job"]["job"], "submission_refused.json")))
    assert ref["problems"] and "failed coupler" in ref["problems"][0]
    assert not os.path.exists(os.path.join(rep["user_job"]["job"], "submission.json"))
    assert man["stay"] is True


def test_refused_on_another_device(tmp_path):
    from qiskit_ibm_runtime import fake_provider
    _, rep = _job(tmp_path)
    with pytest.raises(SystemExit):
        ibm.submit(FakeService(fake_provider.FakeMarrakesh()), rep["user_job"]["job"])
    assert not os.path.exists(os.path.join(rep["user_job"]["job"], "submission.json"))


def test_submission_check_reads_no_map(tmp_path):
    from qiskit import qpy
    from qiskit_ibm_runtime import fake_provider
    from test_t6_no_map import Guarded
    _, rep = _job(tmp_path)
    circs = qpy.load(open(os.path.join(rep["user_job"]["job"], "circuits.qpy"), "rb"))
    assert ibm.problems(circs, Guarded(fake_provider.FakeFez().target)) == []
