"""Sentinel fallback "same_qubits" (2026-11-13): off by default; with it, a record whose 40-site ring needs pendants
(ibm_marrakesh 2026-11-12) builds, the sentinel record says so, and where the rings already agree (ibm_fez) the job is
byte-identical to the build without it. Needs the lab notebook's device records: set TOQB_DIR (as test_ledger)."""
import json
import os

import pytest

TOQB = os.environ.get("TOQB_DIR")
pytestmark = pytest.mark.skipif(not TOQB, reason="needs TOQB_DIR (results/ohm-2026-11-12/fetch_in device records)")


def _dev(name):
    from psf_os import device as Dv
    return Dv.Device.load(os.path.join(TOQB, "results", "ohm-2026-11-12", "fetch_in", name, "dev"))


def _user():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _build(dev, out, fallback):
    from psf_os import sentinel, user2d as U
    spec = dict(sentinel.SENTINEL_JOB, fallback=fallback) if fallback else None
    return U.build_batch(dev, [("u", _user())], str(out), shots=1000, sentinel_spec=spec)


def test_marrakesh_refused_by_default_and_built_with_fallback(tmp_path):
    dev = _dev("ibm_marrakesh")
    with pytest.raises(RuntimeError, match="did not land"):
        _build(dev, tmp_path / "a", None)
    _build(dev, tmp_path / "b", "same_qubits")
    rec = json.load(open(tmp_path / "b" / "runs" / "ksent_n40_D24-fold_rule.json"))
    assert rec["fallback"]["rule"] == "same_qubits"
    assert sorted(rec["fallback"]["user_ring"]) == sorted(rec["fallback"]["sentinel_ring"])
    assert rec["fallback"]["user_ring"] != rec["fallback"]["sentinel_ring"]


def test_fez_identical_with_and_without_fallback(tmp_path):
    dev = _dev("ibm_fez")
    _build(dev, tmp_path / "a", None)
    _build(dev, tmp_path / "b", "same_qubits")
    ma = json.load(open(tmp_path / "a" / "job" / "manifest.json"))
    mb = json.load(open(tmp_path / "b" / "job" / "manifest.json"))
    assert ma["circuits_sha256"] == mb["circuits_sha256"]
    assert "fallback" not in json.load(open(tmp_path / "b" / "runs" / "ksent_n40_D24-fold_rule.json"))


def test_unknown_fallback_refused(tmp_path):
    with pytest.raises(ValueError):
        _build(_dev("ibm_fez"), tmp_path / "a", "anything")
