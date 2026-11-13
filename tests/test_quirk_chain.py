"""psf_os.quirk_chain.candidates: the first candidate is chain_of's chain; every candidate is a usable chain.
The device-record tests need the lab notebook's records: set TOQB_DIR (as test_ledger)."""
import glob
import os

import pytest

from psf_os import chain as CH
from psf_os import quirk_chain as QC
from psf_os.device import Device

TOQB = os.environ.get("TOQB_DIR")
needs = pytest.mark.skipif(not TOQB, reason="needs TOQB_DIR (committed device records under results/)")

# records chosen to cover a shape at its own place, a translated shape (fez 11-05, n 20 and 40: shift 2 and 4 rows),
# and rule C (fez 11-12, marrakesh 11-12 at 91)
RECORDS = ["results/daily/2026-11-05/ibm_fez/dev",
           "results/daily/2026-11-05/ibm_marrakesh/dev",
           "results/xx91-2026-11-12/fetch_in/ibm_fez/dev",
           "results/hub-dt-2026-11-12/fetch_in/ibm_marrakesh/dev",
           "results/du51-2026-11-07/ibm_kingston/dev"]


def devices():
    for r in RECORDS:
        p = os.path.join(TOQB, r)
        if os.path.exists(os.path.join(p, "device.json")):
            yield r, Device.load(p)


@needs
@pytest.mark.parametrize("n", [20, 40, 91])
def test_first_is_chain_of_and_all_are_usable(n):
    got = 0
    for r, dev in devices():
        try:
            labels, how = CH.chain_of(dev, n)
        except CH.Refused:
            with pytest.raises(CH.Refused):
                QC.candidates(dev, n)
            continue
        cs = QC.candidates(dev, n)
        assert cs[0]["qubits"] == QC.default_physical(dev, n), r
        assert cs[0]["rule"] == how["rule"], r
        if how["rule"] == "C":
            assert cs[0]["qubits"] == labels and cs[0]["start"] == how["start"]
        else:
            assert cs[0]["shape"] == how["shape"]
        for c in cs:
            assert len(c["qubits"]) == n and CH.check(dev, c["qubits"]) == [], (r, c["name"])
        assert len({frozenset(c["qubits"]) for c in cs}) == len(cs)
        assert len({c["name"] for c in cs}) == len(cs)
        got += 1
    assert got >= 3


@needs
def test_translated_shape_is_on_physical_qubits():
    dev = Device.load(os.path.join(TOQB, "results/daily/2026-11-05/ibm_fez/dev"))
    labels, how = CH.chain_of(dev, 20)
    pl = CH.shape_place(dev, 20)
    assert how["rule"] == "shape" and (pl["drow"], pl["dcol"]) != (0, 0)
    c0 = QC.candidates(dev, 20)[0]
    assert c0["qubits"] == pl["chain"] and c0["qubits"] != labels   # the snake labels themselves are not the copy


@needs
def test_same_record_same_candidates():
    dev = Device.load(os.path.join(TOQB, "results/xx91-2026-11-12/fetch_in/ibm_fez/dev"))
    assert QC.candidates(dev, 40) == QC.candidates(dev, 40)


def test_refuses_other_devices():
    dev = Device(name="toy", num_qubits=5, couplers=[[0, 1], [1, 2]], failed_couplers=[], failed_qubits=[])
    with pytest.raises(CH.Refused):
        QC.candidates(dev, 3)
