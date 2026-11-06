"""psf_os.chain (rule C, owner 2026-11-06 07:38 CET): reproducible chains round holes, the entrance's health check."""
import dataclasses

import pytest

from psf_os import chain as C
from psf_os import device as Dv
from psf_os import lattice2d as L

# the failed marks of the marrakesh record of 2026-11-05 (geometry: the same Heron r2 lattice as FakeFez)
MARRAKESH_1105 = dict(failed_qubits=[46, 73, 82, 113, 119, 130], failed_couplers=[[113, 119]])


def _fez(**marks):
    return dataclasses.replace(Dv.fetch_fake("FakeFez"), **{**dict(failed_qubits=[], failed_couplers=[]), **marks})


def _snake_s1(n):
    """du91's shape S1 (rows from the top, turn at column 15 on even rows and 1 on odd rows), written out here."""
    out, r, col = [], 0, 0
    while len(out) < n:
        turn = 15 if r % 2 == 0 else 1
        step = 1 if turn >= col else -1
        for c in range(col, turn + step, step):
            out.append(20 * r + c)
        out.append(C.bridge(r, turn))
        r, col = r + 1, turn
    return out[:n]


def _entrance_ok(dev, ch):
    return L.check(dev, [tuple(sorted(e)) for e in zip(ch, ch[1:])], {q: q for q in ch}) == []


def test_without_holes_the_first_start_is_the_s1_snake():
    dev = _fez()
    for n in (51, 71, 91):
        r = C.place(dev, n)
        assert r["start"] == [1, 0, 0] and r["chain"] == _snake_s1(n)


def test_same_record_same_chain_and_steps():
    dev = _fez(**MARRAKESH_1105)
    a, b = C.place(dev, 91), C.place(_fez(**MARRAKESH_1105), 91)
    assert a == b
    assert len(a["chain"]) == 91 and a["steps"] >= 91 and a["starts_tried"] == 1
    assert C.check(dev, a["chain"]) == [] and _entrance_ok(dev, a["chain"])
    assert not set(a["chain"]) & set(MARRAKESH_1105["failed_qubits"])


def test_suspect_qubits_follow_the_user_entrance():
    dev = _fez(failed_couplers=[[32, 33]])
    loose = C.place(dev, 51, suspect=False)      # grid3d's judgement (8b0b34f): the ends of a failed coupler allowed
    strict = C.place(dev, 51)                    # lattice2d.check's judgement (run2d): not allowed
    assert 33 in loose["chain"] and not _entrance_ok(dev, loose["chain"])
    assert not {32, 33} & set(strict["chain"]) and _entrance_ok(dev, strict["chain"])


def test_refusals():
    dev = _fez(**MARRAKESH_1105)
    with pytest.raises(C.Refused):
        C.place(dev, 200)                        # longer than any path
    with pytest.raises(C.Refused):
        C.place(dataclasses.replace(dev, num_qubits=127), 51)
