"""psf_os.lattice2d: translation copies of any edge set on Heron r2 labels; P6's placement unchanged."""
import copy
import random

import pytest

from psf_os import device as Dv
from psf_os import ising2d as I
from psf_os import lattice2d as L


def old_place(dev):
    """ising2d.place as of psf-os 61e9edd (the declared P6 placement), kept here as the reference."""
    working = {tuple(sorted(e)) for e in dev.working_couplers()}
    suspect = {q for c in dev.failed_couplers for q in c}
    reasons = []
    for dr, dc, m in I.copies():
        why = []
        for q in I.QUBITS:
            p = m[q]
            if p >= dev.num_qubits or not dev.qubit_ok(p):
                why.append(f"qubit {p} failed")
            elif p in suspect:
                why.append(f"qubit {p} suspect")
        for a, b in I.EDGES:
            if tuple(sorted((m[a], m[b]))) not in working:
                why.append(f"coupler ({m[a]}, {m[b]}) not working")
        if not why:
            return dr, dc, m, reasons
        reasons.append(dict(drow=dr, dcol=dc, why=why))
    raise RuntimeError(reasons)


def with_marks(name, couplers, qubits):
    d = copy.deepcopy(Dv.fetch_fake(name))
    d.failed_couplers = [tuple(e) for e in couplers]
    d.failed_qubits = list(qubits)
    return d


def outcome(f, dev):
    try:
        return f(dev)
    except RuntimeError:
        return "none"


def same(dev):
    a, b = outcome(old_place, dev), outcome(I.place, dev)
    if a == "none" or b == "none":
        return a == b
    return a[:3] == b[:3] and a[3] == b[3]


def test_copies_are_real_edges_on_marrakesh():
    dev = Dv.fetch_fake("FakeMarrakesh")
    graph = {tuple(sorted(e)) for e in dev.all_couplers()}
    cs = L.copies(dev, I.EDGES)
    assert [(dr, dc) for dr, dc, _ in cs] == I.OFFSETS  # the general enumeration gives exactly P6's six, in order
    for dr, dc, m in cs:
        assert m == {q: I.shift(q, dr, dc) for q in I.QUBITS}
        assert all(tuple(sorted((m[a], m[b]))) in graph for a, b in I.EDGES)


def test_small_pattern_has_many_copies_all_real():
    dev = Dv.fetch_fake("FakeMarrakesh")
    graph = {tuple(sorted(e)) for e in dev.all_couplers()}
    pat = [(3, 4), (3, 16), (16, 23), (22, 23), (23, 24)]  # a T-shaped piece through a bridge
    cs = L.copies(dev, pat)
    assert len(cs) >= 6
    for _, _, m in cs:
        assert all(tuple(sorted((m[a], m[b]))) in graph for a, b in pat)
    # a shift that breaks the bridge parity is refused
    with pytest.raises(ValueError):
        L.shift(3, 1, 0)


def test_suspect_copies_are_passed_over():
    dev = with_marks("FakeKingston", [(9, 10), (29, 38), (85, 86)], [])
    dr, dc, m, reasons = I.place(dev)
    assert (dr, dc) == old_place(dev)[:2] and (dr, dc) != (0, 0)
    assert reasons[0]["why"][:4] == ["qubit 9 suspect", "qubit 10 suspect", "qubit 29 suspect", "qubit 38 suspect"]
    assert not any(p in {9, 10, 29, 38, 85, 86} for p in m.values())


@pytest.mark.parametrize("name,couplers,qubits", [
    ("FakeKingston", None, None), ("FakeFez", None, None), ("FakeMarrakesh", None, None),
    ("FakeFez", [(32, 33)], [72]),
    ("FakeKingston", [(9, 10), (29, 38), (112, 113), (121, 122), (130, 131), (145, 146), (146, 147)], [112, 113, 121, 146]),
    ("FakeKingston", [(9, 10), (29, 38), (112, 113), (121, 122), (130, 131), (145, 146), (146, 147), (85, 86)],
     [112, 113, 121, 146]),
])
def test_p6_placement_unchanged(name, couplers, qubits):
    dev = Dv.fetch_fake(name) if couplers is None else with_marks(name, couplers, qubits)
    assert same(dev)


def test_p6_placement_unchanged_random_marks():
    rnd = random.Random(20261104)
    base = Dv.fetch_fake("FakeKingston")
    edges = sorted(tuple(sorted(e)) for e in base.all_couplers())
    for _ in range(200):
        fc = rnd.sample(edges, rnd.randint(0, 12))
        fq = rnd.sample(range(base.num_qubits), rnd.randint(0, 4))
        assert same(with_marks("FakeKingston", fc, fq))
