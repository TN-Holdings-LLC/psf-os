"""psf_os.crosstalk2d: the split arm gives the standard arm's state; acceptance refuses a missing barrier; the
patterns read back exactly without noise; the readout model recovers a planted correlated readout error."""
import numpy as np
import pytest

from psf_os import crosstalk2d as X
from psf_os import device as Dv
from psf_os import kim as K
from tests.test_kim import EDGES, QUBITS

GEO = EDGES  # the piece's geometry: its own edges
# a bipartite piece (one heavy-hex ring of 12 with two pendants) for the readout patterns; test_kim's piece has a
# 7-cycle and is not bipartite (refused by X.bipartition, as it should be)
HEX = sorted(tuple(sorted((k, (k + 1) % 12))) for k in range(12)) + [(0, 12), (6, 13)]
HQ = sorted({q for e in HEX for q in e})


def test_split_groups_are_non_adjacent_and_cover_the_classes():
    gs = X.groups(EDGES, GEO, split=True)
    assert X.is_split(gs, GEO)
    assert sorted(e for G in gs for e in G) == sorted(EDGES)
    assert len(gs) > len(K.colour(EDGES))
    assert not X.is_split(X.groups(EDGES, GEO, split=False), GEO)


@pytest.mark.parametrize("theta", [0.4, 1.0])
def test_split_and_standard_give_the_same_state(theta):
    from qiskit.quantum_info import Statevector
    for t in (1, 3):
        a = Statevector(X.circuit(QUBITS, EDGES, GEO, theta, t, split=False, measure=False))
        b = Statevector(X.circuit(QUBITS, EDGES, GEO, theta, t, split=True, measure=False))
        c = Statevector(K.circuit(QUBITS, EDGES, theta, t, measure=False))
        assert abs(a.inner(b)) > 1 - 1e-10 and abs(a.inner(c)) > 1 - 1e-10


def test_acceptance_on_compiled_circuits_and_a_missing_barrier():
    from psf_os.compile import RuleCompiler
    dev = Dv.fetch_fake("FakeKingston")
    rc = RuleCompiler(dev)
    qubits, edges = K.region(dev, suspect=True)
    geo = dev.all_couplers()
    for sp in (False, True):
        c = rc.pm.run(X.circuit(qubits, edges, geo, 0.4, 2, sp, width=dev.num_qubits))
        assert X.accept(c, X.designed(edges, geo, 2, sp), geo)[0]
        m = rc.pm.run(X.mirror(qubits, edges, geo, 0.4, 2, sp, width=dev.num_qubits))
        assert X.accept(m, X.designed(edges, geo, 2, sp, mirror=True), geo)[0]
    lc = X.circuit(qubits, edges, geo, 0.4, 1, True, width=dev.num_qubits)
    nb = lc.copy_empty_like()
    for ins in lc.data:
        if ins.operation.name != "barrier":
            nb.append(ins)
    ok, why = X.accept(rc.pm.run(nb), X.designed(edges, geo, 1, True), geo)
    assert not ok


def test_patterns_cover_every_edge_and_read_back_exactly():
    with pytest.raises(ValueError):
        X.bipartition(QUBITS, EDGES)
    from qiskit_aer import AerSimulator
    cov = X.coverage(HQ, HEX, X.pairs_at(HEX, 1))
    assert all(len(s) == 4 for s in cov.values())
    sim = AerSimulator()
    for n in X.PATTERNS:
        fl = X.pattern_set(n, HQ, HEX)
        counts = sim.run(X.pattern_circuit(HQ, fl), shots=50).result().get_counts()
        m, _ = X.zz(counts, len(HQ), [])
        assert np.allclose(m, [-1 if q in fl else 1 for q in HQ])


def test_readout_model_recovers_a_planted_correlated_error():
    # pair (0, 1) with a correlated readout error: when both are 1, both flip to 0 with probability 0.1
    rng = np.random.default_rng(1)
    pos = {q: k for k, q in enumerate(HQ)}
    n = len(HQ)

    def sample(prob_by_bits, shots=200000):
        # independent bits with P(1) given, then the planted joint flip on (0, 1)
        b = (rng.random((shots, n)) < prob_by_bits[None, :]).astype(int)
        both = (b[:, pos[0]] == 1) & (b[:, pos[1]] == 1) & (rng.random(shots) < 0.1)
        b[both, pos[0]] = 0
        b[both, pos[1]] = 0
        out = {}
        for row in b:
            k = "".join(str(x) for x in row[::-1])
            out[k] = out.get(k, 0) + 1
        return out

    pats = {}
    for name in X.PATTERNS:
        fl = set(X.pattern_set(name, HQ, HEX))
        pats[name] = sample(np.array([1.0 if q in fl else 0.0 for q in HQ]), shots=20000)
    flipped = {nm: X.pattern_set(nm, HQ, HEX) for nm in X.PATTERNS}
    M = X.pair_confusion(pats, flipped, pos, (0, 1))
    assert abs(M[0, 3] - 0.1) < 0.01 and abs(M[3, 3] - 0.9) < 0.01
    # a product state with P(1) = 0.3 on each: true C = 0; the model predicts the measured C
    p1 = np.full(n, 0.3)
    meas = X.connected(sample(p1), n, [(pos[0], pos[1])])[(pos[0], pos[1])]
    pred = X.predicted_connected(M, 0.4, 0.4, 0.16)
    assert abs(pred) > 0.005 and abs(meas - pred) < 0.006


def _hex_sampler(seed):
    from qiskit.quantum_info import Pauli, Statevector
    n = len(HQ)
    rng = np.random.default_rng(seed)

    def exact(theta, t):
        sv = Statevector(K.circuit(HQ, HEX, theta, t, measure=False))

        def ev(qs):
            lab = ["I"] * n
            for q in qs:
                lab[n - 1 - q] = "Z"
            return float(sv.expectation_value(Pauli("".join(lab))).real)
        return dict(z=[ev([q]) for q in HQ], pairs={e: (ev([e[0]]), ev([e[1]]), ev(list(e))) for e in HEX}), \
            sv.probabilities()

    def sample(probs, shots, planted, f):
        idx = rng.choice(len(probs), size=shots, p=probs)
        b = ((idx[:, None] >> np.arange(n)) & 1).astype(int)
        for i, j in planted:
            m = (b[:, i] == 1) & (b[:, j] == 1) & (rng.random(shots) < f)
            b[m, i] = 0
            b[m, j] = 0
        keys, cnt = np.unique(b[:, ::-1], axis=0, return_counts=True)
        return {"".join(map(str, k)): int(c) for k, c in zip(keys, cnt)}
    return exact, sample


def test_analysis_end_to_end_with_planted_pair_readout():
    # a joint 11 -> 00 readout flip (probability 0.1) on one matching; no other error. The pair-confusion model
    # predicts the measured ratio on those pairs; the spectator spread is 0 without planted noise and > 0 with it
    exact, sample = _hex_sampler(3)
    planted = K.colour(HEX)[0]
    fl = {nm: X.pattern_set(nm, HQ, HEX) for nm in X.PATTERNS}
    pos = {q: k for k, q in enumerate(HQ)}
    for f in (0.0, 0.1):
        cells, ex = {}, {1.0: {}}
        for t in (2, 3):
            e, pr = exact(1.0, t)
            ex[1.0][t] = e
            for arm in "SD":
                cells[(arm, 1.0, t)] = sample(pr, 20000, planted, f)
        for arm in "SD":
            cells[(arm, 0.0, 1)] = sample(np.eye(2 ** len(HQ))[0], 4000, planted, f)
            cells[("m" + arm, 3)] = sample(np.eye(2 ** len(HQ))[0], 4000, planted, f)
        for nm in X.PATTERNS:
            v = np.zeros(2 ** len(HQ))
            v[sum(1 << q for q in fl[nm])] = 1
            cells[("pat", nm)] = sample(v, 20000, planted, f)
        r = X.analyse(cells, ex, HQ, HEX, fl, theta_q1=1.0, nboot=20)
        assert set(r["verdicts"]) == {"Q1", "Q2_magnetisation", "Q3"}
        pats = {k[1]: c for k, c in cells.items() if k[0] == "pat"}
        conf = {p: X.pair_confusion(pats, fl, pos, p) for p in planted}
        exs = {t: ex[1.0][t]["pairs"] for t in ex[1.0]}
        meas = np.mean([X.amplification({t: cells[(a, 1.0, t)] for t in exs}, exs, pos, planted)[0] for a in "SD"])
        pred = X.readout_amplification(conf, exs, planted)[0]
        if f == 0.0:
            assert pred == pytest.approx(1.0) and r["estimate"]["spectator_tv_max"] == 0.0
            assert abs(meas - 1) < 0.04
        else:
            assert pred < 0.95 and abs(meas - pred) < 0.04 and r["estimate"]["spectator_tv_max"] > 0.02
