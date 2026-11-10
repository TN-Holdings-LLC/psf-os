"""Problem cards: the entrance refuses what it should, the fixed decompositions are the exponentials they claim, and the
exact answers are those of exp(-i dt H_g) products computed independently with dense matrices. Made-up values only."""
import json

import numpy as np
import pytest
from scipy.linalg import expm

from psf_os import card as C


def _chain_card(**over):
    card = {"card": 1, "name": "test chain",
            "lattice": {"kind": "chain", "size": [4]},
            "hamiltonian": {"terms": [
                {"pauli": "XX", "on": "edges", "coef": 0.37, "group": "xx"},
                {"pauli": "YY", "on": "edges", "coef": -0.21, "group": "yy"},
                {"pauli": "ZZ", "on": "edges", "coef": 0.53, "group": "zz"},
                {"pauli": "Z", "on": "sites", "coef": 0.11, "group": "f"},
                {"pauli": "X", "sites": [2], "coef": -0.29, "group": "xx"}]},
            "initial": {"bits": "0110"},
            "time": {"dt": 0.13, "steps": 3, "order": ["f", "zz", "xx", "yy"]},
            "measure": [{"op": "Z", "sites": "all"}, {"op": "ZZ", "sites": [0, 3]}, {"op": "X", "sites": [1]},
                        {"op": "Y", "sites": [2]}]}
    card.update(over)
    return card


def _dense(p, sites, n):
    mats = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.diag([1.0 + 0j, -1.0])}
    out = np.array([[1.0 + 0j]])
    for q in reversed(range(n)):   # little-endian: qubit 0 is the rightmost factor
        out = np.kron(out, mats[p[sites.index(q)]] if q in sites else np.eye(2))
    return out


def _reference(card):
    """Independent: dense exp(-i dt H_g) per group in order, from the initial basis state."""
    P = C.parse(card)
    n = P["n"]
    psi = np.zeros(2 ** n, complex)
    psi[int(P["state"][::-1], 2)] = 1.0
    for s in range(P["steps"]):
        for g in P["orders"][s % len(P["orders"])]:
            Hg = sum(t["coef"] * _dense(t["pauli"], list(t["sites"]), n) for t in P["groups"][g])
            psi = expm(-1j * P["dt"] * Hg) @ psi
    return {C.label(p, s): float(np.real(psi.conj() @ _dense(p, list(s), n) @ psi)) for p, s in P["measure"]}


def test_exact_equals_dense_reference():
    card = _chain_card()
    a, b = C.exact(card), _reference(card)
    assert a.keys() == b.keys()
    assert max(abs(a[k] - b[k]) for k in a) < 1e-12


def test_orders_cycle_and_grid_site_order():
    card = _chain_card(lattice={"kind": "grid2d", "size": [2, 2]},
                       time={"dt": 0.17, "steps": 3, "orders": [["f", "zz", "xx", "yy"], ["yy", "xx", "zz", "f"]]})
    n, edges = C.lattice(card["lattice"])
    assert n == 4 and sorted((a, b) for a, b, _ in edges) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    a, b = C.exact(card), _reference(card)
    assert max(abs(a[k] - b[k]) for k in a) < 1e-12
    # the cycle matters: the same card with one order differs
    one = _chain_card(lattice={"kind": "grid2d", "size": [2, 2]},
                      time={"dt": 0.17, "steps": 3, "order": ["f", "zz", "xx", "yy"]})
    c = C.exact(one)
    assert max(abs(a[k] - c[k]) for k in a) > 1e-6


@pytest.mark.parametrize("p", ["XX", "YY", "ZZ", "XY", "YZ", "ZX", "X", "Y", "Z"])
@pytest.mark.parametrize("a", [0.123, -0.456, np.pi / 4, -np.pi / 4, 3 * np.pi / 4, 1.234 + np.pi])
def test_term_gate_is_the_exponential(p, a):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    n = len(p)
    qc = QuantumCircuit(n)
    C.term_gate(qc, dict(pauli=p, sites=tuple(range(n)), coef=a), 1.0)
    U = Operator(qc).data
    V = expm(-1j * a * _dense(p, list(range(n)), n))
    ph = np.vdot(V.flatten(), U.flatten())
    assert abs(abs(ph) - 2 ** n) < 1e-9          # equal up to a global phase


def test_one_cz_for_quarter_turns():
    from qiskit import QuantumCircuit
    for a in (np.pi / 4, -np.pi / 4):
        qc = QuantumCircuit(2)
        C.zz_rotation(qc, a, 0, 1)
        assert qc.count_ops().get("cz") == 1
    qc = QuantumCircuit(2)
    C.zz_rotation(qc, 0.3, 0, 1)
    assert qc.count_ops().get("cz") == 2


def test_product_states():
    card = _chain_card(initial={"product": "+r-l"}, time={"dt": 0.0, "steps": 1, "order": ["f", "zz", "xx", "yy"]},
                       measure=[{"op": "X", "sites": [0]}, {"op": "Y", "sites": [1]}, {"op": "X", "sites": [2]},
                                {"op": "Y", "sites": [3]}, {"op": "Z", "sites": [0]}])
    e = C.exact(card)
    assert e == pytest.approx({"X_0": 1, "Y_1": 1, "X_2": -1, "Y_3": -1, "Z_0": 0}, abs=1e-12)


def _reasons(card, **kw):
    with pytest.raises(C.CardRefused) as ei:
        C.check(card, **kw)
    return " | ".join(ei.value.reasons)


def test_refused_off_edge_and_not_real():
    card = _chain_card()
    card["hamiltonian"]["terms"].append({"pauli": "ZZ", "sites": [0, 2], "coef": 0.3, "group": "zz"})
    card["hamiltonian"]["terms"].append({"pauli": "Z", "sites": [1], "coef": "0.2+0.1j", "group": "f"})
    card["hamiltonian"]["terms"].append({"pauli": "Z", "sites": [1], "coef": [0.2, 0.1], "group": "f"})
    r = _reasons(card)
    assert "not on an edge" in r and r.count("not a real number") == 2


def test_refused_group_not_commuting_and_bad_order():
    card = _chain_card()
    card["hamiltonian"]["terms"].append({"pauli": "X", "sites": [0], "coef": 0.3, "group": "zz"})
    card["time"]["order"] = ["f", "zz", "xx", "xx", "nope"]
    r = _reasons(card)
    assert "do not commute" in r and "unknown group" in r and "exactly once" in r


def test_refused_size_and_initial():
    card = _chain_card(lattice={"kind": "grid3d", "size": [4, 4, 2]}, initial={"bits": "01"})
    r = _reasons(card)
    assert "only up to 28" in r and "initial" in r
    # without exact answers the size is not a reason
    card["initial"] = {"product": "0"}
    assert C.check(card, exact=False)["sites"] == 32


def test_conserved_charge():
    card = _chain_card()
    card["hamiltonian"]["terms"] = [t for t in card["hamiltonian"]["terms"] if t.get("on") == "edges"]
    card["hamiltonian"]["terms"][1]["coef"] = 0.37          # XX + YY: total Z conserved
    card["time"]["order"] = ["zz", "xx", "yy"]
    card["conserved"] = [{"name": "total Z", "terms": [{"pauli": "Z", "on": "sites", "coef": 1}]}]
    rep = C.check(card)
    q = rep["conserved"][0]
    assert q["comm_H"] < 1e-12 and not q["kept_by_every_group"] and set(q["groups_not_keeping"]) == {"xx", "yy"}
    card["conserved"] = [{"name": "X_0", "terms": [{"pauli": "X", "sites": [0], "coef": 1}]}]
    assert "does not commute with H" in _reasons(card)


def test_settings_and_measured_circuits():
    card = _chain_card()
    P = C.parse(card)
    st = C.settings(P)
    assert st == ["ZZZZ", "-XY-"]
    mc = C.measured_circuits(card)
    assert [b for b, _ in mc] == st and all(qc.count_ops()["measure"] == st[k].replace("-", "").__len__()
                                           for k, (_, qc) in enumerate(mc))


def test_cli(tmp_path, capsys):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_chain_card()))
    assert C.main(["check", str(p)]) == 0
    assert C.main(["exact", str(p), "--out", str(tmp_path / "a.json")]) == 0
    bad = _chain_card(initial={"bits": "2"})
    p.write_text(json.dumps(bad))
    assert C.main(["check", str(p)]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_merge_equals_term_by_term_and_dense():
    card = _chain_card()
    card["hamiltonian"]["terms"] = [  # one group per edge parity, XX + YY + ZZ on each pair (made-up values)
        {"pauli": p, "sites": [a, a + 1], "coef": c, "group": f"e{a % 2}"}
        for a in range(3) for p, c in (("XX", 0.41), ("YY", -0.23), ("ZZ", 0.67))] + [
        {"pauli": "Z", "on": "sites", "coef": 0.19, "group": "f"}]
    card["time"]["order"] = ["f", "e0", "e1"]
    a, b, ref = C.exact(card), C.exact(card, merge=True), _reference(card)
    assert max(abs(a[k] - b[k]) for k in a) < 1e-10 and max(abs(b[k] - ref[k]) for k in b) < 1e-10
    plain, merged = C.circuit(card), C.circuit(card, merge=True)
    assert plain.count_ops()["cz"] == 3 * 3 * 2 * 3         # 3 steps x 3 pairs x 3 terms x 2 CZ
    assert merged.count_ops()["cz"] <= 3 * 3 * 3              # at most 3 CZ per pair and step


def test_merge_keeps_single_terms():
    card = _chain_card()
    assert C.circuit(card, merge=True).count_ops() == C.circuit(card).count_ops()
