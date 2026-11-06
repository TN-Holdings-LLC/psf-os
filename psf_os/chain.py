"""Chains on Heron r2 labels that go round holes: rule C (owner 2026-11-06 07:38 CET: adopted; not counted as a search
on the ground of reproducibility). Workplace port of TOQB notes/chain_detour_2026-11-06/detour.py (chain_dfs).

Heron r2: 8 rows of 16 qubits (label 20 r + c), bridges 20 r + 16 + k between rows r and r + 1 at column 3 + 4k
(r even) or 1 + 4k (r odd).

Rule C. The candidates are the simple paths from a start, in one fixed order: at a row qubit, first along the row in
the row's direction, then the bridge toward row + v, then the bridge toward row - v, then back along the row; at a
bridge, first toward row + v; entering a new row reverses the direction (a snake). Starts are tried in the fixed
order START_ORDER (v = +1 then -1; rows in order, from the top for +1 and the bottom for -1; column 0 then 15). The
first path of n sites is used. No score, no comparison; the walk backtracks, at most BUDGET steps per start (a step
is one visit of a qubit). The same record always gives the same chain and the same step count (both are returned and
belong in the record).

Health: a qubit is usable if it is not failed and, by default, not "suspect" (an end of a failed coupler); a link is
usable if it is a coupler of the record and not failed. With suspect=True this is the judgement of lattice2d.check,
the check the user entrance (run2d) applies, so a chain found here is accepted there at the copy (0, 0). With
suspect=False it is grid3d's judgement since 8b0b34f (failed qubits, failed couplers, couplers of the record only);
a chain found that way can be refused by run2d (seen on fez: qubit 33, an end of the failed coupler 32-33).
Reads only the device record (coupling graph and failed marks); no quality number.

    from psf_os import chain
    r = chain.place(dev, 91)          # dict(chain=[labels], start=(v, row, col), steps=..., starts_tried=..., ...)
"""
from __future__ import annotations

ROWS, COLS, STRIDE = 8, 16, 20
BUDGET = 200000
START_ORDER = tuple((v, r, c) for v in (+1, -1) for r in (range(ROWS) if v > 0 else range(ROWS - 1, -1, -1))
                    for c in (0, COLS - 1))


class Refused(Exception):
    pass


def bridge(r, col):
    """Label of the bridge between row r and r + 1 at column col, or None."""
    if not 0 <= r < ROWS - 1:
        return None
    k, ok = ((col - 3) // 4, (col - 3) % 4 == 0) if r % 2 == 0 else ((col - 1) // 4, (col - 1) % 4 == 0)
    return STRIDE * r + 16 + k if ok and 0 <= k <= 3 else None


def bridge_ends(b):
    """(label in the upper row, label in the lower row) of bridge b."""
    r, k = divmod(b - 16, STRIDE)
    col = 3 + 4 * k if r % 2 == 0 else 1 + 4 * k
    return STRIDE * r + col, STRIDE * (r + 1) + col


def health(dev, suspect=True):
    """(qubit_ok, link_ok) functions for the record (see the module text)."""
    bad = set(dev.failed_qubits)
    badc = {frozenset(c) for c in dev.failed_couplers}
    if suspect:
        bad |= {q for c in dev.failed_couplers for q in c}
    cpl = {frozenset(c) for c in dev.couplers}

    def q_ok(q):
        return q is not None and 0 <= q < dev.num_qubits and q not in bad

    def e_ok(a, b):
        e = frozenset((a, b))
        return e in cpl and e not in badc and q_ok(a) and q_ok(b)

    return q_ok, e_ok


def _neighbours(q, d, v):
    """Fixed order of the next labels from q (row direction d, vertical direction v), with the next direction."""
    rr, cc = divmod(q, STRIDE)
    out = []
    if cc < 16:
        if 0 <= cc + d < COLS:
            out.append((q + d, d))
        for rn in (rr + v, rr - v):
            if 0 <= rn < ROWS:
                b = bridge(min(rr, rn), cc)
                if b is not None:
                    out.append((b, -d))
        if 0 <= cc - d < COLS:
            out.append((q - d, -d))
    else:
        up, dn = bridge_ends(q)
        out += [(dn, d), (up, d)] if v > 0 else [(up, d), (dn, d)]
    return out


def walk(dev, n, v, r, c, budget=BUDGET, suspect=True):
    """Rule C from one start. Returns (chain or longest path reached, steps, reason or None)."""
    q_ok, e_ok = health(dev, suspect)
    s = STRIDE * r + c
    if not q_ok(s):
        return [], 0, "start qubit not usable"
    path, seen, best = [s], {s}, [s]
    steps = 1
    if n <= 1:
        return path[:n], steps, None
    stack = [iter(_neighbours(s, 1 if c == 0 else -1, v))]
    while stack:
        if steps > budget:
            return best, steps, "budget spent"
        q = path[-1]
        for x, dx in stack[-1]:
            if x not in seen and e_ok(q, x):
                path.append(x)
                seen.add(x)
                steps += 1
                if len(path) > len(best):
                    best = list(path)
                if len(path) >= n:
                    return list(path), steps, None
                stack.append(iter(_neighbours(x, dx, v)))
                break
        else:
            stack.pop()
            seen.discard(path.pop())
    return best, steps, "exhausted"


def place(dev, n, budget=BUDGET, suspect=True):
    """The chain of n sites by rule C: dict(chain, start, steps, starts_tried, passed_over, rule, budget, suspect).
    Refused if no start reaches n."""
    if dev.num_qubits != 156:
        raise Refused(f"rule C is written for Heron r2 labels (156 qubits); {dev.name} has {dev.num_qubits}")
    passed = []
    for st in START_ORDER:
        ch, steps, why = walk(dev, n, *st, budget=budget, suspect=suspect)
        if why is None:
            return dict(chain=ch, start=list(st), steps=steps, starts_tried=len(passed) + 1, passed_over=passed,
                        rule="C", budget=budget, suspect=suspect)
        passed.append(dict(start=list(st), reached=len(ch), steps=steps, why=why))
    raise Refused(f"rule C: no start reaches {n} sites; longest {max(p['reached'] for p in passed)}")


def check(dev, chain, suspect=True):
    """Independent check of a chain: distinct labels, every site usable, every link usable. [] if fine."""
    q_ok, e_ok = health(dev, suspect)
    why = []
    if len(set(chain)) != len(chain):
        why.append("a label repeats")
    why += [f"qubit {q} not usable" for q in chain if not q_ok(q)]
    why += [f"link ({a}, {b}) not usable" for a, b in zip(chain, chain[1:]) if not e_ok(a, b)]
    return why
