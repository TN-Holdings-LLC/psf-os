# Problem cards (`psf_os/card.py`)

A problem card is one JSON file that states a time-evolution problem of a spin system: the lattice, the Hamiltonian,
the initial state, the time steps and their Trotter order, what is measured, and optionally a declared conserved
quantity. `psf_os.card` checks it at the entrance (before anything runs), builds the logical first-order Trotter
circuit from it, and computes that circuit's exact expectation values (statevector, up to 28 sites).

Placement is not part of the card. The logical circuit is handed to psf-os's existing entrance. `user2d.py` and
`compile.py` are unchanged.

Status: first version (2026-11-10). Not declared.

```
python -m psf_os.card check CARD.json            # entrance checks; exit 2 with every reason when refused
python -m psf_os.card exact CARD.json [--out A]  # exact expectation values of the card's Trotter circuit
```

```python
from psf_os import card as C
card = C.load("card.json")
C.check(card)        # summary, or CardRefused(reasons)
C.circuit(card)      # logical circuit, qubit i = site i, no measurement
C.measured_circuits(card)   # [(basis, circuit)], one per reading setting
C.exact(card)        # {"Z_0": ..., "Z_0 Z_3": ..., "X_1": ...}
C.hamiltonian(card)  # SparsePauliOp
```

## The format

```json
{
 "card": 1,
 "name": "Heisenberg XXX 4x2x2, dt 0.05",
 "lattice": {"kind": "grid3d", "size": [4, 2, 2]},
 "hamiltonian": {"terms": [
   {"pauli": "XX", "on": "edges", "axis": 0, "coef": 1.0, "group": "x"},
   {"pauli": "ZZ", "sites": [0, 2], "coef": 1.0, "group": "g1"},
   {"pauli": "Z",  "on": "sites", "coef": 0.3, "group": "field"}
 ]},
 "initial": {"bits": "0101101001011010"},
 "time": {"dt": 0.05, "steps": 4, "order": ["field", "x", "g1"]},
 "measure": [{"op": "Z", "sites": "all"}, {"op": "ZZ", "sites": [0, 1]}],
 "conserved": [{"name": "total Z", "terms": [{"pauli": "Z", "on": "sites", "coef": 1}]}]
}
```

### `card`
The format version. It must be `1`.

### `lattice`
Sets the sites and the edges.

| kind | fields | sites | edges |
|---|---|---|---|
| `chain` | `size: [L]` | 0 .. L-1 | (i, i+1) |
| `grid2d` | `size: [L, M]` | (x, y) is site x·M + y | nearest neighbours |
| `grid3d` | `size: [L, M, N]` | (x, y, z) is site (x·M + y)·N + z | nearest neighbours |
| `graph` | `sites: n`, `edges: [[a, b], ...]` | 0 .. n-1 | as listed |

- Grid sites are numbered in `itertools.product` order, which is also the order of `psf_os.grid3d.sites`.
- `"periodic": true` closes every grid axis whose length is more than 2.
- On a grid, each edge also carries its axis (0, 1 or 2), so terms can be selected by direction.
- A long-range model, such as the Schwinger model after the gauge field is eliminated, uses `graph` with every pair
  of sites it couples listed as an edge.

### `hamiltonian.terms`
Each term is a Pauli string on one site or on one edge, with a real coefficient and a group name.

- `pauli`: one or two letters from X, Y, Z.
- Where the term acts, given in one of three ways:
  - `sites: [i]` or `sites: [i, j]`: one explicit term.
  - `on: "sites"`: one-letter Paulis, expanded to every site.
  - `on: "edges"`, with an optional `axis`: two-letter Paulis, expanded to every edge, or to the edges of one axis.
- `coef`: a real number. A complex value, a string or a list is refused, because H must be Hermitian.
- `group`: the name used in the Trotter order. The terms of one group must commute with each other, so their order
  inside the group does not matter.

H = Σ coef · P. Constant terms (multiples of the identity) only add a global phase and are left out.

### `initial`
Give exactly one of the following.
- `bits`: a string of n characters `0` or `1`, where character i is site i.
- `product`: one character for every site, or one character repeated. The characters are `0`, `1`, `+`, `-`, `r`
  (|+i>) and `l` (|−i>).

### `time`
- `dt`: a real number.
- `steps`: an integer, 0 or more.
- The order, given in exactly one of two ways:
  - `order`: a list of group names. Every group appears exactly once, and the same order is used for every step.
  - `orders`: a list of such lists. Step s uses `orders[s mod len(orders)]`. This is needed when the order changes
    from step to step. Examples are a swap network run in reverse on even steps, or the bond order of a placed
    layout.

One step applies exp(−i·dt·coef·P) for each group in the step's order and each term of that group.

### `measure`
A list of `{"op": P, "sites": ...}`.
- P is one or two letters, for example `Z`, `X`, `Y` or `ZZ`.
- The sites can be:
  - `[i]` or `[i, j]` (any two distinct sites, not only edges)
  - `"all"`, for a one-letter op
  - `"edges"`, for a two-letter op

The reading bases follow from this list. Each observable goes into the first setting whose letters agree with it
(`C.settings`). A Z-only card is read in one setting.

### `conserved` (optional)
A list of `{"name": ..., "terms": [...]}`. The terms use the same syntax as `hamiltonian.terms`, without groups and
without the edge rule. The entrance checks [H, Q] = 0 with `SparsePauliOp`. It also reports whether every group
commutes with Q separately, which is when the Trotter circuit keeps Q exactly.

## Checks at the entrance (`CardRefused`, nothing runs)

Every reason found is listed, not only the first one.

- **Format:** the version and the required fields, and a valid lattice.
- **Terms:**
  - The Pauli letters are valid.
  - Sites are inside the lattice and the two sites of a term are different.
  - A two-site term lies on an edge.
  - The coefficient is real (H is Hermitian).
- **Groups and order:**
  - The terms of each group commute pairwise.
  - Every step's order names known groups, and each group exactly once.
- **Initial state:** the length and the characters are valid.
- **Measurement:** the ops and sites are valid.
- **Conserved quantity:** [H, Q] = 0, to a largest coefficient of 1e−10.
- **Size:** at most 28 sites while exact answers are wanted. `check(card, exact=False)` drops this limit.

The conserved-quantity check runs only when the card has no other fault.

Examples of refusals (`python -m psf_os.card check`, exit code 2). The cards are in `examples/cards/`.

```
$ python -m psf_os.card check examples/cards/refused_example.json
REFUSED
  - size: 32 sites; exact answers by statevector only up to 28
  - hamiltonian term 2: ZZ on sites [0, 3] is not on an edge of the lattice
  - hamiltonian term 3: coefficient '0.2+0.1j' is not a real number (H must be Hermitian)
  - group 'zz': terms ZZ[0, 8] and X[0] do not commute (a group's terms must commute)
  - time.order: unknown group(s) ['y']
  - initial: bits must be a string of 32 characters 0/1 (character i = site i)

$ python -m psf_os.card check examples/cards/refused_conserved.json
REFUSED
  - conserved 'total X': does not commute with H (largest coefficient of [H, Q] 2)
```

## Building the circuit

Each term is turned into gates by a fixed decomposition. Nothing is searched.

| term | gates |
|---|---|
| one site | RX, RY or RZ (2·dt·coef) |
| two sites | each qubit turned so that its letter reads as Z (X: H; Y: S† H), then exp(−i·a·ZZ), then turned back |
| exp(−i·a·ZZ), general a | H · CZ · RX(2a) · CZ · H on the second qubit (2 CZ) |
| exp(−i·a·ZZ), a = ±π/4 mod π | RZ(±π/2) on both qubits and one CZ |

Qubit i is site i. The circuit has no measurement. `measured_circuits` adds the basis turns and measures site i into
clbit i.

## Exact answers

`C.exact` returns the exact expectation value of every observable of the card's own Trotter circuit, not of
exp(−iHt).
- Up to 20 sites it uses `qiskit.quantum_info.Statevector`.
- From 21 to 28 sites it uses qiskit-aer's `save_expectation_value`.

## Example cards (`examples/cards/`)

These reproduce existing problems. Each card's exact answers match the existing module to 1e−10 or better.

- **`kicked_ising_4x2x2.json`**: psf_os.grid3d's kicked Ising, θ 0.6.
  - `dt` is 1, the ZZ coefficient is −π/4 and the X coefficient is θ/2.
  - RZZ(−π/2) is exp(+iπ/4·ZZ).
- **`heisenberg_4x2x2_textbook.json`**: Heisenberg XXX, dt 0.05, Néel state, the bonds coloured into 6 commuting
  groups.
- **`heisenberg_4x2x2_heisE_line_kingston_c0_t4.json`**: the same model with the bond order of the heisE line layout
  on the 11-08 ibm_kingston record (candidate 0, t 4).
  - Every bond is its own group, and `orders` gives one order per step.
  - The order belongs to the layout, not to the problem.
- **`schwinger_N6_M1.json`**: the Schwinger model of TOQB gpu/schwinger.py (Martinez et al. 2016), N 6, parameter
  set M1, dt 0.25, 2 steps.
  - The lattice is a `graph` with every pair of sites as an edge.
  - Each layer of the swap network is a group.
  - `orders` runs the layers forward on odd steps and backward on even steps.
