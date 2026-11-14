# psf-os

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI](https://zenodo.org/badge/1442836966.svg)](https://doi.org/10.5281/zenodo.24211404)

A self-checking layer between a user's circuit and an IBM heavy-hex device, without the calibration map.

It places a circuit by formula (a ring by the rule, a 2D circuit by translation of its own pattern), adds checks made
of the user's own circuit to the same job (a floor, a mirror, a ring sentinel), keeps a list of what it found dead,
and returns an answer card. It does not read a calibration quality number: it uses the coupling graph and the
elements marked as failed.

**Status.** Release 0.1.1 (2026-11-14; the code of 0.1.0; changes in [`CHANGELOG.md`](CHANGELOG.md)). Archived on
Zenodo: every release [10.5281/zenodo.24211404](https://doi.org/10.5281/zenodo.24211404) (the latest), 0.1.1
[10.5281/zenodo.24211405](https://doi.org/10.5281/zenodo.24211405). The experiments behind it, and
their records, live in [psf-os-lab-notebook](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook) (named TOQB until
2026-11-12; the old links redirect); every device result below was declared (hash and time locked in git) before its
data existed.

## First device results: practice problems through the user entrance (2026-11-04)

Twenty-two circuits with known answers, written by a user in plain Qiskit (`rz`, `rzz`, `rx`, `h`, `cx`) from the
problems' definitions, were handed to `psf-os run2d` as one batch. psf-os placed each one, added its floor and mirror,
and returned an answer card per circuit. No error mitigation, no correction, no calibration map. The answers were
computed and sealed before the runs.

- P4: the kicked Ising chain of IBM's dual-unitary experiment (arXiv 2411.00765), 33 qubits, two kick strengths,
  t = 2-16.
- P5: the kicked Ising model in the form of IBM's 2023 utility experiment, on the device's own lattice (151 qubits on
  ibm_marrakesh, 153 on ibm_fez), theta = 0.4, 0.8, 1.0, t = 2-8 (up to 1,360 CZ).

| device | READS | AGREES (\|M - exact\| <= 0.05) | BEATS (the better trivial guess) |
|---|---|---|---|
| ibm_marrakesh | 22 / 22 | 19 / 22 | 19 / 22 |
| ibm_fez | 22 / 22 | 18 / 22 | 20 / 22 |

- **Benchmark number:** on both devices the 2D problems at theta 0.8 and 1.0 agree with the exact answer to t = 8
  (151-153 qubits; worst difference on fez 0.023), without correction.
- **Device user's number:**
  - theta 0.4 misses by -0.04 to -0.07 on both devices, the same size: its answers are near 1, and a loss of a few
    percent is enough to lose to "every qubit reads 1".
  - In P4 the one site that carries the signal falls behind with depth: 0.30 against 0.41 at t = 16.
  - One day, two devices 25 minutes apart, both in a good state (floor 0.96). The spread from day to day is not
    measured yet.
- The batch ran end to end with no hand step; on fez the entrance moved P4 off a failed coupler by itself.
- Records: psf-os-lab-notebook [`practice-2026-11-04`](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook/tree/main/results/practice-2026-11-04) and
  [`practice-fez-2026-11-04`](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook/tree/main/results/practice-fez-2026-11-04). The same evening, a correction by the floor and the mirror
  ([fez](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook/tree/main/results/corr-2026-11-04), [marrakesh](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook/tree/main/results/corr-marrakesh-2026-11-04)) worked on one device and not fully on the
  other; it is on hold.

## The daily record (from 2026-11-05)

Every night, on the devices with a short queue, psf-os solves a fixed problem and a new one from a pool whose answers
were sealed (hashes committed) before release; raw values, no correction, failures recorded. Rules:
[`prereg/daily/standing_rules.md`](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook/blob/main/prereg/daily/standing_rules.md); the record:
[`results/daily/RECORD.md`](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook/blob/main/results/daily/RECORD.md). Each line gives the declared verdict and, beside
it, the retained share of the signal (M / M_exact) and the normalised error: the declared criterion is an absolute
difference, so a large answer can DIFFER while keeping as much signal as a small one that AGREES.

## Reproduce

Three levels, from the cheapest. Each needs psf-os and a checkout of [psf-os-lab-notebook](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook), where the records live.

```
git clone https://github.com/TN-Holdings-LLC/psf-os && pip install "./psf-os[sim]"
git clone https://github.com/TN-Holdings-LLC/psf-os-lab-notebook && cd psf-os-lab-notebook
```

1. **Re-score from the committed counts (no IBM account, under a minute).** `python reproduce.py` runs every scorer
   unchanged on a copy of each result's committed raw counts and compares with its committed `scores.json`. On
   2026-11-05, in a fresh environment with qiskit 2.6.1, all 9 results came out the same.
2. **Recompute the exact answers.** The circuits are in each result folder (`circuits/*.qpy` and `.qasm`); the
   answers were computed by the scripts named in each folder's README (CPU for shallow depths, `gpu/*_answers.py` on a
   GPU for the deep light cones) and are checked there by two independent methods.
3. **Run the same circuits on a device.** `psf-os fetch` the device, `psf-os run2d` with the result's circuit files,
   `psf-os submit`, `psf-os collect`, then the result's scorer. A rebuild with another qiskit version gives the same
   gates (checked gate by gate, 2026-11-05) but a different QPY file hash.

Versions behind each result: [`results/ENVIRONMENT.md`](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook/blob/main/results/ENVIRONMENT.md).

## Layers for 2D circuits

| module | does |
|---|---|
| `psf_os.lattice2d` | places any edge set on Heron r2 labels by translation only: the first copy, in a fixed order, with no failed or suspect element; no search |
| `psf_os.user2d` | L6 for 2D: the user's circuit (QPY or OpenQASM) in, one job with its controls out, an answer card back; `rzz(+-pi/2)` is written exactly as one CZ (`compile.rzz_exact`) |
| `psf_os.mirror2d` | the user's circuit, a barrier, its inverse; accepted only with exactly twice the CZ count; flags qubits where the signal is lost |
| floor | the user's compiled circuit with every non-diagonal one-qubit gate removed: the same CZs, ideal ⟨Z⟩ = 1 |
| `psf_os.ibm` | `fetch` (coupling graph and failed elements only), `submit` (the job's own device only, dynamical decoupling and twirling off; nothing is sent if a circuit touches a failed element), `collect` |
| `psf_os.ledger`, `psf_os.repair2d`, `psf_os.crosstalk2d` | drafts (not declared): reliability ledger, the rule-based self-repair chain, crosstalk test pieces |
| `psf_os.grid3d` (prototype) | `psf-os run3d`: kicked Ising on an L x M x N cubic lattice placed by fixed rules (lattice rows on heavy-hex rows, bonds through the bridge qubits, an odd-even sort of rows), floor and mirror in the same job, read with `result2d`; correct without noise; costs several times what a search-based router needs (see psf-os-lab-notebook `notes/os3d_2026-11-06`) |
| `psf_os.dual`, `psf_os.kim`, `psf_os.ising2d` | the practice problems P4, P5, P6 as modules, with their exact answers |

## Added 2026-11-07 to 2026-11-13 (in 0.1.0)

| module | does |
|---|---|
| `psf_os.entrance`, `psf_os.entrance3d` | one entrance for a logical circuit (`psf-os entrance`): the family recognised from the circuit alone by fixed rules (chains, 2D patterns, 4 x M x N cubic lattices), placed, controls added, one job out |
| `psf_os.twin` | the Clifford twin of a compiled circuit: every RZ rounded to a multiple of pi/2, the same pulses on Heron (RZ is virtual), exact values by stabilizer simulation at any size; `kept_share` gives the share of the signal a device keeps (`--twin`) |
| `psf_os.sentinel` fallback | when the rule's ring sentinel cannot be placed (many failed qubits), the user's own qubits carry it (`--sentinel-fallback same_qubits`); the record says so |
| `psf_os.quirk`, `psf_os.quirk_chain` | the user's own quirk ledger orders placement candidates; a special condition of the placement policy, only when declared (`--special quirk_ledger --ledger L --before ISO`), never reading days after `--before` |
| `psf_os.chain`, `psf_os.strip` | chains that go round holes (rule C); honeycomb strips of Heron r2 |
| `psf_os.observables`, `psf_os.exact_tn` | Pauli-string observables beyond per-qubit Z; exact values by a light-cone tensor network (optional `tn` extra) |
| `psf_os.card`, `psf_os.report2d`, `psf_os.gatehash` | problem cards, the extended answer card, a circuit hash that does not depend on QPY |

## Layers for rings

| layer | module | does |
|---|---|---|
| L0 device | `psf_os.device` | coupling graph and failed elements of a device or a fake device; no quality number is read or stored |
| L1 scan | `psf_os.scan` | the rule's mirror echo per ring site (D = 24); dead runs (< 0.12, 3 sites or more) and weak runs (< max(0.12, half the ring median)) |
| L2 registry | `psf_os.registry` | the defect list: black paths LISTED after 2 jobs, expired after 7 days; gray paths for one day only; repair entries |
| L3 allocate | `psf_os.allocate` | the rule's placement with listed paths masked; reports what it skipped |
| L4 repair | `psf_os.repair` | local frame flip, mirror-type circuits only |
| L5 sentinel | `psf_os.sentinel` | M6 prototype: the rule's echo on the user's ring in the same job (4 circuits, no references); CLEAN / WEAK / DEAD by the registry's rules; findings enter the next registry file as gray entries of the day |
| L6 job | `psf_os.job` | a logical circuit in; the assembled job and a report out |

## Rules

The rules are taken unchanged from psf-os-lab-notebook (then TOQB) at 8d44f05: the thresholds, the run length, the 4-qubit centre path,
LISTED after 2 jobs, the 7-day expiry, and the 2-qubit overlap that makes two sightings the same defect. Changing
any of them is the owner's decision.

## Tests

`TOQB_DIR=<psf-os-lab-notebook checkout> python -m pytest -q` (the variable keeps the notebook's old name; 0.1.0:
286 passed, 1 skipped, with the notebook checked out and the optional pieces installed)

| test | what it checks |
|---|---|
| T1 | the registry made from the notebook's device records equals the notebook's `list_after.json`, byte for byte |
| T2 | on FakeKingston, FakeMarrakesh and FakeFez, n = 20, 40, 72, with and without a mask: the same physical ring and two-qubit count as the notebook's `kising_echo build --arm fold:rule` |
| T3 | on FakeFez in simulation (Aer stabilizer, the echo at its Clifford point h = pi/4, a ZZ Pauli error of 0.15 after every CZ on three planted couplers known only to the test): two scans list the two planted couplers that lie on standard rings, the mask hits exactly those couplers, and the usable sites grow (n = 40: 24 -> 40, n = 72: 56 -> 72) |
| T4 | the sh6 identity (OS plan M3, second form) and the echo, entered through the L6 interface on three fake devices, return every bit without noise |
| T5 | the frame flip leaves the ideal circuit unchanged (operator equality); the repair rules |
| T6 | the chain runs through a Target wrapper that lets an error value be compared only with the failed mark and raises on any other quality read; the device record and the compiler's Target hold no quality number |
| bridge | the notebook's `s2rc_device submit --local` runs a psf-os job folder, and psf-os scores what it writes |
| S1-S6 | sentinels: the user's circuits byte-identical with or without them; CLEAN without noise; a coupler planted after placement is flagged DEAD with a run around it; the finding is gray for that day only and adds nothing to LISTED; the time-model cap stops an over-long job; no map is read |
| Q1-Q4 | practice problems: each exact answer agrees by two independent methods; full score without noise; with planted two-qubit depolarizing errors the cluster parity and P1's read stabilizers are better on the avoiding ring; no map is read |
| user2d | P6 as a user circuit lands like P6; a gate off the lattice is refused; the floor keeps the CZs and returns to 0; run2d and the card on a local run; a batch of two circuits in one job with a card per circuit; one bad circuit refuses the batch; `rzz(+-pi/2)` is one CZ with the same unitary |
| ported | 31 of the notebook's tests (defect_list, safe_loop, qca, kising_echo, echo_tomo) on the ported code |

## Commands

A 2D circuit (or several, as one job):

```
psf-os fetch --backend ibm_fez --out DEV                                  # the saved IBM account; graph and failed marks only
psf-os run2d --device DEV --circuit a.qpy b.qasm --out RUN                # place, add floor + mirror + sentinel, assemble
psf-os submit --job RUN/job
psf-os collect --job RUN/job --wait-min 30
psf-os result2d --job RUN/job --runs RUN/runs --answers answers.json      # answer cards; answers optional
```

A ring:

```
psf-os fetch --backend fake:FakeFez --out DEV
psf-os plan --device DEV --job job.json --registry REG.json --out RUN     # no scan today -> a scan job first
psf-os submit --job RUN/scan/job --local                                  # --local: fake devices only
psf-os scan-ingest --device DEV --job RUN/scan/job --runs RUN/scan/runs --registry REG.json --out REG2.json --label s1
psf-os plan --device DEV --job job.json --registry REG2.json --out RUN2   # avoid, place, assemble
psf-os status --registry REG2.json
psf-os result --job RUN2/job/job --runs RUN2/job/runs
```

`job.json`: `{"kind": "sh6", "n": 40, "D": 12, "shots": 1000}`. Kinds: echo, sh6, qca, and the practice problems
`clifford_stab` (`k`, `seed`, optional `offset`, `max_weight`: P1, the random Clifford circuit read through a set of its output stabilizers with disjoint supports, every answer +1 or -1; `clifford_random`, read site by site, is kept for the records but cannot see a defect), `cluster_parity`, `kising_forward` (`L`, n <= 22), whose exact answers are recorded
in the job's record at build time and scored by `psf-os result`.

## Not yet

A declared self-repair test, the spread from day to day, the Rust core (M8), a resident service; observables on the
answer card (`psf_os.observables` is not yet used by the entrance); the twin and the special condition for 3D
circuits. A job is never moved
to another device ("stay"); a result from another device than the one the job was placed for is refused.

Code ported from psf-os-lab-notebook (formerly TOQB; same rights holder) says so at the top of each file.

AGPL v3. See `LICENSE`.
