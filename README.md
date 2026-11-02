# psf-os

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A self-checking layer between a ring circuit and an IBM heavy-hex device, without the calibration map.

It tests its own rings with a mirror echo, keeps a list of the paths it found dead, and places the next ring around
them. It does not read a calibration quality number: it uses the coupling graph and the elements marked as failed.

**Status.** Skeleton for milestone M5 of the OS plan (TOQB `prereg/drafts-2026-11-01/os_plan_DRAFT.md`). It is
not declared, and no performance is claimed. The experiments behind it, and their records, live in TOQB.

## Layers

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

The rules are taken unchanged from TOQB at 8d44f05: the thresholds, the run length, the 4-qubit centre path,
LISTED after 2 jobs, the 7-day expiry, and the 2-qubit overlap that makes two sightings the same defect. Changing
any of them is the owner's decision.

## Tests

`TOQB_DIR=<TOQB checkout> python -m pytest -q`

| test | what it checks |
|---|---|
| T1 | the registry made from TOQB's device records equals TOQB's `list_after.json`, byte for byte |
| T2 | on FakeKingston, FakeMarrakesh and FakeFez, n = 20, 40, 72, with and without a mask: the same physical ring and two-qubit count as TOQB's `kising_echo build --arm fold:rule` |
| T3 | on FakeFez in simulation (Aer stabilizer, the echo at its Clifford point h = pi/4, a ZZ Pauli error of 0.15 after every CZ on three planted couplers known only to the test): two scans list the two planted couplers that lie on standard rings, the mask hits exactly those couplers, and the usable sites grow (n = 40: 24 -> 40, n = 72: 56 -> 72) |
| T4 | the sh6 identity (OS plan M3, second form) and the echo, entered through the L6 interface on three fake devices, return every bit without noise |
| T5 | the frame flip leaves the ideal circuit unchanged (operator equality); the repair rules |
| T6 | the chain runs through a Target wrapper that lets an error value be compared only with the failed mark and raises on any other quality read; the device record and the compiler's Target hold no quality number |
| bridge | TOQB's `s2rc_device submit --local` runs a psf-os job folder, and psf-os scores what it writes |
| S1-S6 | sentinels: the user's circuits byte-identical with or without them; CLEAN without noise; a coupler planted after placement is flagged DEAD with a run around it; the finding is gray for that day only and adds nothing to LISTED; the time-model cap stops an over-long job; no map is read |
| Q1-Q4 | practice problems: each exact answer agrees by two independent methods; full score without noise; with planted two-qubit depolarizing errors the cluster parity and P1's read stabilizers are better on the avoiding ring; no map is read |
| ported | 31 of TOQB's tests (defect_list, safe_loop, qca, kising_echo, echo_tomo) on the ported code |

## Commands

```
psf-os fetch --backend fake:FakeFez --out DEV
psf-os plan --device DEV --job job.json --registry REG.json --out RUN     # no scan today -> a scan job first
psf-os submit --job RUN/scan/job --local                                  # fake devices only in M5
psf-os scan-ingest --device DEV --job RUN/scan/job --runs RUN/scan/runs --registry REG.json --out REG2.json --label s1
psf-os plan --device DEV --job job.json --registry REG2.json --out RUN2   # avoid, place, assemble
psf-os status --registry REG2.json
psf-os result --job RUN2/job/job --runs RUN2/job/runs
```

`job.json`: `{"kind": "sh6", "n": 40, "D": 12, "shots": 1000}`. Kinds: echo, sh6, qca, and the practice problems
`clifford_stab` (`k`, `seed`, optional `offset`, `max_weight`: P1, the random Clifford circuit read through a set of its output stabilizers with disjoint supports, every answer +1 or -1; `clifford_random`, read site by site, is kept for the records but cannot see a defect), `cluster_parity`, `kising_forward` (`L`, n <= 22), whose exact answers are recorded
in the job's record at build time and scored by `psf-os result`.

## Not in M5

The IBM submission inside psf-os (`submit` without `--local`; for now jobs go through TOQB's `s2rc_device submit`,
owner's decision of 2 November, always with `--stay` (TOQB 904e78f) so that a job is never moved to another device; if the job's own device has a failed element under it, nothing is sent and the job is planned again from a new fetch. A result from another device than the one the job was placed for is refused), repair discovery (M7), the Rust core
(M8), a resident service.

Code ported from TOQB (same rights holder) says so at the top of each file.

AGPL v3. See `LICENSE`.
