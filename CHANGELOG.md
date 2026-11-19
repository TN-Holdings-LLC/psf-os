# Changelog

## 0.2.0 (2026-11-19)

Real-time control on IBM Heron: circuits that measure in the middle and act on the result while they run, the build
rules the devices taught, and a loop of rounds that chooses each round's prescription from the previous round's
scores. Everything below was used by the declared device tests of 2026-11-16 to 2026-11-19 (records in
psf-os-lab-notebook; paper 10 of TN Holdings) or is the port of their tools. Nothing reads a calibration quality
number: placement uses the coupling graph and the elements marked as failed, and the loop's ledger reads the device
record's calibration time stamp only. Single builds of 0.1 jobs are unchanged.

- **Mid-circuit control** (`psf_os.feedforward`, 2026-11-16): a job kind for circuits with mid-circuit measurements and
  `if_test` blocks (form `blocks`: fixed sub-circuits of one-qubit gates and neighbour CZ, as IBM runs them), placed on
  a rule-C chain, with the off arm (the conditionals removed, or given explicitly) and a floor arm in the same job; the
  answer card counts every shot. Repeat-until-success chains (`rus_chain`), top-level delays that hold a schedule,
  placement given from outside (`chains=`), bundles of independent items on one chain (`build_bundle`, with `gap` and
  `bit_order`).
- **Register conditions** (2026-11-18): `if_test` on a whole classical register (at most 32 bits, IBM's limit), several
  tests on one register (a table look-up), registers that partition the bits in order; `rep3_lookup`, the distance-3
  repetition code with a look-up decoder before a T. Every job with conditionals records its conditionals, condition
  bits and broadcasts in the manifest (IBM's count as read from its text), never enforced.
- **Build rules** (2026-11-18), from what the devices did: `build(..., pack_fixed=True)` packs the fixed arms of many
  items into one circuit each (they read as they read alone, and the device time halves); `build_bundle` refuses two
  items with conditional operations side by side unless `allow_conditional_neighbours=True` (the control arm lost 0.20
  with ten items and 0.06 on the first of two); every manifest records the build environment (Python, qiskit,
  qiskit-aer, qiskit-ibm-runtime, numpy) and `job.verify_env` compares it.
- **The compounding loop** (`psf_os.loop`, 2026-11-19), moved from the notebook: the rule (R1 overshoot, R2 frames, the
  rungs +M then +F, R3 stop-loss, R4 keep; each threshold 2 SE), the ledger (used only on the same device with the same
  calibration time, per key only where the night's own diagnostics pick the same family; each line names its night
  with the date), and the night runner (fetch, diagnostics, rounds with the budget checked before a round is built,
  check, submit, collect, score, rule; `resume` never sends a job that was sent; only a FAILED job ends a night, any
  other state stops it resumably). The experiment (circuits, scores, first rule) is given from outside; the notebook's
  `gpu/os_loop.py` is the one for the RUS chain.
- **Verify across machines**: `verify` compares the QPY bytes, and when another qiskit / numpy / Python wrote different
  bytes, the gates (`gatehash.circuits_digest`: conditional blocks opened, RZ in (-pi, pi], parameters to 1e-10), with
  the test circuits and the ring sentinel apart and the manifest's per-cell byte hashes left out; a record the reader's
  qiskit cannot read (a newer QPY format) is NOT CHECKED, neither SAME nor DIFFERENT. Verify with a qiskit at least as
  new as the one that recorded. `psf_os.ibm.collect` writes the job's state as a value (`collect_status.json`,
  `ibm.job_state`).
- Packaging: only `psf_os` is installed (`[tool.setuptools.packages.find]`); the repository's `vs_check/` holds the
  workplace VS's check scripts and is not part of the package.

Tests: 334 passed, 1 skipped (with psf-os-lab-notebook checked out as `TOQB_DIR` and the optional pieces installed);
the heavy ones are marked (see the README).

`.zenodo.json` links this release to papers 9 (doi:10.5281/zenodo.24278888) and 10 (doi:10.5281/zenodo.24367001) as
well.

## 0.1.1 (2026-11-14)

The same code as 0.1.0. Added `.zenodo.json`, so that each GitHub release is archived on Zenodo with a DOI, with
TN Holdings as the author and the release linked to papers 6 and 7.

## 0.1.0 (2026-11-14)

The first release: the code as used for the declared device tests of 2026-11-01 to 2026-11-13 (records in
[psf-os-lab-notebook](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook); papers 6, 7 and 8 of TN Holdings).
Nothing in it reads a calibration quality number: placement uses the coupling graph and the elements marked as failed.

Rings (L0-L6): device, scan, registry (the defect list), allocate, repair, ring sentinel, job; `psf-os plan`.

2D circuits: placement by translation of the circuit's own pattern (`lattice2d`), the user entrance (`user2d`,
`psf-os run2d`), floor and mirror in the same job, the answer card and its extended report, `fetch` / `submit` /
`collect` (dynamical decoupling and twirling off; nothing is sent if a circuit touches a failed element).

Added 2026-11-07 to 2026-11-13:

- One entrance for a logical circuit (`psf-os entrance`): chains, 2D patterns and 4 x M x N cubic lattices recognised
  from the circuit alone by fixed rules (`entrance`, `entrance3d`).
- The Clifford twin (`twin`, `--twin`): the compiled circuit with every RZ rounded to a multiple of pi/2, the same
  pulses on Heron, exact values at any size; the kept share of the signal.
- Sentinel fallback (`--sentinel-fallback same_qubits`): the user's own qubits carry the sentinel when the rule's ring
  cannot be placed; the record says so.
- The quirk ledger (`quirk`, `quirk_chain`, `--special quirk_ledger`): the user's own records order placement
  candidates, as a declared special condition only, never reading days after `--before`.
- Chains round holes (rule C, `chain`), honeycomb strips (`strip`), Pauli-string observables (`observables`), exact
  values by a light-cone tensor network (`exact_tn`, optional), problem cards, a QPY-free circuit hash.

Tests: 286 passed, 1 skipped (with psf-os-lab-notebook checked out as `TOQB_DIR` and the optional pieces installed).

Not in 0.1.0: observables on the answer card (the module exists; the entrance does not use it yet); the twin and the
quirk ledger for 3D circuits; a declared self-repair test; the Rust core; a resident service.
