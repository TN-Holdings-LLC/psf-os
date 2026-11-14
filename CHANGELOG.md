# Changelog

## 0.1.0 (2026-11-14)

The first release: the code as used for the declared device tests of 2026-11-01 to 2026-11-13 (records in
[psf-os-lab-notebook](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook); papers 6 and 7 of TN Holdings).
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
