# psf-os 069463b (19c): what the workplace VS found (2026-11-19, instruction "その3")

The full check is in the lab notebook: `notes/os_loop_2026-11-19/vs/VS_CHECK_19c.md` (commit ba830519 on c806f8a0).
Nothing in `psf_os/` or `tests/` is changed.

- **verify across machines: a false DIFFERENT remains.** One dry night was built on Windows (Python 3.11, qiskit-aer
  0.18.0, numpy 2.4.6) and on a pod with the owner's versions (Python 3.12, qiskit-aer 0.17.2, numpy 2.5.3). Each
  night verifies SAME on its own machine, and DIFFERENT on the other: every job "rebuilt: /cells[0]/run_sha256".
  - The test circuits' gates are the same in all five jobs. Every re-score, rule, prediction and re-simulation is
    SAME.
  - `same_job` leaves out the top-level byte hashes when the bytes differ, but not the per-cell ones
    (`cells[*].qpy_sha256`, `cells[*].run_sha256`).
  - `verify_cells_19c.patch` (not applied) leaves those out too. With it, the other machine's night verifies SAME,
    and a copy with one rz angle moved by 0.01 still reads "DIFFERENT r1 job rebuilt: test circuits differ (gates)".
- **collect as values**: `job_state` and `ibm.collect` (with a fake service) are as written. In the loop, UNKNOWN,
  DONE without raw counts and a missing collect_status.json stop the night not resumably. Since a job with a
  submission.json is never sent again, these could stop resumably; only FAILED needs to end the night.
