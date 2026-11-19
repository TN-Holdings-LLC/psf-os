# psf-os v0.2-proto at 11f07b0: what the workplace VS checked (2026-11-19, workplace instruction "その2")

The full check, with the dry nights, the scenarios and the repetition code, is in the lab notebook:
`notes/os_loop_2026-11-19/vs/VS_CHECK.md` and `notes/repcode_2026-11-19/vs/VS_CHECK.md` (commit 4f0c16e3 on 65c7d3fa).
Nothing in `psf_os/` or `tests/` is changed.

- **Test suite** on a pod (Linux, Python 3.12, qiskit 2.5.2, qiskit-aer 0.18.0, qiskit-ibm-runtime 0.51.0, 11
  workers): 295 passed, 30 skipped, 0 failed in 17 min 38 s (`pod_pytest_tail.txt`).
  - The 30 skips are environmental: no notebook on the pod (TOQB_DIR), no quimb or cotengra, PSF_OS_HEAVY unset.
- **The notebook-dependent tests**, run on Windows with TOQB_DIR: 13 passed, 2 failed.
  - `test_entrance::test_chain_of_equals_the_notebook_builder_on_every_record`: the notebook's `hub_cost.py` imports
    the Unix-only `resource`. This is the machine; it needs a Linux run.
  - `test_feedforward::test_pack_fixed_on_tonights_pa_record_gives_the_same_ff_bytes`: the same QPY format byte, but
    different bytes. The instructions are the same in all ten items; a few angles differ in their last bits (at most
    1.8e-15), and so does the global phase (a few ulp, and 0 against 2 pi). The record comes from the owner's machine
    (Python 3.12, numpy 2.5.3).
  - The test's rule "the same format byte means the same bytes" does not hold across numpy / Python versions.
    Comparing the gates, the angles and the global phase (modulo 2 pi) to about 1e-12 whenever the bytes differ
    would keep it meaningful.
- **`psf_os.loop.next_stacks`** on hand-made rounds (`vs_rule.py`, `rule_checks.txt`): 14 of 14.
  - R1 and R2 come before the rungs; +M comes before +F (and F first when the rungs are given so).
  - F needs a gap against the base's frame, and no rung climbs with B.
  - R3 takes F off with the base switched back, and takes M off.
  - The night stops at a fixed point, at the round limit and on the budget.
- **Points** (not changed):
  - RF does not look at the verdict, so it can come before R3 on a key NOT SHOWN twice.
  - `start(k_max=0, budget=0)` falls back to the defaults.
  - A dry night stops on a missing result as "not finished after 60 min", though nothing waited.
