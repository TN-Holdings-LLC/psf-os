# exact_tn acceptance, 2026-11-11 (workplace; criteria fixed in home's instruction before running; tolerance 1e-10)

Machine: 2 CPUs, 8 GB; runs under `ulimit -v 6500000`. Files: `accept.jsonl` (checks 1, 2 for Z106 and weight-10, 3,
reach; 07:16-07:58 CET), `accept_b.jsonl` (3b and weight-17; 07:57-08:32), `w17_k11_retry.json` (k 11 with
max_size 2^25), `w17_k4_k11_after_fix.jsonl` (k 4 and 11 after the change that searches the other form when the first
stays wider than 2^20).

| check | result |
|---|---|
| 1. cards up to 20 sites, tn against statevector (5 cards incl. kicked Ising 3 steps, merge on / off, 3 extra X / Y / two-site observables each) | worst 1.7e-13: PASS |
| 2. Eagle, Z106 4 steps, 17 points, against TOQB's released answers | worst 1.2e-13: PASS |
| 2. Eagle, weight-10 5 steps, 17 points | worst 1.6e-14: PASS |
| 2 (optional). Eagle, weight-17 5 steps, 17 points | 15 points worst 2.7e-14 in the run; k 4 refused (cost guard 10^12 flops, the fused form at 2^27 taken) and k 11 MemoryError; both after the fix: 6.3e-20 and 3.7e-16 (width 2^17 and 2^19). PASS after the fix |
| 3. kicked Ising 4x6x3 (72 sites), t 4, theta 0 and pi/2, every site, against Aer's stabilizer method | worst 2.7e-64: PASS (at pi/2 every <Z> is 0) |
| 3b (added). 6 stabilizers of the pi/2 state, weight 15-31, exact +-1 | worst 1.4e-14 |
| 4. tests | test_exact_tn.py 6 passed, 1 heavy skipped; the whole suite 209 passed, 18 skipped before the last two changes; test_exact_tn / card / kim / ising2d 83 passed after them |

Reach (not a criterion; kicked Ising 4x6x3, theta 0.8, site 46): t 3 width 2^10, 20 s; t 4 width 2^25, 47 s
(contraction 2.5 s); t 5 and t 6 refused by the memory guard (orders found 2^60 and 2^109; search 62 s and 131 s):
not reached with this search, which is not a proof that no narrower order exists.
