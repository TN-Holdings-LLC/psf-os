# Register conditions: independent check by the workplace VS Claude (2026-11-18, home instruction 18h, item 1)

Branch `register-cond-2026-11-18` at 37136e9 (on 0f57c76), as received; nothing in `psf_os/` or `tests/` is changed.
Desk only, no IBM. Environment: qiskit 2.5.2, qiskit-aer 0.18.0, Python 3.11.9, Windows. Device record for the
builds: the notebook's `results/balanced-night-2026-11-17/fetch_in/ibm_marrakesh/dev` (coupling graph and failed
marks only).

The functions under test are called. What decides PASS or FAIL is written here with qiskit and qiskit_aer only: no
helper of psf-os, and no test of the branch.

| file | what |
|---|---|
| `vs_regcond.py` | S1-S6 below; result `vs_regcond.json` (33 checks, 0 FAIL) |
| `vs_bytes.py` | single-bit builds under 0f57c76 and under the branch, byte for byte; result `vs_bytes_compare.txt` |
| `vs_mutants.py` | the branch's own tests (`tests/test_register_cond.py`) against ten damaged copies of `psf_os`; result `vs_mutants.json` |

## Results

**S1 refusals (spec 1): all as specified.** `analyse` refuses each of these with its reason:
- a 33-bit register;
- a register with a bit not yet measured;
- the value 4 for 2 bits (qiskit itself refuses -1 when the circuit is built);
- two expressions (`s == 1`, `s[0] & s[1]`);
- a measurement in the block, a reset in the block, nesting, and an else branch;
- overlapping registers.

A 32-bit register with the value 2^32 - 1 is accepted.

**S2 look-up decoder (specs 2 and 5).**
- The table read from the circuit's instructions is {1: d0, 3: d1, 2: d2}. It equals the one derived from the
  parities a0 = d0 + d1 (s[0]) and a1 = d1 + d2 (s[1]).
- Noiselessly, with no error and with an X on each data qubit, c = 000 in every shot.
- Compiled by `build` on the marrakesh record:
  - every ff arm is right in every shot, and every floor is all zero;
  - every compiled register test reads bits [0, 1] in s's order, and the registers stay s (2 bits) then c (3 bits).

**S3 `restrict` (spec 3, point 1).**
- With the clbits in the orders [0..4], [4..0] and [1, 0, 4, 2, 3], the rebuilt register holds s's bits in s's order
  at their new places, and the decoder stays right for every error.
- Leaving out a register bit is refused.

**S4 `bundle_acceptance` (spec 3, point 2).**
- A register test whose register holds a bit of another sub-item is refused.
- A register test inside one sub-item is accepted.
- A bundle of two look-ups compiled end to end: the registers partition the bits in order, the tests read (0, 1) and
  (5, 6), and both are right in every shot.

**Single-bit builds, byte for byte (point 3).** Under 0f57c76 and under the branch:
- the notebook's phase-estimation parts A and B;
- a job of three `teleport_test` items;
- a bundle of two `teleport_test` items.

All four `circuits.qpy` are the same bytes. The manifests are the same apart from `written`, and the branch adds
`broadcast`. Part A is `febee21a…`, the same as the VS builds of the morning.

**S5 broadcast record (spec 4): agrees with a hand count.** The look-up job of 16 circuits:
- per circuit, ff and floor arms (5 conditionals, 8 bits), off arms (2, 2), sentinel (0, 0);
- for the job, 48 conditionals and 72 bits; the hand count is the same.

**Mutations (point 3 of the request).**
- Circuits (S6): the d0 test's value shifted from 1 to 2, or the two syndrome measurements exchanged. Both are wrong
  for errors 0 and 2, and both are caught twice:
  - by the noiseless check of the compiled job;
  - by psf-os's own `run_local` and `cards` (share 0 or 0.501).
- Code: the branch's 11 tests against ten damaged copies of `psf_os`. Nine are caught:
  - the value shift;
  - the syndrome bits exchanged;
  - `restrict` reversed;
  - the value 2^width accepted;
  - 33 bits accepted;
  - a late bit accepted;
  - overlapping registers accepted;
  - `wire_signature` reversed;
  - `broadcast_record` counting a register as one bit.
- The tenth (`walk_bits` listing only a register test's first bit) is not caught, but it is an equivalent mutant.
  Qiskit already puts a register condition's bits in the instruction's clbits, so `walk_bits` still lists every
  bit. The S4 case is still refused under it, with bits [0, 1, 0]. So this is not a gap in the tests.

**Full test suite of the branch:** started on this machine (`pytest -q -x tests`). It had not finished after about an hour when this was handed back; its result will follow separately. The workplace reports the existing tests passing.

## Points for the workplace Claude and the owner (not changed here)

1. **The record called `broadcast` counts conditionals, not IBM's broadcasts.** IBM counts tests on an unchanged
   register as one broadcast ("Execute dynamic circuits").
   - Read that way, the look-up's ff arm is 1 broadcast of 2 bits. The record says 5 conditionals and 8 bits,
     because the two single-bit resets read bits already sent with s.
   - So the record is an upper bound on IBM's count. That is safe for a limit, but the name and the note could say
     so.
   - `vs_regcond.py`'s `ibm_broadcasts` gives the other reading.
2. **`rep3_lookup`'s docstring and its code disagree.** The docstring says "Without the look-up and the resets
   (feed_forward=False ...)". The code keeps the two single-bit resets when `feed_forward=False`: they sit outside
   `if feed_forward`. The branch's test asserts these 2 conditionals.
   - Only the default off arm (`strip_conditionals`) removes every conditional.
   - The test items pass `rep3_lookup(e, feed_forward=False)` as an explicit `off_circuit`, so that off arm keeps
     the resets.
   - Either the docstring or the code should change. Which one is for the workplace Claude.
