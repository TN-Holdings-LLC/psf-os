# Changelog

## 0.3.3rc1 (candidate, 2026-11-23; not tagged, not released)

Checks of the circuit as sent, and the device's lengths of time. Every circuit that 0.3.2 builds, 0.3.3rc1 builds byte
for byte the same (the six recorded nights of 2026-11-21 and 2026-11-22 rebuilt with both: every file the same); what
is new is checked and recorded beside the job.

### The name of a candidate

- **A candidate is named `X.Y.ZrcN`** (PEP 440) and becomes `X.Y.Z` only when it is released. 0.3.1's candidate
  (a6170e5, the likely value not parked by default) and its release (7caedaa, parked) both said "0.3.1"; this cannot
  happen again.
- Versions are compared in PEP 440's order (`feedforward.version_key`): `0.3.3rc1` comes before `0.3.3`. The safety
  checks of a declared version (`parks_by_default`, `parks_likely_by_default`) use that order, so a candidate is never
  taken for its release; `feedforward.same_version` compares the exact name, rc included.

### Added

- **The parking checked in the compiled circuit** (`psf_os.compiled`, used by `build`, `build_bundle` and
  `build_pack_fixed`). Every build follows each compiled arm noiselessly, branch by branch, through every wait (a
  mid-circuit read with the conditionals after it, or a run of conditionals on bits read earlier). A qubit that psf-os
  parked must stay parked in the compiled circuit for the whole wait:
  - parked because it was in |1> on every branch: P(|1>) 0 during the wait;
  - parked because it was more likely in |1>: P(|1>) below 0.5 during the wait.

  Otherwise the build is refused before anything is written. A wait is found in the compiled circuit by its anchors,
  which compilation keeps: each measurement by its bit and the how-many-th write to it, each conditional by its test
  and the how-many-th conditional with that test.
- **The waits as sent, in the manifest.** `manifest["compiled"]["records"]`, per record and arm: every wait with, per
  waiting qubit (physical), P(|1>) at its start and the largest during it, and whether it is entangled in any branch;
  the parked waits checked.
- **The fences' cost, in the manifest.** For an arm that psf-os fenced or parked, the compiled depth without and with
  them; with durations in the device record also the lengths in dt and µs.
- **A summary of every circuit as sent.** `manifest["compiled"]["summary"]`, per circuit: CZ, depth and the SHA-256 of
  its gate sequence (conditional blocks opened, RZ in (-pi, pi]: `gatehash`). `loop.verify` prints, beside each job's
  SAME or DIFFERENT, one line on what changed (`loop.summary_line`), from the manifests' summaries or, for a job
  recorded before 0.3.3rc1, from its circuits.
- **Lengths of time in the device record** (the owner's permission of 2026-11-23: lengths of time may be read, no error
  number). `Device.timing`, read by the fetch with `device.timing_of`: the duration of x, sx, cz, measure and reset per
  qubit or coupler, in dt, and the timing constraints (granularity, min_length, pulse_alignment, acquire_alignment).
  Only an instruction property's `duration` is read. `from_target` still reads failed marks only (T6 unchanged); the
  compiler still sees no duration (`map_free_target`). A record without timing builds as before and is written
  without the key.
- **Delays on the device's time grid.** With timing constraints in the device record, a delay whose length is not a
  multiple of the granularity and the pulse alignment, or is shorter than min_length, is refused before the job is
  written (`job.delay_grid_problems`).
- **The conditionals' waits estimated.** With durations in the device record, `compiled.lengths` gives each compiled
  arm's length and, per conditional, the measurement it reads and its longest block, in dt: a lower bound of the wait,
  the decision latency being in no record.

### Changed

- **The loop pairs by default on the device.** `loop.start(paired=None)`: a device night is paired (the incumbent and
  the candidate in the same job), since the same circuits drift between jobs (results/loopvar-night-2026-11-20: L 2,
  p 0.006). A device night that decides across jobs is refused unless `cross_job=True`, which run.json records. Dry
  runs keep their default; recorded nights verify and replay from their own run.json.
- `loop.same_job` leaves out a manifest key added after the recorded build (`compiled`), as it leaves out the build
  environment; the circuits are compared as before.

### Tests

`tests/test_checks_durations.py` (12). `TOQB_DIR` set: see the README.

## 0.3.2 (2026-11-23)

Time units and the charge for delays. On 2026-11-22 (results/ipe-branch-cost-night-2026-11-22) a test was declared
with delays written as a number of dt while dt was unknown. Read at run time, dt was 4 ns, so the delay per
conditional was 18 µs and each circuit idled 270 to 1008 µs. The job was charged 112 s against a model of 59.6 s,
because the model priced no delay. A job without delays builds and is priced exactly as in 0.3.1.

### Added

- **`dt` in the device record.**
  - What it is: `Device.dt` is the backend configuration's clock step in seconds. It is a unit of time, the same for
    every qubit, not a calibration or quality number.
  - Where it comes from: `ibm.fetch` and `fetch_fake` pass it to `from_target(..., step=backend.dt)`. `from_target`
    itself still reads failed marks only (test T6 unchanged).
  - Older records: a record without dt loads as before (`dt=None`) and is written without the key, so records before
    0.3.2 stay byte for byte.
- **Delays reach the job file in dt.** `job.delays_to_dt` rewrites every delay written in a unit of time (s, ms, us,
  ns, ps) in dt before the run records' QPY is written (`user2d._record`, `job.build`). qiskit 2.5.2's QPY keeps a
  delay's duration but not its unit: a delay in us came back as the same number of dt, and one in s was refused.
  - **Refused:** a delay in a unit of time without a dt in the device record, and a duration that is not a whole
    number of dt.
  - Delays already in dt are kept.
- **Delays recorded.** A job holding a delay gets the manifest key `delays`:
  - dt;
  - the limit;
  - per circuit its delay in dt and µs (the longest sum over one qubit);
  - the longest;
  - the seconds they add.

  A job without a delay has no such key.
- **The limit.** `DELAY_LIMIT_US` = 100 µs [OWNER] is the longest delay one circuit may hold. `job.assemble` refuses a
  job with a longer one before anything is written, unless `allow_long_delay=True`. The switch is passed by `build`,
  `build_bundle` and `build_pack_fixed`.
  - Without a dt, delays in dt are recorded in dt only, and are neither limited nor priced. This keeps the earlier
    records' rebuilds unchanged.

### Changed

- **The time model prices the delays.**
  - The formula: `qpu_seconds(n_circuits, shots, delay_seconds=0.0)` adds `shots × delay_seconds`, where
    `delay_seconds` is the sum of the circuits' delays.
  - Where it is used: `assemble` passes the delays, so `qpu_seconds_model`, the `max_seconds` refusal and the loop's
    budget (which reads `qpu_seconds_model` from build.json) all include them.
  - Without delays the value is the one 0.3.1 gave.
  - The night of 2026-11-22 rebuilt with its dt gives 59.6 + 54.8 = 114.4 s against 112 s charged (+2.1 %). The
    105-job relation (2.6 s + 0.000269 s per circuit-shot) with the same delay term gives 111.4 s. The base of the
    model is not changed here.

## 0.3.1 (2026-11-22)

The code sent to the device on 2026-11-22 (a6170e5, results/likely-park-night-2026-11-22) and the release commit on top
of it. The night read the declared rule's first row, so the release turns `park_likely` on by default. Every other
default is 0.3.0's (park, paired, strict_cz, replay, ...).

### Changed default

- **The likely value is parked by default** (`build`, `build_bundle`, `build_pack_fixed`: `park_likely=None` means
  `PARK_LIKELY_DEFAULT`, which is True).
  - Where `park` applies (not with `park=False`; never on a `park_skip` item, the unparked twin), every waiting qubit
    with a noiseless P(|1>) above 0.5 gets an X pair. The pair is behind a partial fence: the parked qubits and the
    wait's own qubits, at those waits only.
  - Checked at every build as `park` is: the noiseless distribution, and each arm's compiled CZ against the unparked
    arms. Otherwise the build is refused.
- **Rebuilding records.**
  - A recorded build without the "park_likely" key is remade without it (`park_likely_of_record`).
  - The loop's `verify` and a resumed night follow run.json's "park_likely" (absent: False); new nights record it.
  - `park_default(False)` sets both park defaults off, as before 0.3.0; `park_default(value, likely=...)` sets them
    apart.
- **The safety guard.** A declared psf-os (`declared=` or PSF_OS_DECLARED) without this default: a version before
  0.3.1, or a commit such as a6170e5. With it, a build that does not say `park_likely` is refused for any item that
  parking the likely value would change. An item it leaves alone is built as either version builds it.
  - The candidate a6170e5 also called itself "0.3.1", with the default off. Every build of it named `park_likely`
    explicitly: the night's tool passes the L items by name.
  - Rebuilt with this release, the night's job has the same gates in all 65 cells.
- psf_os_build records `park_likely` and `park_likely_set` when it applies.

### Evidence (L − D, the likely value parked against 0.3.0's default, in one job; IPE control beside)

| night | device | NF m3 L3 | P0 L3 | P3 L3 | IPE control |
|---|---|---|---|---|---|
| 2026-11-21 (results/rus-fence-night-2026-11-21; on the full fence: M − F) | ibm_fez | +0.0877 KEEPS MORE | +0.0140 KEEPS MORE | +0.0021 SAME | +0.0291 KEEPS MORE |
| 2026-11-22 (results/likely-park-night-2026-11-22; MP against D) | ibm_kingston | **+0.0972 KEEPS MORE** | **+0.0247 KEEPS MORE** | +0.0094 SAME | +0.0523 KEEPS MORE |

No item overshot. The declared rule, applied to the counted items (none with a KEEPS LESS, two with a KEEPS MORE, the
gate open), makes park_likely the default.

### Limits

- **Only the RUS chains and IPE.** Among the notebook's tools only IPE (always |1>, 0.3.0's park) and the RUS chains
  have a waiting qubit to park. The textbook circuit, the repetition code and decoy's sub-item have none.
- **MP has run on one night, one device** (ibm_kingston). The full-fence form ran on another (ibm_fez).
- The rule uses noiseless probabilities. On the items run, every parked qubit was at P(|1>) 0.675 or above.

### The candidate (from the branch likely-park-2026-11-22; its default was off)

- **`footprint.park_likely(qc, threshold=0.5)`** and **`build(park_likely=False | True | [tags])`**.
  - After `park`, every waiting qubit whose noiseless probability of |1> is above 0.5 (entangled ones too) gets an X
    before and after the wait.
  - The pair is kept by a **partial fence**: at those waits only, a barrier on the parked qubits and the wait's own
    qubits. Every other qubit is left to the compiler as without fences.
  - Checked as `park` is: the noiseless distribution, and each arm's compiled CZ against the unparked arms. A pair the
    compiler would have cancelled unfenced therefore refuses the build.
  - The records, checks.json and psf_os_build carry `park_likely`.
  - Without the argument everything is 0.3.0's, byte for byte.
- **Evidence:** results/rus-fence-night-2026-11-21 (ibm_fez). Parking the likely value on top of the full fence gave
  NF +0.0877, P0 +0.0140 (both KEEPS MORE). The full fence itself cost P0 −0.0453 and P3 −0.0243, so M against D read
  NF +0.079, P0 −0.031, P3 −0.022.
- **Desk** (notes/likely_park_2026-11-22, illustrative damping): the partial fence keeps most of the parking (NF
  +0.268 against D, of M's +0.324) and does not carry the full fence's effects. Never run on a device.
- Tests: `tests/test_park_likely.py` (7).

## 0.3.0 (2026-11-21)

The code is as sent to the device on 2026-11-21 at 11:00 CET (cedd90d; results/qpe-parked-night-2026-11-21);
the release commit changes CHANGELOG.md, README.md and .zenodo.json only.

What changed since 0.2.0, with the device record each change rests on. Every recorded build remakes as it was
built: a recorded build has no "park" key and is remade unparked; the version and park setting in the manifest
(`psf_os_build`) are left out when a rebuilt job is compared (`loop.same_job`). In the notebook, reproduce.py gives
every entry SAME with this version, and `verify` gives SAME for the loops of 2026-11-19 and 2026-11-20.

### Changed defaults

- **A waiting qubit in |1> is parked in |0>** (`build`, `build_bundle`, `build_pack_fixed`: `park=None` means
  `PARK_DEFAULT`, which is True).
  - Where: at every mid-circuit read and every run of conditionals. The waits are fenced as
    `fence_waits(conditionals=True)`.
  - What: each qubit that waits in |1> in every branch is put in |0> for the wait: X before the fence, X after.
    Qubits in |0>, carriers and branch-dependent qubits are left alone.
  - Checks at build: the noiseless distribution is unchanged, and the compiled CZ count equals the unparked one;
    otherwise refused.
  - The off arm is parked as the ff arm is.
  - The records, checks.json and the build report say where and what was parked, with excited before and after.
  - **Evidence:** results/weekend-shared-night-2026-11-21, test C. On IPE n = 4, waiting in |0> (pk) against the
    superposition as sent before (pl) gave +0.0387 (2 SE 0.0045), KEEPS MORE. Waiting in |1> (fe) against pl gave
    -0.0042 (2 SE 0.0049), SAME WITHIN ERROR.
  - **Run as the default:** results/qpe-parked-night-2026-11-21 (psf-os at cedd90d, another calibration). The parked
    IPE against its unparked twin (`park_skip`) in the same job: +0.0349, +0.0312, +0.0386 at n = 4, 5, 6 (2 SE about
    0.0075), KEEPS MORE at every n; the parked IPE beyond its ceiling at every n; the parked loss per added bit FLAT.
    **Two nights, one device, one circuit family (IPE).** The switch stays.
  - Among the notebook's tools only IPE changes: every wait parks its target, and compiled depth grows by up to +16 at
    n = 6. The RUS chains, the textbook circuit, the repetition code and decoy's rd2 have nothing to park
    (notes/park_default_2026-11-21).
- **A test that compares with the unparked form says `park=False`** (the owner's rule of 2026-11-21).
  `build(park=True, park_skip=[tags])` builds the named items unparked in the same job (the parked and unparked
  form of one circuit side by side; their records say "park": {"park": false, "skipped": "park_skip"}).
- **The safety guard.** A tool declared on another psf-os names it: `declared=` or the environment variable
  `PSF_OS_DECLARED`. If that version's park default differs from this one's, a build that does not say `park` is
  refused before anything is written. Any commit, or a version before 0.3.0, counts as not parking. For example, a
  tool declared on a164fc5 and built with 0.3.0 must say park=False to build the IPE it declared.
- **Every manifest names its build** (`psf_os_build`: version, park, how park was set, declared).
- **run_local's seeds** are kept apart per circuit and part (after-2026-11-20; `stride=1` gives the 0.2.0 seeds).

### Added, off by default

- **Record-only checks** (`checks.json`, on by default, outside the job): CZ as written and as compiled, and the
  footprint as sent and as written. `strict_cz=True` refuses a difference; it is used only when chosen.
  - **Evidence:** the repetition code of 2026-11-19 (results/repcode-night-2026-11-19) ran with 16 CZ on the
    error-on-d0 items but 14 on the other three. The compiler cancelled a CX pair (notes/repcode_2026-11-19_cz.md).
    No verdict changed; the pooled share mixes 14 and 16 CZ.
- **The wait footprint** (`psf_os.footprint`): carriers and excited qubits at each wait, exact over the branches.
  - **Evidence:** test A of the weekend. On the repetition code, fewer carriers at equal CZ (F6P - F10) gave
    +0.0153 (2 SE 0.0081), KEEPS MORE. Fewer CZ at equal carriers (F6 - F6P) gave +0.0224 (2 SE 0.0077), KEEPS MORE.
    The forms also differ in depth (a declared limit).
- **Fences** (`build(fence=True)`, `fence_waits`): the compiler cannot merge gates across a wait (on IPE it had held
  the target in a superposition).
- **The paired loop** (`loop.next_stacks_paired`, `start(..., paired=True)`; used only when chosen): the incumbent and
  the candidate in one job, decided by their difference there.
  - **Evidence for the need:** results/loopvar-night-2026-11-20. Copies inside one job agree; jobs minutes apart
    moved at L 2 (p 0.006, excess SD 0.014).
  - **Run once:** results/loop-paired-night-2026-11-20, 110 s, two rounds, every decision inside one job, verify SAME.
- **`build(ff_only=[...])`**: those items' ff arm only. The paired loop uses it, and so did the weekend's shared night
  (111 s for three tests).
- **Replay** (`start(..., replay_from=RECORD)`, dry runs only): a night answered from a recorded night's raw counts.
  - It is strict: every cell must come from one recorded job (the owner's decision of 2026-11-21). Otherwise it stops
    with "記録にない" and names the missing cells and the closest recorded job. A paired decision read across two jobs
    is not the loop's decision.
  - A replayed night takes the recorded night's park setting.
  - A broken ledger (not JSON, a bad line) is not used, and the reason is written. Before, the night stopped with a
    Python error.
- The ring sentinel over the test (`sentinel_over_test=True`), `ring_chains`, and receivers for the decoy and loopvar
  tests: as in the branch notes below.

### Tests

384 passed, 1 skipped (the heavy exact_tn case), with the notebook as TOQB_DIR; reproduce.py 88 of 88 SAME (89 with
the parked night of 2026-11-21); `verify` SAME for the loops of 2026-11-19 and 2026-11-20. Run by the home VS and again
at home. New in 0.3.0: `tests/test_park_default.py`,
`tests/test_release_030.py`, `tests/test_loop_replay.py`, `tests/test_loop_paired.py`. `test_footprint`'s fence test
pins park=False, since it is about the unparked IPE.

### The branches merged into 0.3.0 (their notes, as written)

### The branch park-default-2026-11-21 (from loop-paired-2026-11-21 a164fc5)

A waiting qubit is parked in |0> by default (the owner's decision of 2026-11-21 9:59 JST). The weekend's test C
(results/weekend-shared-night-2026-11-21, IPE n = 4) gave: waiting in |0> keeps +0.0387 (2 SE 0.0045) more; waiting
in |1> is the same as the superposition (-0.0042). That is one night on one device, so the switch stays.

- **`psf_os.footprint.park_waits(qc)`**:
  - It finds every wait: a mid-circuit measurement group or a run of conditionals, fenced as
    `fence_waits(conditionals=True)`.
  - It puts in |0> each qubit that waits there in |1> in every branch: X before the wait's fence, X after the fence
    that closes it. The fences keep the pair from cancelling and keep gates out of the wait.
  - It leaves alone qubits in |0>, carriers (superposition or entangled) and qubits whose basis value depends on the
    branch.
  - States are counted as `wait_footprint` counts them (branching, exact).
  - The exact distribution of the classical bits must be unchanged (`exact_distribution`); otherwise ValueError.
  - When nothing is parked, the circuit itself is returned, unfenced.
- **`feedforward.build(park=None)`**:
  - None means `PARK_DEFAULT`, which is True. The context `park_default(value)` sets it for a block.
  - The ff and off arms are parked, and each compiled CZ count must equal the unparked one; otherwise refused.
  - With park on, records, `checks.json` and the build report carry "park": where, which qubits (logical and
    physical), and excited before and after.
  - With `park=False` nothing is added, byte for byte as before. An item with nothing to park is built as without
    parking.
  - `park_of_record(rec)` gives False when the key is absent.
- **`loop`**:
  - `start` records "park": true in run.json for new nights.
  - `Night.go` and `verify` build under run.json's "park", False when absent. So a recorded night is rebuilt as it
    was built, and a resumed old night goes on as it began.
- `build_bundle`, `build_pack_fixed` and the other builders are unchanged (no parking).
- Tests: `tests/test_park_default.py` (10).

### The branch ledger-replay-2026-11-21 (from loop-paired-2026-11-21 a164fc5; not declared)

The ledger's path checked without the device (the home instruction of 2026-11-21, the owner's decision of 2026-11-21
0:3x JST). Off by default: a night without `replay_from` builds, runs, rules and verifies as before.

- **Replay** (`psf_os.loop.start(..., replay_from=RECORD)`, dry runs only, never with noise):
  - Each job of the dry night is answered from a recorded night's raw counts (`replay_job`) instead of a simulation.
  - A test circuit is looked up by its cell name and its gates (`psf_os.gatehash`). The ring sentinel is looked up by
    its name only: it is never judged, and its synthesis follows the versions.
  - The recorded job that holds most of the job's cells is the primary source. A cell it lacks comes from another
    recorded job, and `replay.json` names the source of every cell.
  - A circuit the record does not hold stops the night: "記録にない", not resumable, nothing written into the job.
    There is no fall back to a simulation.
  - A replayed night's builds are the device night's (`Night.build_dry`: no dry Clifford sentinel).
  - `verify` answers every replayed job again from the record and compares the counts.
- **A broken ledger is not used and is named** (`ledger_load`, `ledger_start`). Before this branch, such a ledger
  stopped the night with a Python error. These cases now give `why` "ledger unreadable (...)", or a `per_key` reason:
  - a file that is not JSON or not a list;
  - a last line that is not a night;
  - `last_stacks` that is not a table;
  - a key whose stack is not a list of layers.
  A good ledger gives the same decision as before.
- Tests: `tests/test_loop_replay.py` (7).

### The branch loop-paired-2026-11-21 (from after-2026-11-20; a prototype, not declared)

The loop compares inside one job (the home instruction of 2026-11-21, the owner's decision of 2026-11-20). Everything
is off by default: a night without `paired` rules, runs and verifies as before, and `build` without `ff_only` is byte
for byte as before.

- **The paired rule** (`psf_os.loop.next_stacks_paired`; `Night` with the cfg `paired`, `start(..., paired=True)`):
  per key an incumbent and a candidate in the same job.
  - The decision (`paired_decision`) is d = |P_inc - exact| - |P_cand - exact|, with SE from both ff arms. The
    candidate is adopted at d >= 2 SE; it is not adopted at d <= -2 SE or when it overshoots; between, the incumbent
    is kept.
  - The next candidate (`paired_next_candidate`) is the next untried step: B after an overshoot, the diagnostics' base
    on P0, R2's Q on B, then the rungs M and F. A step that was not adopted is not tried again that night. A key with
    no step left stops.
  - Round 1 (`paired_rule1`): every incumbent is P0, and the candidate is the ledger's stack or the diagnostics' stack.
  - `verify` re-rules round 1 and every rule of a paired night.
  - The ledger's line for a paired night holds the incumbents after its last decision.
  - The option `climb_on_kept` (default off): on INCUMBENT KEPT, the next candidate is built on the kept candidate.
- **`build(ff_only=[tags])`**: those items get their ff arm only (no off, no floor), and their records carry
  `ff_only`. In the paired loop this is the incumbent's ff beside the candidate's three arms.
- Tests: `tests/test_loop_paired.py` (9).

### The branch after-2026-11-20 (not released; not used by the declared tests of 2026-11-20)

What the circuits look like as they are sent, recorded by default; what the compiler may do around a wait, chosen by
argument. No default changes a job: `job/`, `runs/` and `build.json` are byte for byte as in 0.2.0 (but run_local's
seeds, below), so every recorded build reproduces as before. Nothing reads a calibration quality number.

- **Record-only checks, on by default** (`build` / `build_bundle`, `record_checks="record"`; `"off"` writes nothing):
  every build also writes `OUT/checks.json`, beside the job and outside it. Per item and arm it holds the CZ count as
  written and as compiled; a difference is noted on stderr and never refused (`strict_cz=True` refuses). It also holds
  the wait footprint as sent and as written, or `counted: false` with the reason when a circuit is above the
  footprint's bounds (14 active qubits; 2^20 branch amplitudes). The build never waits on it. A tool that builds
  in a temporary folder and assembles the job itself (the notebook's decoy and interfere tools) does not keep it; the
  job is the same either way.
- **The wait footprint** (`psf_os.footprint`): at each mid-circuit read, the waiting qubits not in a computational
  basis state (carriers) and the expected number in |1> (excited). It is computed exactly over every branch of the
  circuit, as written or as sent, at the read or over the whole wait (`span=True`), and at runs of conditionals
  decided on bits read earlier (`conditionals=True`: IPE's phase feedback; the checks count these too). `build(footprint=True)` writes it
  into the records (this changes the records; off by default).
- **Around a wait** (off by default):
  - `build(fence=True)` puts a barrier on every qubit before each mid-circuit read and after each run of
    conditionals; `fence_waits(qc, conditionals=True)` also before each run of conditionals that does not follow a
    read. The compiler then cannot merge gates across the wait: compiled IPE otherwise held its target in a
    superposition while it waited (with the reads fenced only, still at its phase-feedback conditionals).
  - `build(strict_cz=True)` refuses a compiled CZ count other than the written one (the level-1 compiler cancels a CX
    pair with nothing between them; a barrier keeps it).
- **The ring sentinel over the test** (off by default): `build(..., sentinel_over_test=True)` prefers, at the same ring
  length, a ring that holds the test's qubits, and records the cover. `place.rule_choice(prefer=...)` and
  `RuleCompiler(dev, prefer=...)` support it. `ring_chains(dev, n)` lists chains on the sentinel's own ring.
- **Receivers for the decoy and loopvar tests** (off by default):
  - `build_bundle(behind_fixed_first=True)` allows conditional neighbours behind a fixed first item.
  - `loop.rejudge` and `next_stacks(between_job_sigma=...)` add a between-job term to the loop's threshold.
- **run_local's seeds** (the one changed default): circuit k, part j at seed + w (64 k + j), w = max(10,000, shots + 1),
  so no two runs share a per-shot seed. Aer seeds shot i with seed + i, so the 0.2.0 seeds gave neighbouring circuits
  the same shots shifted by one. `stride=1` keeps the 0.2.0 seeds and reproduces recorded dry runs. Of the notebook's
  tools that build on this version, 13 of 14 noiseless dry runs read the same words. balanced_night now reads its true
  shot spread: the 0.2.0 seeds had hidden it.

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
