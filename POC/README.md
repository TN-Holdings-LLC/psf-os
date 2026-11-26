# psf-os proof of concept: running circuits on IBM devices without the calibration map

**In one sentence.** psf-os places a circuit on an IBM heavy-hex device without reading any calibration quality
number, tests the device with short circuits of its own in the same job, keeps a qubit that waits in |0> rather
than |1>, makes real-time decisions by comparing inside one job, and attaches to every result a report of how far
that result can be trusted.

This page is for teams considering a proof-of-concept evaluation. Researchers will find the full records in the
papers and in the lab notebook, [psf-os-lab-notebook](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook) (links
below).

**Status.** psf-os 0.3.3 (2026-11-24; [doi:10.5281/zenodo.24496267](https://doi.org/10.5281/zenodo.24496267)),
AGPL-3.0, used by its authors; no wide user base has tested it yet. Every device number below comes from a test
declared in git (hash and time locked) before its device record existed.
<!-- src: psf-os pyproject.toml version 0.3.3; CHANGELOG.md "## 0.3.3 (2026-11-24)"; notebook notes/paper11/ZENODO.md
     (psf-os 0.3.3 doi:10.5281/zenodo.24496267) -->

## What it does

- **No calibration map.** Placement reads the coupling graph and the elements marked as failed, never an error rate:
  the same circuit lands on the same qubits whatever the day's calibration says, so a run can be declared before the
  calibration is fetched.
- **Checks in the same job, reported, never used to correct.** A floor, a mirror of the compiled circuit, a ring
  sentinel and a Clifford twin run beside the user's circuit. The answer card says whether the result reads and how
  much of the signal the device kept.
- **Waiting qubits parked in |0>.** A qubit that waits through a mid-circuit measurement in |1> is turned to |0> for
  the wait and back after (0.3.0); one that is only more likely in |1> as well (0.3.1, behind a partial fence). From
  0.3.3, every build checks in the compiled circuit that a parked qubit stays parked.
- **Decisions made inside one job.** A loop that chooses between two circuits runs both in the same job on the
  device (0.3.3 default), because the same circuits drift between jobs sent seconds apart.
- **Refusal with a reason.** A circuit that touches a failed element, or whose shape psf-os does not recognise, is
  not sent, and the reason is given.

What it does not do: general circuits (only chains, rings, 2D patterns of a lattice, 4 x M x N cubic lattices and
the dynamic-circuit families of Papers 10 and 11, on IBM heavy-hex devices); error mitigation (so deep circuits often
come back as "does not read"); any quantum advantage (every answer below is computed exactly on a classical machine).

## What was measured on devices

Each item gives the headline number and, beside it, the number that says what the headline is made of. Records:
the papers listed at the end, and the notebook's `results/` folders named in the source notes.

- **Placing a ring around paths found dead** (Papers 5 and 6; ibm_kingston, ibm_fez, ibm_marrakesh, 31 October to
  2 November 2026). Headline: on ibm_kingston, usable sites of a 40-qubit ring 22 -> 40 on each of three days.
  Essential: on the sites that were already usable the signal moved from 0.41 to 0.44; the gain is in width, not
  in a better signal everywhere, and in three of ten rows the mean signal on usable sites fell.
  <!-- src: notebook results/coverage-days-2026-10-31-day1/coverage.json (n40 22 -> 40, mean_usable 0.410 -> 0.441),
       results/coverage-days-2026-11-01-day2/coverage.json, results/coverage-days-2026-11-02-day3/coverage.json,
       results/two-devices-2026-11-01/{fez,marrakesh}/avoid/coverage.json; Paper 6 numbers.json width.*;
       as in psf-zero POC/README.md at 8769aa5 -->
- **User-written circuits with known answers** (Paper 7; ibm_fez and ibm_marrakesh). Headline: 22 practice circuits
  written in plain Qiskit, all 22 READ on both devices. Essential: 19 and 18 of them AGREE within the declared
  |M - M_exact| <= 0.05, and that criterion is biased both ways: a problem that DIFFERS kept 0.82-0.86 of its signal,
  as much as one that AGREES (0.73-0.85). The retention is therefore printed beside every verdict.
  <!-- src: notebook notes/paper7/paper7.tex abstract ("all 22 READ on both; 19 and 18 AGREE"; "0.82--0.86";
       "0.73--0.85") -->
- **IBM's dual-unitary kicked-Ising chain, raw** (arXiv:2411.00765; Papers 7 and 8). Headline: at N = 91 on
  ibm_marrakesh every depth up to t = 24 (2,205 CZ) is readable and the decay rates agree with cos 2h at all three h
  (largest |k - cos 2h| 0.0037). Essential: at h = 0.15 the rate sat +0.0037 above cos 2h on both devices, and a
  light-cone test failed as predicted (every front arrived late, the shape of the spread was reproduced).
  <!-- src: notebook notes/paper8/paper8.tex abstract ("N = 91", "2\,205 CZ", "0.0037", "+0.0037", light-cone
       sentence); results/xx91-2026-11-12 -->
- **What decides whether a raw result reads** (Paper 9; 179 results with an exact answer). Headline: 160 of the 166
  results that keep at least 0.293 of the exact pattern read. Essential: below 0.293 a result cannot read by
  construction, so only the converse is measured; and dividing raw values by a measured share changed few verdicts
  and made some results worse.
  <!-- src: notebook notes/paper9/paper9.tex abstract ("179", "160 of" / "the 166 results with $r \ge 0.293$ read", "0.293") -->
- **Real-time control beyond any correction of the uncontrolled circuit** (Paper 10). Headline: iterative phase
  estimation at n = 4 on ibm_marrakesh, 0.903 against a correction ceiling of 0.471. Essential: of 57 declared
  predictions over four nights, 37 held and 20 failed (all reported); what the device keeps of the noiseless value
  varies by night (0.61-0.87 for the same balanced pair on two nights).
  <!-- src: notebook notes/paper10_ff/paper10.tex abstract ("0.903 against 0.471 at n=4"; "33 held and 16 failed";
       "4 held and 4 failed"; "0.61--0.87"); results/ff-night2-2026-11-17/README.md (marrakesh n = 4 0.903) -->
- **Rules for waiting qubits** (Paper 11; ibm_marrakesh, ibm_fez, ibm_kingston; 706 s charged). Headline: 11 declared
  predictions, all held; parking a qubit that waits in |1> kept +0.039 (2 SE 0.005) on phase estimation at n = 4.
  Essential: a prediction was declared only where an earlier device record grounded it, and over the series 20 of 68
  failed, all on Paper 10's nights; fencing a qubit so that it waits in |1> instead of a superposition read SAME
  WITHIN ERROR (-0.004, 2 SE 0.005) on the one night it was sent; the drift between jobs (p = 0.006, one of three
  chain lengths) is why 0.3.3 pairs by default.
  <!-- src: notebook notes/paper11/paper11.tex abstract ("706 s charged; 11 declared predictions, all held";
       "+0.039 (2 SE 0.005)"; "20 of 68 failed"; "-0.004, 2 SE 0.005"; "p = 0.006");
       results/loopvar-night-2026-11-20 (p 0.006, CHANGELOG 0.3.3) -->

**Limits.** Three IBM Heron r2 devices (ibm_fez, ibm_marrakesh, ibm_kingston) over November 2026; a few circuit
families, each placed by a fixed rule. Results vary by night and by calibration, and the papers report every
declared prediction that failed. The dead paths psf-os lists are paths; their cause is not claimed. Without error
mitigation, deep circuits often do not read; psf-os says so instead of correcting them.

## Try it in five minutes (a fake device, no IBM account)

```bash
git clone https://github.com/TN-Holdings-LLC/psf-os
pip install -e "psf-os[sim]"                              # behind a network that re-signs TLS, pip may need your
                                                          # organisation's certificate settings
python psf-os/POC/quickstart.py                          # writes ./quickstart_out
```

**This is a simulation with defects planted by the script, not a device result.** The script says so on its first and
last lines. It plants a two-qubit error on two couplers of IBM's FakeFez, scans twice, lists the planted paths, and
solves two practice problems with exact answers on the standard ring and on the avoiding ring (a cluster-state parity
and a random Clifford circuit read through its stabilizers). The scan runs at the Clifford point of the echo so that a
laptop can simulate 72 qubits; the device scan uses the full echo. It then shows two later defaults with psf-os's own
functions: a waiting qubit in |1> parked in |0> with the exact distribution unchanged (no noise is simulated for this
step), and a device night that would decide across jobs refused.

One run with psf-os 0.3.3, output unedited. It took 144 s on one machine (qiskit 2.5.2, Aer 0.17.2) and 774 s on
another (qiskit 2.6.2, Aer 0.18.4), each running other simulations at the same time; the output was the same on both
but for the time:

```
SIMULATION with planted defects on FakeFez -- not a device result
1. device record: fake_fez, 176 couplers, 7 failed, 0 failed qubits (no quality number read)
2. planted (known to this script only): couplers [(15, 19), (75, 59)], two-qubit depolarizing 0.15
3. two scans (24 circuits each); LISTED paths: [[14, 15, 19, 35], [74, 75, 59, 55]]
4. practice problems (exact answers recorded at build time), standard ring against avoiding ring:
   P2 cluster parity, n = 40: fraction right 0.920 (standard ring) -> 1.000 (avoiding ring); sentinel on the avoiding ring: CLEAN; masked couplers [[15, 19], [59, 75]]
   P1 random Clifford (k = 4), read stabilizers, n = 40: fraction right 0.980 (standard ring) -> 1.000 (avoiding ring); sentinel on the avoiding ring: CLEAN; masked couplers [[15, 19], [59, 75]]
5. parking (0.3.0 and later): 1 wait, qubit(s) parked [2]; waiting qubits in |1> (expected number) 1 -> 0; exact distribution unchanged: True
6. pairing (0.3.3 default on the device): a device night deciding across jobs is REFUSED (a device night that decides across jobs (paired=False) needs cross_job=True: the same circuits drift between jobs (loopvar 2026-11-20); the default on the device is paired)
7. done in 144 s on this machine (psf-os 0.3.3). SIMULATION with planted defects on FakeFez -- not a device result.
```

A perfect 1.000 here means that the planted errors were the whole of the noise and were avoided; on a device every ring has
its own noise, as the numbers above show. **Simulation, not a device result.**

## Run it on your own IBM Quantum account

psf-os uses the IBM Quantum account you have saved yourself with `QiskitRuntimeService.save_account(...)`. It never
reads, writes or asks for an API key, and neither do we: keep your key in your own saved account, never in a file, an
e-mail or a chat.

```
psf-os fetch --backend ibm_fez --out DEV                                  # coupling graph and failed marks only
psf-os entrance check --device DEV --circuit my.qpy                       # recognised? placed where? refused why?
psf-os entrance build --device DEV --circuit my.qpy --out RUN             # place, add the checks, assemble one job
psf-os submit --job RUN/job                                               # the job says "stay" on this device
psf-os collect --job RUN/job --wait-min 30
psf-os entrance score --out RUN --answers answers.json                     # answer cards; answers optional
```

(`psf-os run2d` and `psf-os result2d` do the same for a batch of 2D circuits; the main README lists every command.)

**What it costs in device time.** Two estimates, side by side:

- the formula fitted to our charged jobs: 2.6 s + 0.000269 s x circuits x shots;
- psf-os's own model (`job.qpu_seconds`): 0.036 s per circuit + 0.00027 s per circuit-shot + 3 s per job, plus the
  delays.

For example, 8 circuits with a 4-circuit sentinel at 1,000 shots: about 5.8 s by the formula and 6.7 s by the model.
The notebook's latest declared job, 169 circuits at 1,000 shots, came to 48.1 s by the formula and 54.7 s by the
model; IBM charged 47 s.
<!-- src: psf-os psf_os/job.py qpu_seconds; notebook prereg/ohm_ruler3_night_2026-11-25.md "2.6 s + 0.000269 s x
     circuits x shots" and its table ("169", "48.1", "54.7"); results/ohm-ruler3-night-2026-11-25/README.md
     ("47 s charged"); 12 x 1000: 2.6 + 3.228 = 5.83, 0.432 + 3.24 + 3 = 6.67 -->

## A proof-of-concept evaluation

1. Fill in the request form (`psf_os_poc_request_form.docx`): the shape of your computation, the device, who runs the
   jobs, and what you want measured.
2. If your computation is sensitive, we send the mutual NDA (`psf_zero_nda_template.pdf`) first.
3. We fix the test in writing before anything runs: what is run, what is measured, what would count as a result.
4. The result comes back in the report template (`psf_os_poc_report_template.docx`): standard placement and
   self-checked placement side by side, the headline number next to the essential one, and the limits.

Contact: love.os.architect@proton.me.

## Papers

All by TN Holdings, on Zenodo.

- Paper 11: Where a Qubit Waits: Rules for Real-Time Control from Declared Device Tests on IBM Heron
  (doi:[10.5281/zenodo.24496368](https://doi.org/10.5281/zenodo.24496368)).
- Paper 10: Decisions No Correction of the Uncontrolled Circuit Can Reach: Real-Time Control on IBM Heron, Declared
  Before the Data (doi:[10.5281/zenodo.24367001](https://doi.org/10.5281/zenodo.24367001)).
- Paper 9: What Makes Raw Quantum Dynamics Readable? The Share of Signal Kept, Across 179 Results on IBM Heron
  (doi:[10.5281/zenodo.24278888](https://doi.org/10.5281/zenodo.24278888)).
- Paper 8: Readable Quantum Dynamics Without Error Mitigation: Dual-Unitary Chains up to 91 Qubits and 3D Lattice
  Tests (doi:[10.5281/zenodo.24136512](https://doi.org/10.5281/zenodo.24136512)).
- Paper 7: Exact answers, no map, no mitigation: user-written circuits through a placement-only OS on IBM Heron r2
  devices (doi:[10.5281/zenodo.24046673](https://doi.org/10.5281/zenodo.24046673)).
- Paper 6: What Sets the Readable Depth of a Placed Ring, and How Far a Self-Checking Layer Pushes It
  (doi:[10.5281/zenodo.23883938](https://doi.org/10.5281/zenodo.23883938)).
- Paper 5: Readable Depth Without the Calibration Map
  (doi:[10.5281/zenodo.23827536](https://doi.org/10.5281/zenodo.23827536)).
- psf-os 0.3.3: doi:[10.5281/zenodo.24496267](https://doi.org/10.5281/zenodo.24496267); every release:
  doi:[10.5281/zenodo.24211404](https://doi.org/10.5281/zenodo.24211404).

<!-- src: notebook notes/paper7/ZENODO.md .. notes/paper11/ZENODO.md, notes/paper10_ff/ZENODO.md (titles, DOIs);
     psf-zero POC/README.md at 8769aa5 (Papers 5, 6); psf-os README.md (every release) -->

## Licence

AGPL-3.0 (psf-os). See `LICENSE` at the top of the repository. The request form, report template and NDA template in
this folder are templates to fill in, not part of the software.
