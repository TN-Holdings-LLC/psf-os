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
| L5 sentinel | `psf_os.sentinel` | interface only in M5 |
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

Code ported from TOQB (same rights holder) says so at the top of each file.

AGPL v3. See `LICENSE`.
