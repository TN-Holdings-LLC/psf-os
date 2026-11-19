# psf-os, the 0.2 release: Zenodo input (DRAFT, workplace 2026-11-19; the form of the notebook's notes/paper9/ZENODO.md)

Not released. Each GitHub release is archived on Zenodo through `.zenodo.json` (0.1.1); this is what that file and the
record should say for the 0.2 release. `[...]` is filled at release.

- Release: tag v[number] on the branch that holds v0.2-proto-2026-11-18 and the fixes after the night of 2026-11-19
  (after-2026-11-19); `pyproject.toml` `version` and `psf_os/__init__.py` `__version__` both set to [number] (both
  say 0.1.1 now; a wheel built with 0.2.0rc0 in both installed and imported, workplace 2026-11-19).
- Title (unchanged): psf-os: a self-checking layer between a user's circuit and an IBM heavy-hex device, without the
  calibration map
- Creator: TN Holdings (no personal name; no e-mail).
- Resource type: Software. Version: [number]. Date: [date]. Licence: AGPL-3.0-only.
- Description (proposed; the 0.1 text first, the 0.2 sentence after it):
  psf-os places a circuit by formula (a ring by the rule, a 2D circuit by translation of its own pattern), adds checks
  made of the user's own circuit to the same job (a floor, a mirror, a ring sentinel), keeps a list of what it found
  dead, and returns an answer card. It does not read a calibration quality number: it uses the coupling graph and the
  elements marked as failed. Version 0.2 adds real-time control: circuits that measure in the middle and act on the
  result while they run, each with its uncontrolled twin (the off arm) and a floor in the same job; conditions on a
  whole classical register; build rules taken from what the devices did (fixed circuits packed together, conditional
  items kept apart, the build environment recorded); and a loop of rounds that diagnoses the device, applies a fixed
  rule and runs the chosen circuit unattended, with a ledger keyed to the device record's calibration time stamp. The
  declared device tests behind it are in https://github.com/TN-Holdings-LLC/psf-os-lab-notebook.
- Keywords: quantum computing; IBM Quantum; heavy-hex; circuit placement; error detection; reproducibility; dynamic
  circuits; mid-circuit measurement; feed-forward.
- Related identifiers: as 0.1.1 (the three papers it lists: isSupplementTo; the notebook: isSupplementedBy), and
  isSupplementTo paper 9 (10.5281/zenodo.24278888) and paper 10 (10.5281/zenodo.[paper 10 DOI]).

## `.zenodo.json` as proposed (replace at release; the paper 10 DOI filled first)

```json
{
  "title": "psf-os: a self-checking layer between a user's circuit and an IBM heavy-hex device, without the calibration map",
  "upload_type": "software",
  "creators": [{"name": "TN Holdings"}],
  "description": "[the description above, as one paragraph]",
  "license": "AGPL-3.0-only",
  "keywords": ["quantum computing", "IBM Quantum", "heavy-hex", "circuit placement", "error detection",
               "reproducibility", "dynamic circuits", "mid-circuit measurement", "feed-forward"],
  "related_identifiers": [
    {"identifier": "10.5281/zenodo.23883938", "relation": "isSupplementTo", "scheme": "doi"},
    {"identifier": "10.5281/zenodo.24046673", "relation": "isSupplementTo", "scheme": "doi"},
    {"identifier": "10.5281/zenodo.24136512", "relation": "isSupplementTo", "scheme": "doi"},
    {"identifier": "10.5281/zenodo.24278888", "relation": "isSupplementTo", "scheme": "doi"},
    {"identifier": "10.5281/zenodo.[paper 10]", "relation": "isSupplementTo", "scheme": "doi"},
    {"identifier": "https://github.com/TN-Holdings-LLC/psf-os-lab-notebook", "relation": "isSupplementedBy", "scheme": "url"}
  ]
}
```

## Before tagging

- The full test suite with the notebook as `TOQB_DIR` (CHANGELOG's Tests line filled from it).
- `pip wheel . --no-deps` builds (setuptools >= 77 by build isolation) and the wheel holds `psf_os` only.
- The README's "Real-time control (0.2.0, in preparation)" heading loses "in preparation".
