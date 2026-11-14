"""psf-os: a self-checking layer between a ring circuit and an IBM heavy-hex device (OS plan milestone M5, skeleton).

Layers (OS plan, TOQB prereg/drafts-2026-11-01/os_plan_DRAFT.md):

  L0 device    psf_os.device    coupling graph and failed elements only (no quality number is read or stored)
  L1 scan      psf_os.scan      the rule's mirror echo read per ring site; dead and weak runs
  L2 registry  psf_os.registry  the defect list: black paths carried over days, gray paths for one day only
  L3 allocate  psf_os.allocate  the rule's placement with listed paths masked; reports what it skipped
  L4 repair    psf_os.repair    local frame flip, mirror-type circuits only
  L5 sentinel  psf_os.sentinel  interface only in M5
  L6 job       psf_os.job       logical circuit in; assembled job and report out

The module boundaries are pure functions with JSON-able inputs and outputs, so that the core can later be replaced
by Rust (milestone M8). Code taken from TOQB says so at the top of the file.
"""

__version__ = "0.1.1"

# The declared rules of TOQB (defect_list.py, safe_loop.py at 8d44f05), used unchanged. Changing any of them is the
# owner's decision.
RULES = dict(
    dead_threshold=0.12,  # a site is dead if its echo signal at D = 24 is below this
    weak_relative=0.5,    # a site is weak if below max(dead_threshold, weak_relative x ring median)
    min_run=3,            # a dead or weak run is at least this many consecutive ring sites (cyclic)
    centre=4,             # the listed path is the centre of a run, this many qubits
    min_repeats=2,        # a black path is LISTED after this many jobs
    expiry_days=7.0,      # and EXPIRED when not seen for this long
    overlap=2,            # two sightings are the same defect if their centres share this many qubits
    scan_D=24,            # the scan echo's full depth (one-way L = 12)
)
