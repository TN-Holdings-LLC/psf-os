"""L5: sentinels. Interface only in M5 (OS plan: M6 adds echo circuits on the user's ring to the user's job and flags
the result when they read below the registry's line).

The chain calls `plan` and records what it returns; in M5 it returns no circuits.
"""
from __future__ import annotations


def plan(dev, ring, registry, now):
    """Sentinel jobs for a placed ring: [] in M5."""
    return []


def verdict(job_dir, runs, registry):
    """The drift flag of a finished job: not implemented in M5."""
    return dict(status="NOT IMPLEMENTED (M6)")
