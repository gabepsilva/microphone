"""Temporary Darwin session adapter for phase-2 porting.

The real ancestry and aggregate-device cleanup adapter is introduced together
with the Core Audio helper in the next phase; this adapter fails closed until
then so the picker cannot mistake another process for TagAlong's own audio.
"""

from __future__ import annotations


def session_of(_pid, **_kwargs):
    """Reject process identity checks until the Darwin adapter exists."""
    raise RuntimeError(
        "Darwin process identity is not implemented in this build; "
        "macOS helper ownership requires the next compatibility phase."
    )


def started_here(_pid, **_kwargs):
    """Reject process identity checks until the Darwin adapter exists."""
    raise RuntimeError(
        "Darwin process identity is not implemented in this build; "
        "macOS helper ownership requires the next compatibility phase."
    )


def orphans(**_kwargs):
    """No Darwin helper sweep is enabled before the Core Audio helper exists."""
    return []


def sweep_orphans(**_kwargs):
    """No Darwin helper sweep is enabled before the Core Audio helper exists."""
    return 0
