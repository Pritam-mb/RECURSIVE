"""
Process-wide simulation clock.

A single offset, applied on top of wall-clock UTC, is the source of truth for
the epoch used by the snapshot and alert pipelines. Keeping it here (rather
than in the API module) means the background refresh loop and the request
handlers agree on "now", so shifting the clock is not silently reverted by the
next background tick.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

MAX_OFFSET_HOURS = 24.0

_offset_hours: float = 0.0


def get_offset_hours() -> float:
    return _offset_hours


def set_offset_hours(offset_hours: float) -> float:
    """
    Set the simulation clock offset, bounded so a typo cannot request an
    absurd epoch. Raises ValueError when out of range.
    """
    global _offset_hours

    value = float(offset_hours)
    if not -MAX_OFFSET_HOURS <= value <= MAX_OFFSET_HOURS:
        raise ValueError(
            f"offset_hours must be within [-{MAX_OFFSET_HOURS:g}, {MAX_OFFSET_HOURS:g}]"
        )

    _offset_hours = value
    return _offset_hours


def reset() -> None:
    global _offset_hours
    _offset_hours = 0.0


def real_now() -> datetime:
    """Unshifted wall-clock UTC. Use for audit/expiry timestamps."""
    return datetime.now(timezone.utc)


def simulation_now() -> datetime:
    """Epoch the simulation is currently pinned to."""
    return datetime.now(timezone.utc) + timedelta(hours=_offset_hours)
