"""Agency inference shared across the backend (single source of truth)."""

from __future__ import annotations

AGENCY_ALIASES = [
    ("STARLINK", "SpaceX"),
    ("SPACEX", "SpaceX"),
    ("ISS", "ISS"),
    ("COSMOS", "Roscosmos"),
    ("RESURS", "Roscosmos"),
    ("FENGYUN", "CNSA"),
    ("YAOGAN", "CNSA"),
    ("NOAA", "NOAA"),
    ("HUBBLE", "NASA"),
    ("IRIDIUM", "Iridium"),
]


def infer_agency(name: str) -> str:
    upper_name = (name or "").upper()
    for keyword, agency in AGENCY_ALIASES:
        if keyword in upper_name:
            return agency
    return "Unknown"