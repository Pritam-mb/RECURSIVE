"""
agency_authority.py — Per-agency session authority model for satellite commanding.

Replaces the hardcoded `agency_auth = True` in sim_engine.py with a real
authority enforcement model. Each agency has a session that expires after
8 hours; the session ID is checked before any maneuver is authorised.

DEMO_SESSION: A special session that can command any satellite. This is
the session used when the frontend does not specify a session_id, ensuring
the demo always works while the authority model is real.

Usage:
    from app.core.agency_authority import authority_manager

    # Check authority
    ok = authority_manager.check_authority("DEMO_SESSION", norad_id, sat_name)

    # Create agency-specific session
    session_id = authority_manager.create_session("SpaceX")
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Agency → satellite name pattern mapping ───────────────────────────────────
# Each agency controls satellites whose names contain any of the listed patterns.
# Patterns are checked case-insensitively.

AGENCY_SATELLITE_PATTERNS: dict[str, list[str]] = {
    "SpaceX":        ["STARLINK", "DRAGON", "SPACEX"],
    "ESA":           ["SENTINEL", "GAIA", "XMM", "ENVISAT", "ERS", "METEOSAT"],
    "ISRO":          ["GSAT", "INSAT", "CARTOSAT", "RESOURCESAT", "RISAT", "IRNSS"],
    "ISS/NASA":      ["ISS", "ZARYA", "ZVEZDA", "UNITY", "DESTINY"],
    "ROSCOSMOS":     ["COSMOS", "GLONASS", "RESURS", "METEOR", "ELEKTRO"],
    "CNSA":          ["FENGYUN", "YAOGAN", "TIANGONG", "BEIDOU", "SHIYAN"],
    "US Space Force": ["NAVSTAR", "GPS", "MILSTAR", "MUOS", "WGS", "SBIRS"],
    "NOAA":          ["NOAA", "GOES", "POES", "SUOMI"],
    "NASA":          ["HUBBLE", "TERRA", "AQUA", "AURA", "LANDSAT", "SWIFT"],
    "Iridium":       ["IRIDIUM"],
    "OneWeb":        ["ONEWEB"],
    "Amazon":        ["KUIPER"],
}

# Agencies that are known but control no active TLEs — can only observe
_OBSERVER_AGENCIES = {"UNKNOWN"}

# Session expiry duration
_SESSION_DURATION_HOURS = 8


class AgencyAuthorityManager:
    """
    Manages per-agency authority sessions for satellite commanding.

    Attributes
    ----------
    sessions : dict[str, dict]
        Maps session_id → {agency, authorized_at, expires_at}
    """

    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}
        # Initialise the DEMO_SESSION that never expires
        self._create_demo_session()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _create_demo_session(self):
        """Create the permanent DEMO_SESSION used by the frontend."""
        self.sessions["DEMO_SESSION"] = {
            "session_id": "DEMO_SESSION",
            "agency": "ALL",
            "authorized_at": datetime.now(timezone.utc).isoformat(),
            # Year 9999 — effectively never expires
            "expires_at": datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc).isoformat(),
            "is_demo": True,
        }

    def _is_expired(self, session: dict) -> bool:
        try:
            expires_at = datetime.fromisoformat(session["expires_at"])
            return datetime.now(timezone.utc) > expires_at
        except Exception:
            return True

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_controlling_agency(self, satellite_name: str) -> str:
        """
        Determine which agency controls a satellite by its TLE name.

        Returns the controlling agency name, or "UNKNOWN" if no pattern matches.
        """
        if not satellite_name:
            return "UNKNOWN"
        upper_name = satellite_name.upper()
        for agency, patterns in AGENCY_SATELLITE_PATTERNS.items():
            for pattern in patterns:
                if pattern.upper() in upper_name:
                    return agency
        return "UNKNOWN"

    def create_session(self, agency_name: str) -> str:
        """
        Create a new 8-hour authority session for the given agency.

        Returns the session_id string (UUID4).
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=_SESSION_DURATION_HOURS)
        self.sessions[session_id] = {
            "session_id": session_id,
            "agency": agency_name,
            "authorized_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "is_demo": False,
        }
        logger.info("Created session %s for agency '%s' (expires %s)", session_id, agency_name, expires.isoformat())
        return session_id

    def check_authority(
        self,
        session_id: str,
        norad_id: int,
        satellite_name: str,
    ) -> bool:
        """
        Check whether a session has authority to command a satellite.

        Rules (in order):
          1. DEMO_SESSION → always True (demo mode)
          2. Session not found → False
          3. Session expired → False
          4. Session agency == "ALL" → True (admin)
          5. Session agency == satellite controlling agency → True
          6. Satellite controlling agency == "UNKNOWN" → True
             (uncontrolled debris / uncatalogued objects can be maneuvered by anyone)
          7. Otherwise → False
        """
        # Always allow DEMO_SESSION
        if session_id == "DEMO_SESSION":
            return True

        session = self.sessions.get(session_id)
        if session is None:
            logger.warning("Authority check: session '%s' not found", session_id)
            return False

        if self._is_expired(session):
            logger.warning("Authority check: session '%s' expired", session_id)
            return False

        session_agency = session.get("agency", "UNKNOWN")

        # Admin / all-agency session
        if session_agency == "ALL":
            return True

        controlling_agency = self.get_controlling_agency(satellite_name)

        # Session agency matches satellite controlling agency
        if session_agency == controlling_agency:
            return True

        # Uncontrolled object — anyone can maneuver
        if controlling_agency == "UNKNOWN":
            return True

        logger.info(
            "Authority denied: session agency='%s', satellite='%s' controlled by '%s'",
            session_agency, satellite_name, controlling_agency,
        )
        return False

    def get_session_satellites(
        self,
        session_id: str,
        all_satellite_names: list[str],
    ) -> list[str]:
        """
        Return the list of satellite names this session can command.

        For DEMO_SESSION or ALL-agency sessions, returns all satellites.
        For agency-specific sessions, returns matching satellites plus
        all UNKNOWN satellites.
        """
        session = self.sessions.get(session_id)
        if session is None or self._is_expired(session):
            return []

        agency = session.get("agency", "UNKNOWN")
        if agency in ("ALL",):
            return list(all_satellite_names)

        return [
            name for name in all_satellite_names
            if self.get_controlling_agency(name) in (agency, "UNKNOWN")
        ]

    def get_session_info(self, session_id: str) -> dict | None:
        """Return session info dict or None if not found / expired."""
        session = self.sessions.get(session_id)
        if session is None:
            return None
        return {
            **session,
            "expired": self._is_expired(session),
        }

    def list_sessions(self) -> list[dict]:
        """Return list of all session info dicts (including expired)."""
        return [self.get_session_info(sid) for sid in self.sessions]


# ── Module-level singleton ─────────────────────────────────────────────────────
# Import this instance everywhere — never instantiate AgencyAuthorityManager directly.
authority_manager = AgencyAuthorityManager()
