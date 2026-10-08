"""
agency_authority.py — Per-agency session authority model for satellite commanding.

Ownership comes from the single shared attribution in app.core.agency
(CelesTrak SATCAT owner first, operator name patterns second), so the agency
shown in the UI and the agency that is allowed to command an object are the
same thing by construction.

Rules (check_authority / explain_authority):
  1. DEMO_SESSION is an explicit demo super-user. It is honoured only while
     demo mode is enabled (env ORBIT_SENTINEL_DEMO_SESSION, default "1"); set
     it to "0" in any real deployment. Its decisions are tagged "demo_session".
  2. Unknown / expired sessions are refused.
  3. Objects SATCAT (or the name) classifies as DEB or R/B carry no propulsion
     and cannot be commanded by agency sessions.
  4. Session agency "ALL" (admin) may command any manoeuvrable object.
  5. A session may command objects whose controlling agency matches its own
     (legacy names such as ROSCOSMOS / CNSA / ISS/NASA are mapped onto the
     SATCAT-derived labels).
  6. Unknown owner: a REAL catalogued object (SATCAT or bundled TLE) with an
     unresolved owner is NOT commandable by agency sessions. Only synthetic
     objects that exist in no catalogue (scenario injections) may be commanded
     by any valid session.

Usage:
    from app.core.agency_authority import authority_manager
    ok = authority_manager.check_authority(session_id, norad_id, sat_name)
    session_id = authority_manager.create_session("SpaceX")
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.agency import UNKNOWN_AGENCY, canonical_agencies, infer_agency

logger = logging.getLogger(__name__)

DEMO_SESSION_ID = "DEMO_SESSION"


def demo_session_enabled() -> bool:
    return os.getenv("ORBIT_SENTINEL_DEMO_SESSION", "1") == "1"


def _object_facts(norad_id: int | None, satellite_name: str) -> dict[str, Any]:
    """Controlling agency, object type and catalogue status for one object."""
    object_type = None
    catalogued = False
    try:
        from app.core import satcat

        rec = satcat.lookup(norad_id) if norad_id is not None else None
        if rec is not None:
            object_type = rec.get("object_type")
            catalogued = True  # SATCAT entry or present in the bundled TLE catalogue
        if object_type in (None, "UNK"):
            object_type = satcat.object_type_from_name(satellite_name) if satellite_name else object_type
    except Exception:  # pragma: no cover - satcat data missing
        pass
    return {
        "controlling_agency": infer_agency(satellite_name, norad_id),
        "object_type": object_type or "UNK",
        "catalogued": catalogued,
    }


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
        self.sessions[DEMO_SESSION_ID] = {
            "session_id": DEMO_SESSION_ID,
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

    def get_controlling_agency(self, satellite_name: str, norad_id: int | None = None) -> str:
        """Controlling agency from the shared SATCAT-first attribution (app.core.agency)."""
        return infer_agency(satellite_name or "", norad_id)

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

    def explain_authority(
        self,
        session_id: str,
        norad_id: int | None,
        satellite_name: str,
    ) -> dict[str, Any]:
        """Authority decision with the rule that produced it (see module docstring)."""
        facts = _object_facts(norad_id, satellite_name)
        decision = {"allowed": False, "rule": "", "session_agency": None, **facts}

        if session_id == DEMO_SESSION_ID:
            decision["session_agency"] = "ALL"
            if demo_session_enabled():
                decision.update(allowed=True, rule="demo_session")
            else:
                decision["rule"] = "demo_session_disabled"
            return decision

        session = self.sessions.get(session_id)
        if session is None:
            decision["rule"] = "session_not_found"
            return decision
        if self._is_expired(session):
            decision["rule"] = "session_expired"
            return decision

        session_agency = session.get("agency", UNKNOWN_AGENCY)
        decision["session_agency"] = session_agency

        if facts["object_type"] in ("DEB", "R/B"):
            decision["rule"] = "object_not_manoeuvrable"
            return decision
        if session_agency == "ALL":
            decision.update(allowed=True, rule="admin_session")
            return decision

        controlling = facts["controlling_agency"]
        if controlling != UNKNOWN_AGENCY and controlling in canonical_agencies(session_agency):
            decision.update(allowed=True, rule="agency_match")
            return decision
        if controlling == UNKNOWN_AGENCY:
            if facts["catalogued"]:
                decision["rule"] = "catalogued_owner_unresolved"
            else:
                decision.update(allowed=True, rule="synthetic_object_no_owner")
            return decision

        decision["rule"] = "agency_mismatch"
        return decision

    def check_authority(
        self,
        session_id: str,
        norad_id: int,
        satellite_name: str,
    ) -> bool:
        """True when the session may command the object (rules in module docstring)."""
        decision = self.explain_authority(session_id, norad_id, satellite_name)
        if not decision["allowed"]:
            logger.info(
                "Authority denied (%s): session agency='%s', object='%s' (NORAD %s) controlled by '%s'",
                decision["rule"], decision["session_agency"], satellite_name, norad_id,
                decision["controlling_agency"],
            )
        return bool(decision["allowed"])

    def get_session_satellites(
        self,
        session_id: str,
        all_satellite_names: list[str],
        norad_ids: list[int] | None = None,
    ) -> list[str]:
        """Names of the objects this session may command (ids improve attribution)."""
        session = self.sessions.get(session_id)
        if session is None or self._is_expired(session):
            return []
        ids = norad_ids if norad_ids is not None else [None] * len(all_satellite_names)
        return [
            name for name, nid in zip(all_satellite_names, ids)
            if self.check_authority(session_id, nid, name)
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
