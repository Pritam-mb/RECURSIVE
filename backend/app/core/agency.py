"""
Agency / operator attribution — the single table shared by the whole backend
(snapshot labels, alert metadata, /api/agencies and the authority model in
agency_authority.py).

Resolution order for ``infer_agency(name, norad_id)``:

1. CelesTrak SATCAT owner code for the NORAD id (app.core.satcat), or for the
   object name when the id is not given and the name is unambiguous in SATCAT.
   The owner code is a country / organisation (US, CIS, PRC, ESA, ...).
2. Operator refinement: a few well-known constellations / programmes are
   attributed to their operator (STARLINK -> SpaceX) but ONLY when the name
   pattern is consistent with the SATCAT owner (a "STARLINK" object owned by
   US). This keeps operator-level sessions (SpaceX, NASA, ...) meaningful
   without overriding catalogue facts.
3. If SATCAT has no record, the same name patterns are used on their own.
4. Otherwise "Unknown".
"""

from __future__ import annotations

import re
from functools import lru_cache

UNKNOWN_AGENCY = "Unknown"

# SATCAT owner code -> short agency label used across the UI and the authority
# model. Codes not listed fall back to the CelesTrak owner name.
OWNER_AGENCY_LABELS: dict[str, str] = {
    "US": "USA", "CIS": "Russia/CIS", "PRC": "China", "IND": "India", "ISRO": "India",
    "JPN": "Japan", "FR": "France", "UK": "UK", "ESA": "ESA", "ESRO": "ESA", "ISS": "ISS",
    "ITSO": "Intelsat", "GLOB": "Globalstar", "IRID": "Iridium", "ORB": "ORBCOMM",
    "O3B": "SES", "SES": "SES", "EUTE": "Eutelsat", "EUME": "EUMETSAT", "IM": "Inmarsat",
    "AB": "Arabsat", "AC": "AsiaSat", "ABS": "ABS", "SKOR": "South Korea", "ROC": "Taiwan",
    "GER": "Germany", "IT": "Italy", "CA": "Canada", "SPN": "Spain", "AUS": "Australia",
    "NATO": "NATO", "SEAL": "Sea Launch",
}

# Operator refinement: (regex on upper-case name, operator, SATCAT owners for
# which the pattern is accepted). Owner None means "any / no SATCAT record".
OPERATOR_PATTERNS: list[tuple[str, str, frozenset[str] | None]] = [
    (r"\bSTARLINK\b", "SpaceX", frozenset({"US"})),
    (r"\bDRAGON\b|\bSPACEX\b", "SpaceX", frozenset({"US"})),
    (r"\bKUIPER\b", "Amazon", frozenset({"US"})),
    (r"\bIRIDIUM\b", "Iridium", frozenset({"US", "IRID"})),
    (r"\bONEWEB\b", "OneWeb", frozenset({"UK"})),
    (r"\bNAVSTAR\b|\bGPS\b|^USA \d+|\bMILSTAR\b|\bMUOS\b|\bWGS\b|\bSBIRS\b|\bDMSP\b",
     "US Space Force", frozenset({"US"})),
    (r"\bNOAA\b|\bGOES\b|\bSUOMI\b|\bJPSS\b", "NOAA", frozenset({"US"})),
    (r"\bHST\b|\bHUBBLE\b|\bTERRA\b|\bAQUA\b|\bAURA\b|\bLANDSAT\b|\bSWIFT\b|\bTDRS\b|\bICESAT\b",
     "NASA", frozenset({"US"})),
    (r"\bISS\b|\bZARYA\b", "ISS", frozenset({"ISS"})),
    (r"\bSENTINEL\b|\bGAIA\b|\bENVISAT\b|\bCRYOSAT\b|\bSWARM\b|\bGALILEO\b", "ESA", None),
    (r"\bCOSMOS\b|\bKOSMOS\b|\bGLONASS\b|\bRESURS\b|\bMETEOR\b|\bELEKTRO\b|\bSOYUZ\b|\bPROGRESS\b",
     "Russia/CIS", frozenset({"CIS"})),
    (r"\bFENGYUN\b|\bYAOGAN\b|\bTIANGONG\b|\bTIANHE\b|\bBEIDOU\b|\bSHIYAN\b|\bCZ-", "China",
     frozenset({"PRC"})),
    (r"\bGSAT\b|\bINSAT\b|\bCARTOSAT\b|\bRESOURCESAT\b|\bRISAT\b|\bIRNSS\b|\bOCEANSAT\b", "India",
     frozenset({"IND", "ISRO"})),
]
_COMPILED = [(re.compile(p), op, owners) for p, op, owners in OPERATOR_PATTERNS]

# Legacy agency names (sessions minted by older clients / main.py startup)
# mapped onto the labels above. "ISS/NASA" covers both the station and NASA.
LEGACY_AGENCY_ALIASES: dict[str, tuple[str, ...]] = {
    "ROSCOSMOS": ("Russia/CIS",),
    "CNSA": ("China",),
    "ISRO": ("India",),
    "ISS/NASA": ("ISS", "NASA"),
    "UNKNOWN": (UNKNOWN_AGENCY,),
}

# Backwards-compatible view of the name patterns (keyword, agency) for older
# importers (sim_engine, cascade_planner). Derived from OPERATOR_PATTERNS.
AGENCY_ALIASES: list[tuple[str, str]] = [
    ("STARLINK", "SpaceX"), ("SPACEX", "SpaceX"), ("ISS", "ISS"), ("COSMOS", "Russia/CIS"),
    ("RESURS", "Russia/CIS"), ("FENGYUN", "China"), ("YAOGAN", "China"), ("NOAA", "NOAA"),
    ("HUBBLE", "NASA"), ("IRIDIUM", "Iridium"),
]


def owner_label(owner_code: str | None) -> str:
    """Short agency label for a SATCAT owner code."""
    if not owner_code:
        return UNKNOWN_AGENCY
    from app.core.satcat import OWNER_NAMES, UNRESOLVED_OWNERS

    if owner_code in UNRESOLVED_OWNERS:
        return UNKNOWN_AGENCY
    return OWNER_AGENCY_LABELS.get(owner_code) or OWNER_NAMES.get(owner_code, owner_code)


def _operator_from_name(name: str, owner_code: str | None) -> str | None:
    upper = (name or "").upper()
    if not upper:
        return None
    for regex, operator, owners in _COMPILED:
        if regex.search(upper):
            if owner_code is None or owners is None or owner_code in owners:
                return operator
    return None


def _satcat_owner(name: str | None, norad_id) -> tuple[str | None, str]:
    """(owner_code, how) from SATCAT by id, else by unambiguous name."""
    try:
        from app.core import satcat
    except Exception:  # pragma: no cover - data module missing
        return None, "none"
    if norad_id is not None:
        rec = satcat.lookup(norad_id)
        if rec and rec.get("source") == "celestrak_satcat":
            return rec.get("owner"), "satcat_id"
    code = satcat.owner_for_name(name)
    if code:
        return code, "satcat_name"
    return None, "none"


@lru_cache(maxsize=65536)
def _resolve(name: str, norad_id: int | None) -> tuple[str, str, str | None]:
    owner_code, how = _satcat_owner(name, norad_id)
    if owner_code is not None:
        from app.core.satcat import UNRESOLVED_OWNERS

        operator = _operator_from_name(name, owner_code)
        if operator:
            return operator, f"{how}+operator_pattern", owner_code
        if owner_code not in UNRESOLVED_OWNERS:
            return owner_label(owner_code), how, owner_code
        # SATCAT says TBD/UNK: a name pattern may still identify it.
        operator = _operator_from_name(name, None)
        if operator:
            return operator, "name_pattern", owner_code
        return UNKNOWN_AGENCY, how, owner_code
    operator = _operator_from_name(name, None)
    if operator:
        return operator, "name_pattern", None
    return UNKNOWN_AGENCY, "none", None


def _norm_id(norad_id) -> int | None:
    try:
        return int(norad_id) if norad_id is not None else None
    except (TypeError, ValueError):
        return None


def infer_agency(name: str, norad_id: int | None = None) -> str:
    """Agency label for an object (SATCAT owner first, name pattern fallback)."""
    return _resolve(name or "", _norm_id(norad_id))[0]


def agency_attribution(name: str, norad_id: int | None = None) -> dict:
    """Agency label plus provenance: how it was derived and the SATCAT owner code."""
    agency, source, owner_code = _resolve(name or "", _norm_id(norad_id))
    owner_name = None
    if owner_code:
        from app.core.satcat import OWNER_NAMES

        owner_name = OWNER_NAMES.get(owner_code, owner_code)
    return {"agency": agency, "agency_source": source, "owner": owner_code, "owner_name": owner_name}


def canonical_agencies(agency_name: str | None) -> tuple[str, ...]:
    """Labels a (possibly legacy) agency name stands for, e.g. ROSCOSMOS -> Russia/CIS."""
    if not agency_name:
        return (UNKNOWN_AGENCY,)
    alias = LEGACY_AGENCY_ALIASES.get(agency_name.strip().upper())
    return alias if alias else (agency_name.strip(),)
