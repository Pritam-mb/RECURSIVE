"""
Offline CelesTrak SATCAT lookup.

Data: backend/app/data/satcat_snapshot.csv, a subset of the public CelesTrak
satellite catalogue (https://celestrak.org/pub/satcat.csv) restricted to the
NORAD ids in the bundled TLE file. Rebuild it with
``python scripts/build_satcat_snapshot.py``.

``lookup(norad_id)`` returns the catalogue record (owner code + owner name,
object type, launch/decay dates, radar cross-section, international
designator). When an id is not in SATCAT (e.g. Alpha-5 analyst objects) the
record is derived from the TLE itself: the international designator in TLE
line 1 gives the launch year and launch number, and the object type is
inferred from the name. Such records carry ``source="tle_derived"`` and
``owner=None`` so nothing downstream mistakes them for catalogue facts.
"""

from __future__ import annotations

import csv
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "data" / "satcat_snapshot.csv"
TLE_PATH = Path(__file__).resolve().parents[1] / "simulation" / "tle-data.txt"

# CelesTrak SATCAT "OWNER" (source) codes -> names, transcribed from
# https://celestrak.org/satcat/sources.php (retrieved 2026-10-08).
OWNER_NAMES: dict[str, str] = {
    "AB": "Arab Satellite Communications Organization", "ABS": "Asia Broadcast Satellite",
    "AC": "Asia Satellite Telecommunications Company (ASIASAT)", "ALG": "Algeria", "ANG": "Angola",
    "ARGN": "Argentina", "ARM": "Republic of Armenia", "ASRA": "Austria", "AUS": "Australia",
    "AZER": "Azerbaijan", "BEL": "Belgium", "BELA": "Belarus", "BERM": "Bermuda",
    "BGD": "Peoples Republic of Bangladesh", "BHR": "The Kingdom of Bahrain",
    "BHUT": "The Kingdom of Bhutan", "BOL": "Bolivia", "BRAZ": "Brazil", "BUL": "Bulgaria",
    "BWA": "Republic of Botswana", "CA": "Canada", "CHBZ": "China/Brazil", "CHTU": "China/Türkiye",
    "CHLE": "Chile", "CIS": "Commonwealth of Independent States (former USSR)", "COL": "Colombia",
    "CRI": "Republic of Costa Rica", "CZCH": "Czech Republic (former Czechoslovakia)",
    "DEN": "Denmark", "DJI": "Republic of Djibouti", "ECU": "Ecuador", "EGYP": "Egypt",
    "ESA": "European Space Agency", "ESRO": "European Space Research Organization",
    "EST": "Estonia", "ETH": "Ethiopia",
    "EUME": "European Organization for the Exploitation of Meteorological Satellites (EUMETSAT)",
    "EUTE": "European Telecommunications Satellite Organization (EUTELSAT)",
    "FGER": "France/Germany", "FIN": "Finland", "FR": "France", "FRIT": "France/Italy",
    "GER": "Germany", "GHA": "Republic of Ghana", "GLOB": "Globalstar", "GREC": "Greece",
    "GRSA": "Greece/Saudi Arabia", "GUAT": "Guatemala", "HRV": "Republic of Croatia",
    "HUN": "Hungary", "IM": "International Mobile Satellite Organization (INMARSAT)",
    "IND": "India", "INDO": "Indonesia", "IRAN": "Iran", "IRAQ": "Iraq", "IRID": "Iridium",
    "IRL": "Ireland", "ISRA": "Israel", "ISRO": "Indian Space Research Organisation",
    "ISS": "International Space Station", "IT": "Italy",
    "ITSO": "International Telecommunications Satellite Organization (INTELSAT)",
    "JPN": "Japan", "KAZ": "Kazakhstan", "KEN": "Republic of Kenya", "LAOS": "Laos",
    "LKA": "Democratic Socialist Republic of Sri Lanka", "LTU": "Lithuania", "LUXE": "Luxembourg",
    "MA": "Morocco", "MALA": "Malaysia", "MCO": "Principality of Monaco",
    "MDA": "Republic of Moldova", "MEX": "Mexico", "MMR": "Republic of the Union of Myanmar",
    "MNE": "Montenegro", "MNG": "Mongolia", "MUS": "Mauritius",
    "NATO": "North Atlantic Treaty Organization", "NETH": "Netherlands", "NICO": "New ICO",
    "NIG": "Nigeria", "NKOR": "Democratic People's Republic of Korea", "NOR": "Norway",
    "NPL": "Federal Democratic Republic of Nepal", "NZ": "New Zealand", "O3B": "O3b Networks",
    "ORB": "ORBCOMM", "PAKI": "Pakistan", "PERU": "Peru", "POL": "Poland", "POR": "Portugal",
    "PRC": "People's Republic of China", "PRY": "Republic of Paraguay",
    "PRES": "People's Republic of China/European Space Agency", "QAT": "State of Qatar",
    "RASC": "RascomStar-QAF", "ROC": "Taiwan (Republic of China)", "ROM": "Romania",
    "RP": "Philippines (Republic of the Philippines)", "RWA": "Republic of Rwanda",
    "SAFR": "South Africa", "SAUD": "Saudi Arabia", "SDN": "Republic of Sudan",
    "SEAL": "Sea Launch", "SEN": "Republic of Senegal", "SES": "SES", "SGJP": "Singapore/Japan",
    "SING": "Singapore", "SKOR": "Republic of Korea", "SLB": "Solomon Islands", "SPN": "Spain",
    "STCT": "Singapore/Taiwan", "SVN": "Slovenia", "SWED": "Sweden", "SWTZ": "Switzerland",
    "TBD": "To Be Determined", "THAI": "Thailand", "TMMC": "Turkmenistan/Monaco",
    "TUN": "Republic of Tunisia", "TURK": "Türkiye", "UAE": "United Arab Emirates",
    "UK": "United Kingdom", "UKR": "Ukraine", "UNK": "Unknown", "URY": "Uruguay",
    "US": "United States", "USBZ": "United States/Brazil", "VAT": "Vatican City State",
    "VENZ": "Venezuela", "VTNM": "Vietnam", "ZWE": "Republic of Zimbabwe",
}

# Owner codes that do not identify a responsible party.
UNRESOLVED_OWNERS = {"", "TBD", "UNK"}

_OBJECT_TYPES = {"PAY": "PAY", "R/B": "R/B", "DEB": "DEB"}
_ALPHA5 = "ABCDEFGHJKLMNPQRSTUVWXYZ"

_lock = threading.Lock()
_records: dict[int, dict[str, Any]] | None = None
_name_owner: dict[str, str] | None = None
_tle_index: dict[int, tuple[str, str]] | None = None
_snapshot_meta: str = ""


def parse_catnum(field: str) -> int:
    """TLE catalogue number including Alpha-5 ids ('T0000' -> 270000)."""
    field = field.strip()
    if field and field[0].isalpha():
        return (_ALPHA5.index(field[0].upper()) + 10) * 10000 + int(field[1:])
    return int(field)


def rcs_size_class(rcs_m2: float | None) -> str | None:
    """Space-Track RCS size convention: SMALL < 0.1 m^2 <= MEDIUM < 1 m^2 <= LARGE."""
    if rcs_m2 is None:
        return None
    if rcs_m2 < 0.1:
        return "SMALL"
    if rcs_m2 < 1.0:
        return "MEDIUM"
    return "LARGE"


def object_type_from_name(name: str | None) -> str:
    upper = (name or "").upper()
    if not upper or upper.startswith("TBA") or upper.startswith("OBJECT "):
        return "UNK"
    if " DEB" in upper or upper.startswith("DEB") or "(DEB" in upper:
        return "DEB"
    if "R/B" in upper or " AKM" in upper or " PKM" in upper:
        return "R/B"
    return "PAY"


def _float(value: str) -> float | None:
    try:
        return float(value) if value not in ("", None) else None
    except ValueError:
        return None


def _load() -> None:
    global _records, _name_owner, _snapshot_meta
    with _lock:
        if _records is not None:
            return
        records: dict[int, dict[str, Any]] = {}
        names: dict[str, set[str]] = {}
        try:
            with open(SNAPSHOT_PATH, encoding="utf-8") as handle:
                first = handle.readline()
                if first.startswith("#"):
                    _snapshot_meta = first[1:].strip()
                else:
                    handle.seek(0)
                for row in csv.DictReader(handle):
                    try:
                        norad = int(row["norad_id"])
                    except (TypeError, ValueError):
                        continue
                    owner = (row.get("owner") or "").strip()
                    rcs = _float(row.get("rcs_m2", ""))
                    rec = {
                        "norad_id": norad,
                        "name": row.get("name") or None,
                        "owner": owner or None,
                        "owner_name": OWNER_NAMES.get(owner, owner) if owner else None,
                        "object_type": _OBJECT_TYPES.get((row.get("object_type") or "").strip(), "UNK"),
                        "ops_status": row.get("ops_status") or None,
                        "launch_date": row.get("launch_date") or None,
                        "launch_site": row.get("launch_site") or None,
                        "decay_date": row.get("decay_date") or None,
                        "intl_designator": row.get("intl_designator") or None,
                        "period_min": _float(row.get("period_min", "")),
                        "inclination_deg": _float(row.get("inclination_deg", "")),
                        "apogee_km": _float(row.get("apogee_km", "")),
                        "perigee_km": _float(row.get("perigee_km", "")),
                        "rcs_m2": rcs,
                        "rcs_size": rcs_size_class(rcs),
                        "source": "celestrak_satcat",
                    }
                    records[norad] = rec
                    if rec["name"]:
                        names.setdefault(rec["name"].upper(), set()).add(owner)
        except FileNotFoundError:
            logger.warning("SATCAT snapshot missing at %s; run scripts/build_satcat_snapshot.py", SNAPSHOT_PATH)
        # A name resolves to an owner only when every catalogue entry with that
        # name has the same owner (e.g. all "COSMOS 2251 DEB" are CIS).
        _name_owner = {n: next(iter(o)) for n, o in names.items() if len(o) == 1 and next(iter(o))}
        _records = records
        logger.info("SATCAT snapshot loaded: %d records", len(records))


def _load_tle_index() -> dict[int, tuple[str, str]]:
    global _tle_index
    if _tle_index is None:
        index: dict[int, tuple[str, str]] = {}
        try:
            with open(TLE_PATH, encoding="utf-8") as handle:
                name = ""
                for line in handle:
                    if line.startswith("0 "):
                        name = line[2:].strip()
                    elif line.startswith("1 "):
                        try:
                            index[parse_catnum(line[2:7])] = (name, line.rstrip("\n"))
                        except ValueError:
                            pass
        except FileNotFoundError:
            pass
        _tle_index = index
    return _tle_index


def intl_designator_from_tle(line1: str) -> str | None:
    """TLE line-1 cols 10-17 ('98067A') -> COSPAR id '1998-067A'."""
    field = (line1[9:17] if len(line1) >= 17 else "").strip()
    if len(field) < 5 or not field[:2].isdigit():
        return None
    yy = int(field[:2])
    year = 1900 + yy if yy >= 57 else 2000 + yy  # Sputnik era cutoff, as in the TLE spec
    return f"{year}-{field[2:]}"


def _tle_derived(norad_id: int, name: str | None, line1: str | None) -> dict[str, Any] | None:
    if line1 is None and name is None:
        entry = _load_tle_index().get(norad_id)
        if entry is None:
            return None
        name, line1 = entry
    intl = intl_designator_from_tle(line1) if line1 else None
    return {
        "norad_id": norad_id,
        "name": name,
        "owner": None,
        "owner_name": None,
        "object_type": object_type_from_name(name),
        "ops_status": None,
        "launch_date": None,
        "launch_year": int(intl[:4]) if intl else None,
        "launch_site": None,
        "decay_date": None,
        "intl_designator": intl,
        "rcs_m2": None,
        "rcs_size": None,
        "source": "tle_derived",
    }


@lru_cache(maxsize=65536)
def _lookup_cached(norad_id: int) -> dict[str, Any] | None:
    _load()
    rec = (_records or {}).get(norad_id)
    if rec is not None:
        out = dict(rec)
        out["launch_year"] = int(rec["launch_date"][:4]) if rec.get("launch_date") else None
        return out
    return _tle_derived(norad_id, None, None)


def lookup(norad_id: int | str | None, *, name: str | None = None, tle_line1: str | None = None) -> dict[str, Any] | None:
    """
    SATCAT record for a NORAD id, or a TLE-derived record when the id is not
    catalogued, or None when nothing is known (e.g. synthetic scenario ids).
    """
    try:
        nid = int(norad_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    rec = _lookup_cached(nid)
    if rec is None and (name or tle_line1):
        rec = _tle_derived(nid, name, tle_line1)
    return dict(rec) if rec is not None else None


def is_catalogued(norad_id: int | str | None) -> bool:
    """True when the id is a real SATCAT entry (not TLE-derived, not synthetic)."""
    rec = lookup(norad_id)
    return bool(rec and rec.get("source") == "celestrak_satcat")


def owner_for_name(name: str | None) -> str | None:
    """SATCAT owner code for an object name, when the name is unambiguous."""
    if not name:
        return None
    _load()
    return (_name_owner or {}).get(name.strip().upper())


def snapshot_info() -> dict[str, Any]:
    _load()
    return {"path": str(SNAPSHOT_PATH), "records": len(_records or {}), "meta": _snapshot_meta}
