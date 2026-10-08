"""
TLE fetcher with Space-Track credentials support.
Parses 3-line TLE format (0-name, 1-line1, 2-line2).
"""

import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SPACE_TRACK_BASE_URL = "https://www.space-track.org"
SPACE_TRACK_LOGIN_URL = f"{SPACE_TRACK_BASE_URL}/ajaxauth/login"
SPACE_TRACK_QUERY_PATHS = [
    "/basicspacedata/query/class/gp/decay_date/null-val/epoch/%3Enow-10/orderby/norad_cat_id/format/3le",
    "/basicspacedata/query/class/tle_latest/orderby/norad_cat_id/format/3le",
]

TLE_DATA_URL = (
    "https://satellitetracker3d.nyc3.cdn.digitaloceanspaces.com/tle-data.txt"
)


def _get_spacetrack_credentials() -> tuple[str, str] | None:
    user = os.getenv("SPACETRACK_USER")
    password = os.getenv("SPACETRACK_PASS")
    if not user or not password:
        return None
    return user, password


def parse_3line_tle(text: str) -> list[dict]:
    """
    Parse 3-line TLE text format.
    Lines: '0 NAME', '1 ...', '2 ...'
    Returns list of {name, norad_id, line1, line2}.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    tles = []
    i = 0

    while i < len(lines) - 2:
        line0 = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        # Validate: line0 starts with '0 ', line1 with '1 ', line2 with '2 '
        if line0.startswith("0 ") and line1.startswith("1 ") and line2.startswith("2 "):
            name = line0[2:].strip()
            try:
                norad_id = int(line1[2:7].strip())
            except ValueError:
                norad_id = 0

            tles.append({
                "name": name,
                "norad_id": norad_id,
                "line1": line1,
                "line2": line2,
            })
            i += 3
        else:
            i += 1

    return tles


def load_local_tles(max_sats: int = 0) -> list[dict]:
    """
    Load the bundled local TLE snapshot.
    max_sats: limit number of satellites (0 = all).
    Returns list of {name, norad_id, line1, line2}.
    """
    file_path = Path(__file__).parent.parent / "simulation" / "tle-data.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tles = parse_3line_tle(text)

    if max_sats > 0:
        tles = tles[:max_sats]

    logger.info(f"Fetched {len(tles)} TLEs from local file")
    return tles


async def fetch_remote_tles(
    max_sats: int = 0,
    timeout_seconds: float = 15.0,
) -> list[dict]:
    """
    Fetch TLEs from the remote CDN source.
    max_sats: limit number of satellites (0 = all).
    Returns list of {name, norad_id, line1, line2}.
    """
    timeout = httpx.Timeout(timeout_seconds, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(TLE_DATA_URL)
        response.raise_for_status()
        text = response.text

    tles = parse_3line_tle(text)

    if max_sats > 0:
        tles = tles[:max_sats]

    logger.info(f"Fetched {len(tles)} TLEs from remote CDN")
    return tles


async def fetch_spacetrack_tles(
    max_sats: int = 0,
    timeout_seconds: float = 15.0,
) -> list[dict]:
    """
    Fetch current TLEs from Space-Track using the configured credentials.
    """
    credentials = _get_spacetrack_credentials()
    if credentials is None:
        raise RuntimeError("Space-Track credentials are not configured")

    identity, password = credentials
    timeout = httpx.Timeout(timeout_seconds, connect=5.0)
    headers = {
        "User-Agent": "OrbitalSentinel/1.0",
        "Accept": "text/plain, */*",
    }

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        login_response = await client.post(
            SPACE_TRACK_LOGIN_URL,
            data={"identity": identity, "password": password},
        )
        login_response.raise_for_status()

        last_error: Exception | None = None
        for query_path in SPACE_TRACK_QUERY_PATHS:
            query_url = f"{SPACE_TRACK_BASE_URL}{query_path}"
            try:
                response = await client.get(query_url)
                response.raise_for_status()
                tles = parse_3line_tle(response.text)
                if not tles:
                    logger.warning("Space-Track query returned no parseable TLEs: %s", query_path)
                    continue

                if max_sats > 0:
                    tles = tles[:max_sats]

                logger.info("Fetched %d TLEs from Space-Track", len(tles))
                return tles
            except Exception as error:
                last_error = error
                logger.warning("Space-Track query failed for %s: %s", query_path, error)

    raise RuntimeError(f"Space-Track fetch failed: {last_error}")


async def fetch_tles(max_sats: int = 0) -> list[dict]:
    """
    Fetch TLEs from Space-Track when credentials are available.
    Falls back to the CDN and then the bundled local snapshot.
    """
    credentials = _get_spacetrack_credentials()
    if credentials is not None:
        try:
            return await fetch_spacetrack_tles(max_sats=max_sats)
        except Exception as error:
            logger.warning(f"Space-Track fetch failed, falling back: {error}")

    try:
        return await fetch_remote_tles(max_sats=max_sats)
    except Exception as e:
        logger.warning(f"Remote TLE fetch failed, using local snapshot: {e}")
        try:
            return load_local_tles(max_sats=max_sats)
        except Exception as fallback_error:
            logger.error(f"TLE read error: {fallback_error}")
            return []
