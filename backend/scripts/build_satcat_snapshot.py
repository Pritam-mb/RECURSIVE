"""
Rebuild backend/app/data/satcat_snapshot.csv from the CelesTrak SATCAT.

The snapshot is the offline, commit-able subset of the public CelesTrak
satellite catalogue (https://celestrak.org/pub/satcat.csv) restricted to the
NORAD ids that appear in the bundled TLE file
(backend/app/simulation/tle-data.txt). It is what app.core.satcat.lookup()
reads at runtime, so the backend never needs network access for ownership,
object type, launch/decay dates or radar cross-section.

Usage (from backend/):
    python scripts/build_satcat_snapshot.py                 # download + rebuild
    python scripts/build_satcat_snapshot.py --input satcat.csv   # use a local copy
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
TLE_FILE = BACKEND / "app" / "simulation" / "tle-data.txt"
OUT_FILE = BACKEND / "app" / "data" / "satcat_snapshot.csv"
SATCAT_URL = "https://celestrak.org/pub/satcat.csv"

# Output columns (renamed from SATCAT's upper-case headers).
COLUMNS = [
    ("NORAD_CAT_ID", "norad_id"),
    ("OBJECT_NAME", "name"),
    ("OBJECT_ID", "intl_designator"),
    ("OBJECT_TYPE", "object_type"),
    ("OPS_STATUS_CODE", "ops_status"),
    ("OWNER", "owner"),
    ("LAUNCH_DATE", "launch_date"),
    ("LAUNCH_SITE", "launch_site"),
    ("DECAY_DATE", "decay_date"),
    ("PERIOD", "period_min"),
    ("INCLINATION", "inclination_deg"),
    ("APOGEE", "apogee_km"),
    ("PERIGEE", "perigee_km"),
    ("RCS", "rcs_m2"),
]

_ALPHA5 = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # Alpha-5 skips I and O


def parse_catnum(field: str) -> int:
    """TLE catalogue number, including Alpha-5 ids (e.g. 'T0000' -> 270000)."""
    field = field.strip()
    if field and field[0].isalpha():
        return (_ALPHA5.index(field[0].upper()) + 10) * 10000 + int(field[1:])
    return int(field)


def tle_norad_ids(path: Path = TLE_FILE) -> set[int]:
    ids: set[int] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("1 "):
                try:
                    ids.add(parse_catnum(line[2:7]))
                except ValueError:
                    continue
    return ids


def load_satcat_text(input_path: str | None) -> str:
    if input_path:
        return Path(input_path).read_text(encoding="utf-8")
    request = urllib.request.Request(SATCAT_URL, headers={"User-Agent": "OrbitSentinel/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URL
        return response.read().decode("utf-8")


def build(input_path: str | None = None, out_path: Path = OUT_FILE) -> dict:
    wanted = tle_norad_ids()
    text = load_satcat_text(input_path)
    reader = csv.DictReader(io.StringIO(text))
    missing_cols = [src for src, _ in COLUMNS if src not in (reader.fieldnames or [])]
    if missing_cols:
        raise SystemExit(f"SATCAT format changed, missing columns: {missing_cols}")

    rows = []
    for row in reader:
        try:
            norad = int(row["NORAD_CAT_ID"])
        except (TypeError, ValueError):
            continue
        if norad in wanted:
            rows.append([row[src].strip() for src, _ in COLUMNS])
    rows.sort(key=lambda r: int(r[0]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(f"# source={SATCAT_URL} retrieved={stamp} rows={len(rows)} "
                     f"filter=norad ids in app/simulation/tle-data.txt ({len(wanted)})\n")
        writer = csv.writer(handle)
        writer.writerow([dst for _, dst in COLUMNS])
        writer.writerows(rows)
    return {"tle_ids": len(wanted), "rows": len(rows), "out": str(out_path), "retrieved": stamp}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="local satcat.csv instead of downloading")
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args(argv)
    info = build(args.input, Path(args.out))
    print(f"wrote {info['rows']} SATCAT rows for {info['tle_ids']} TLE ids -> {info['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
