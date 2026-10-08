"""Shared pytest fixtures for the backend regression suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# main.py loads backend/.env on import. A developer's local .env typically sets
# ENABLE_EXTENDED_PIPELINE=1, which would flip the defaults several tests assert
# on and eagerly build the ML runtime. Tests always run against code defaults.
os.environ["ORBIT_SENTINEL_SKIP_DOTENV"] = "1"
