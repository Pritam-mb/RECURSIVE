from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def _resolve_artifact_path(filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return ARTIFACT_DIR / path


def artifact_path(filename: str) -> Path:
    return _resolve_artifact_path(filename)


def ensure_artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def load_artifact(filename: str) -> dict[str, Any] | None:
    path = _resolve_artifact_path(filename)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_artifact(filename: str, payload: dict[str, Any]) -> Path:
    ensure_artifact_dir()
    path = _resolve_artifact_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
