"""Tiny JSON file cache with an age limit."""

# ── In plain words ───────────────────────────────────────────────────────────
# A tiny helper: save a Python dict as a JSON file and read it back, ignoring the
# file if it is older than a given number of days. Used for the star-fact cache
# (data/cache/resolver/) and the sector lists.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def read_json(path: Path, max_age_days: float | None = None) -> Any | None:
    """Return cached content, or None if missing, unreadable or too old."""
    try:
        if max_age_days is not None and time.time() - path.stat().st_mtime > max_age_days * 86400:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, content: Any) -> None:
    """Write atomically so a crash never leaves a half-written cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
