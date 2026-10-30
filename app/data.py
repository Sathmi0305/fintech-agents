"""Sample data source. In production these lookups would hit a credit bureau API,
an open-banking aggregator and a core banking system; here they read a JSON file."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "applicants.json"


@lru_cache(maxsize=1)
def _load() -> dict[str, dict]:
    with open(DATA_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return {a["applicant_id"]: a for a in raw["applicants"]}


def get_applicant(applicant_id: str) -> dict | None:
    return _load().get(applicant_id.strip().upper())


def list_applicants() -> list[dict]:
    return list(_load().values())
