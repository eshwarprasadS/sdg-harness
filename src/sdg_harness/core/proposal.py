from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Proposal(BaseModel):
    changes: dict[str, Any]
    rationale: dict[str, str]
    expected_effect: str
    risk: str
