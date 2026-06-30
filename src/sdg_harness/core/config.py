from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class IterationConfig(BaseModel):
    config: dict[str, Any]
    mutable_keys: list[str]
    fixed_keys: list[str]
    metadata: dict[str, Any] = {}
