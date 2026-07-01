from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from sdg_harness.loop.trajectory import TrajectoryResult

logger = structlog.get_logger()


class TrajectoryStorage:

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, result: TrajectoryResult) -> Path:
        path = self._base_dir / f"{run_id}.json"
        data = result.model_dump()
        path.write_text(json.dumps(data, indent=2))
        logger.info("trajectory_saved", run_id=run_id, path=str(path))
        return path

    def load(self, run_id: str) -> TrajectoryResult:
        path = self._base_dir / f"{run_id}.json"
        if not path.exists():
            msg = f"Run '{run_id}' not found at {path}"
            raise FileNotFoundError(msg)
        data: dict[str, Any] = json.loads(path.read_text())
        logger.info("trajectory_loaded", run_id=run_id, path=str(path))
        return TrajectoryResult.model_validate(data)

    def list_runs(self) -> list[str]:
        runs = sorted(
            p.stem
            for p in self._base_dir.glob("*.json")
            if p.is_file()
        )
        logger.info("runs_listed", count=len(runs))
        return runs
