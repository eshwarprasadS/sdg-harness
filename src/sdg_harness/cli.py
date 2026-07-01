from __future__ import annotations

import json
import sys
from typing import Any

import click
import structlog

from sdg_harness.core.config import IterationConfig
from sdg_harness.core.record import IterationRecord
from sdg_harness.inner_loop.base import InnerLoopRunner
from sdg_harness.inner_loop.generic import GenericRunner
from sdg_harness.inner_loop.toy_task import ToyTaskRunner
from sdg_harness.loop.controller import LoopController
from sdg_harness.tracking.metrics import MetricsTracker
from sdg_harness.tracking.storage import TrajectoryStorage

logger = structlog.get_logger()


def _default_analyzer(
    config: IterationConfig, result: Any,
) -> Any:
    from sdg_harness.core.analysis import AnalysisReport

    return AnalysisReport(
        summary="auto-generated analysis",
        data_quality_issues=[],
        training_issues=[],
        eval_breakdown=result.metrics,
        strengths=[],
        weaknesses=[],
        root_causes=[],
        suggested_directions=[],
    )


def _default_proposer(
    config: IterationConfig, analysis: Any,
) -> Any:
    from sdg_harness.core.proposal import Proposal

    return Proposal(
        changes={},
        rationale={},
        expected_effect="no changes",
        risk="none",
    )


def _max_iterations_stop(max_iter: int) -> Any:
    def _stop(records: list[IterationRecord]) -> bool:
        return len(records) >= max_iter

    return _stop


@click.group()
def main() -> None:
    """SDG Harness - Agentic SDG optimization harness."""


@main.command()
@click.option("--runner", type=click.Choice(["toy", "generic"]), default="toy")
@click.option("--max-iterations", type=int, default=3)
@click.option("--output-dir", type=str, default=".sdg-runs")
@click.option("--run-id", type=str, default=None)
@click.option("--config-file", type=click.Path(exists=True), default=None)
@click.option("--command", type=str, default=None, multiple=True)
def run(
    runner: str,
    max_iterations: int,
    output_dir: str,
    run_id: str | None,
    config_file: str | None,
    command: tuple[str, ...],
) -> None:
    """Run an optimization loop."""
    import time

    if run_id is None:
        run_id = f"run-{int(time.time())}"

    config_dict: dict[str, Any] = {}
    if config_file:
        with open(config_file) as f:
            config_dict = json.load(f)

    initial_config = IterationConfig(
        config=config_dict.get("config", {"num_samples": 10, "difficulty": 1.0}),
        mutable_keys=config_dict.get("mutable_keys", ["num_samples", "difficulty"]),
        fixed_keys=config_dict.get("fixed_keys", []),
        metadata=config_dict.get("metadata", {}),
    )

    inner_runner: InnerLoopRunner
    if runner == "toy":
        inner_runner = ToyTaskRunner()
    elif runner == "generic":
        if not command:
            click.echo("Error: --command is required for generic runner", err=True)
            sys.exit(1)
        inner_runner = GenericRunner(command=list(command))
    else:
        click.echo(f"Error: unknown runner '{runner}'", err=True)
        sys.exit(1)

    tracker = MetricsTracker()
    storage = TrajectoryStorage(output_dir)

    controller = LoopController(
        runner=inner_runner,
        analyzer=_default_analyzer,
        proposer=_default_proposer,
        should_stop=_max_iterations_stop(max_iterations),
        initial_config=initial_config,
    )

    logger.info("cli_run_started", run_id=run_id, runner=runner)
    result = controller.run(
        task_description=f"CLI run with {runner} runner",
        pipeline_info={"runner": runner, "max_iterations": max_iterations},
    )

    for record in result.iterations:
        if record.result is not None:
            tracker.record(
                iteration_id=record.iteration_id,
                cost=record.result.cost,
                tokens_in=0,
                tokens_out=0,
                latency_seconds=record.result.duration_seconds,
            )

    path = storage.save(run_id, result)
    summary = tracker.summary()
    n_iter = len(result.iterations)
    click.echo(f"Run '{run_id}' completed: {n_iter} iterations")
    best = result.best_iteration
    score = result.best_score
    click.echo(f"Best iteration: {best} (score: {score:.4f})")
    click.echo(f"Total cost: {summary['total_cost']:.4f}")
    click.echo(f"Saved to: {path}")


@main.command()
@click.option("--output-dir", type=str, default=".sdg-runs")
@click.argument("run_id")
def resume(output_dir: str, run_id: str) -> None:
    """Resume a previous run (loads trajectory state)."""
    storage = TrajectoryStorage(output_dir)
    try:
        result = storage.load(run_id)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    n_iter = len(result.iterations)
    click.echo(f"Loaded run '{run_id}': {n_iter} iterations completed")
    best = result.best_iteration
    score = result.best_score
    click.echo(f"Best iteration: {best} (score: {score:.4f})")
    click.echo(f"Stop reason: {result.stop_reason}")


@main.command()
@click.option("--output-dir", type=str, default=".sdg-runs")
@click.argument("run_id", required=False)
def inspect(output_dir: str, run_id: str | None) -> None:
    """Inspect run results. Lists all runs if no run_id given."""
    storage = TrajectoryStorage(output_dir)

    if run_id is None:
        runs = storage.list_runs()
        if not runs:
            click.echo("No runs found.")
            return
        click.echo("Available runs:")
        for r in runs:
            click.echo(f"  {r}")
        return

    try:
        result = storage.load(run_id)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Run: {run_id}")
    click.echo(f"Iterations: {len(result.iterations)}")
    click.echo(f"Best iteration: {result.best_iteration}")
    click.echo(f"Best score: {result.best_score:.4f}")
    click.echo(f"Total cost: {result.total_cost:.4f}")
    click.echo(f"Stop reason: {result.stop_reason}")

    for record in result.iterations:
        metrics_str = ""
        if record.result is not None:
            metrics_str = json.dumps(record.result.metrics, indent=None)
        click.echo(f"  [{record.iteration_id}] metrics={metrics_str}")
