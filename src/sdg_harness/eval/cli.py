from __future__ import annotations

import asyncio
import json
import sys

import click

from sdg_harness.eval.cost import CostTracker
from sdg_harness.eval.models import L1Config
from sdg_harness.eval.weak_strong import WeakStrongEvaluator


@click.group("eval")
def eval_group() -> None:
    """Evaluation commands for SDG quality measurement."""


@eval_group.command("l1")
@click.option(
    "--samples",
    required=True,
    type=click.Path(exists=True),
    help="Path to JSONL samples file",
)
@click.option(
    "--weak-model",
    required=True,
    help="litellm model string for weak model",
)
@click.option(
    "--strong-model",
    required=True,
    help="litellm model string for strong model",
)
@click.option(
    "--rollouts",
    default=3,
    type=int,
    help="Number of rollouts per model per sample",
)
@click.option(
    "--max-cost",
    default=5.0,
    type=float,
    help="Maximum cost budget in USD",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Path to write JSON results",
)
@click.option(
    "--temperature",
    default=1.0,
    type=float,
    help="Sampling temperature for LLM calls",
)
def l1_command(
    samples: str,
    weak_model: str,
    strong_model: str,
    rollouts: int,
    max_cost: float,
    output_path: str,
    temperature: float,
) -> None:
    """Run L1 weak-strong evaluation on JSONL samples."""
    sample_list: list[dict[str, str]] = []
    with open(samples) as f:
        for line in f:
            line = line.strip()
            if line:
                sample_list.append(json.loads(line))

    config = L1Config(
        weak_model=weak_model,
        strong_model=strong_model,
        num_rollouts=rollouts,
        temperature=temperature,
        max_cost=max_cost,
    )
    cost_tracker = CostTracker(max_cost=max_cost)
    evaluator = WeakStrongEvaluator(
        config=config, cost_tracker=cost_tracker
    )

    result = asyncio.run(evaluator.evaluate_batch(sample_list))

    with open(output_path, "w") as f:
        f.write(result.model_dump_json(indent=2))

    min_pass_rate = 0.0
    if result.pass_rate < min_pass_rate:
        sys.exit(1)
