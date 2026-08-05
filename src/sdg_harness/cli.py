from __future__ import annotations

import click

from sdg_harness.eval.cli import eval_group


@click.group()
def cli() -> None:
    """SDG Harness -- Agentic SDG optimization."""


cli.add_command(eval_group)
