# Factory Configuration -- sdg-harness

## Goal
Build an agentic SDG optimization harness where LLM agents drive the search for better synthetic data generation configurations. A Planner agent analyzes the pipeline and infers what can change. An inner loop executes generate-train-evaluate. An Analyst agent reads actual data samples, training curves, and eval breakdowns to produce a structured diagnosis. A Proposer agent translates that diagnosis into concrete config changes. A Loop Controller orchestrates iterations until a stopping condition is met.

## Language
Python 3.11+

## Scope
- `src/sdg_harness/` -- all application source code
- `tests/` -- test suite
- `eval/` -- evaluation harness
- `examples/` -- example scripts and toy tasks
- `pyproject.toml` -- project metadata and dependencies

## Guards
- Do not delete or overwrite existing tests
- Do not modify files outside the declared scope
- Do not introduce secrets or credentials into the repo
- Do not lower the eval threshold

## Command
python eval/score.py

## Threshold
0.5

## Smoke Test
python -c 'import json; print(json.dumps({"status": "ok"}))'

## Target Branch
main

## Eval Spec
- Build and run the project's primary entry point without errors

## Eval Profile
Dimensions (from eval_profile.json):
- **syntax_check** (weight: 0.83) -- Verify code has no syntax errors [parser: exit_code]
- **observability** (weight: 0.17) -- Analyze logging coverage, structured logging, and request tracing [parser: json]

Tier: fallback | Confidence: 0.2 | Human reviewed: yes

## Key Libraries
- `pydantic` -- config validation and serialization
- `structlog` -- structured logging for trajectory analysis
- `click` -- CLI interface
- `litellm` -- LLM provider abstraction for agent calls
- `pytest` + `ruff` + `mypy` -- quality tooling
