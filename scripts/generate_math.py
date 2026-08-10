#!/usr/bin/env python3
"""Generate GSM8K-style math word problems using GPT-4o via litellm."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import litellm
import structlog

log = structlog.get_logger()

SYSTEM_PROMPT = (
    "You are a math problem generator. Generate unique grade school math "
    "word problems with step-by-step solutions."
)

GENERATION_PROMPT = (
    "Generate a unique grade school math word problem requiring 2-4 step "
    "arithmetic reasoning. The problem should have a specific numeric answer. "
    "Output format:\n"
    "QUESTION: <the problem>\n"
    "SOLUTION: <step by step>\n"
    "ANSWER: <single number>"
)


def parse_response(text: str) -> dict[str, str] | None:
    """Extract question, solution, and numeric answer from LLM response."""
    q_match = re.search(r"QUESTION:\s*(.+?)(?=\nSOLUTION:)", text, re.DOTALL)
    s_match = re.search(r"SOLUTION:\s*(.+?)(?=\nANSWER:)", text, re.DOTALL)
    a_match = re.search(r"ANSWER:\s*([^\n]+)", text)

    if not (q_match and s_match and a_match):
        return None

    question = q_match.group(1).strip()
    solution = s_match.group(1).strip()
    answer_raw = a_match.group(1).strip()

    answer_clean = re.sub(r"[,$\s]", "", answer_raw)
    try:
        float(answer_clean)
    except ValueError:
        return None

    return {"question": question, "solution": solution, "answer": answer_clean}


async def generate_one(
    sem: asyncio.Semaphore,
    model: str,
    idx: int,
    stats: dict,
) -> dict | None:
    """Generate a single math problem via the LLM API."""
    async with sem:
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": GENERATION_PROMPT},
                ],
                temperature=1.0,
                max_tokens=1024,
            )
        except Exception as exc:
            log.warning("api_call_failed", idx=idx, error=str(exc))
            return None

    content = response.choices[0].message.content or ""

    try:
        cost = litellm.completion_cost(completion_response=response)
        stats["cost"] += cost
    except Exception:
        pass

    if response.usage:
        stats["prompt_tokens"] += response.usage.prompt_tokens or 0
        stats["completion_tokens"] += response.usage.completion_tokens or 0

    parsed = parse_response(content)
    if parsed is None:
        log.warning("parse_failed", idx=idx)
        return None

    return {
        "prompt": parsed["question"],
        "expected_answer": parsed["answer"],
        "solution": parsed["solution"],
        "sample_id": f"gen_{idx:04d}",
    }


async def run(args: argparse.Namespace) -> None:
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(20)
    stats = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    target = args.num_samples
    oversample = int(target * 1.2)

    log.info(
        "generation_start",
        target=target,
        oversample_attempts=oversample,
        model=args.model,
    )

    results: list[dict] = []
    batch_size = 100

    for batch_start in range(0, oversample, batch_size):
        if len(results) >= target:
            break

        batch_end = min(batch_start + batch_size, oversample)
        tasks = [
            generate_one(sem, args.model, i, stats)
            for i in range(batch_start, batch_end)
        ]
        batch_results = await asyncio.gather(*tasks)

        for r in batch_results:
            if r is not None and len(results) < target:
                results.append(r)

        log.info(
            "progress",
            generated=len(results),
            target=target,
            attempts=batch_end,
            cost=f"${stats['cost']:.4f}",
        )

    with open(output_path, "w") as f:
        for sample in results:
            f.write(json.dumps(sample) + "\n")

    log.info(
        "generation_complete",
        num_samples=len(results),
        output=str(output_path),
        total_cost=f"${stats['cost']:.4f}",
        prompt_tokens=stats["prompt_tokens"],
        completion_tokens=stats["completion_tokens"],
    )

    print(f"\nGenerated {len(results)} samples -> {output_path}")
    print(f"Total cost: ${stats['cost']:.4f}")
    print(
        f"Tokens: {stats['prompt_tokens']} prompt, "
        f"{stats['completion_tokens']} completion"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GSM8K-style math word problems"
    )
    parser.add_argument(
        "--num-samples", type=int, default=2500,
        help="Number of samples to generate (default: 2500)",
    )
    parser.add_argument(
        "--output", type=str,
        default="~/data/datasets/generated_math.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="LLM model for generation (default: gpt-4o)",
    )
    asyncio.run(run(parser.parse_args()))
