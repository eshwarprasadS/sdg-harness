#!/usr/bin/env python3
"""Format OpenMathInstruct-2 and GSM8K data for the D-vs-B experiment."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import structlog

log = structlog.get_logger()


def load_dataset_flexible(path: Path):
    """Load a dataset from disk, trying HuggingFace format then JSONL/parquet."""
    from datasets import DatasetDict, load_from_disk

    try:
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            split = "train" if "train" in ds else list(ds.keys())[0]
            log.info("loaded_hf_dataset_dict", path=str(path), split=split)
            return ds[split]
        log.info("loaded_hf_dataset", path=str(path), num_rows=len(ds))
        return ds
    except Exception:
        pass

    parquet_files = list(path.glob("**/*.parquet"))
    if parquet_files:
        from datasets import Dataset

        ds = Dataset.from_parquet([str(f) for f in parquet_files])
        log.info(
            "loaded_parquet", path=str(path),
            num_files=len(parquet_files), num_rows=len(ds),
        )
        return ds

    jsonl_files = list(path.glob("**/*.jsonl")) + list(path.glob("**/*.json"))
    if jsonl_files:
        from datasets import load_dataset

        ds = load_dataset("json", data_files=[str(f) for f in jsonl_files], split="train")
        log.info(
            "loaded_jsonl", path=str(path),
            num_files=len(jsonl_files), num_rows=len(ds),
        )
        return ds

    raise FileNotFoundError(
        f"Could not load dataset from {path}. Expected HuggingFace dataset, "
        "parquet, or JSONL files."
    )


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the numeric answer after #### in GSM8K answer format."""
    match = re.search(r"####\s*(.+)", answer_text)
    if not match:
        raise ValueError(
            f"Could not extract #### answer from: {answer_text[:100]}"
        )
    raw = match.group(1).strip()
    return raw.replace(",", "")


def prepare_omi2(
    omi2_path: Path,
    num_samples: int,
    output_path: Path,
    seed: int = 42,
) -> int:
    """Load, subsample, and format OpenMathInstruct-2 data."""
    log.info("loading_omi2", path=str(omi2_path))
    ds = load_dataset_flexible(omi2_path)

    if num_samples > len(ds):
        log.warning(
            "requested_more_than_available",
            requested=num_samples, available=len(ds),
        )
        num_samples = len(ds)

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), num_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "w") as f:
        for i, idx in enumerate(indices):
            row = ds[idx]

            problem = row.get("problem", row.get("question", ""))
            solution = row.get("generated_solution", row.get("solution", ""))
            answer = row.get("expected_answer", row.get("answer", ""))

            if not problem or not answer:
                log.warning("skipping_empty_row", idx=idx)
                continue

            sample = {
                "prompt": str(problem),
                "expected_answer": str(answer).replace(",", ""),
                "solution": str(solution),
                "sample_id": f"omi2_{written:05d}",
            }
            f.write(json.dumps(sample) + "\n")
            written += 1

            if (i + 1) % 1000 == 0:
                log.info("omi2_progress", processed=i + 1, written=written)

    log.info(
        "omi2_complete",
        written=written, output=str(output_path),
    )
    return written


def prepare_gsm8k_test(
    gsm8k_path: Path,
    output_path: Path,
) -> int:
    """Load and format GSM8K test set."""
    log.info("loading_gsm8k", path=str(gsm8k_path))
    ds = load_dataset_flexible(gsm8k_path)

    from datasets import DatasetDict

    # If loaded as DatasetDict, get the test split
    if isinstance(ds, DatasetDict):
        if "test" in ds:
            ds = ds["test"]
        else:
            log.warning("no_test_split_found, using first split")
            ds = ds[list(ds.keys())[0]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    errors = 0

    with open(output_path, "w") as f:
        for i in range(len(ds)):
            row = ds[i]

            question = row.get("question", row.get("problem", ""))
            answer_text = row.get("answer", "")

            if not question:
                log.warning("skipping_empty_question", idx=i)
                continue

            try:
                numeric_answer = extract_gsm8k_answer(answer_text)
            except ValueError as exc:
                log.warning("answer_extraction_failed", idx=i, error=str(exc))
                errors += 1
                continue

            sample = {
                "prompt": str(question),
                "expected_answer": numeric_answer,
                "sample_id": f"gsm8k_test_{written:04d}",
            }
            f.write(json.dumps(sample) + "\n")
            written += 1

    log.info(
        "gsm8k_test_complete",
        written=written, errors=errors, output=str(output_path),
    )
    return written


def main(args: argparse.Namespace) -> None:
    omi2_path = Path(args.omi2_path).expanduser()
    gsm8k_path = Path(args.gsm8k_path).expanduser()
    output_omi2 = Path(args.output_omi2).expanduser()
    output_gsm8k = Path(args.output_gsm8k_test).expanduser()

    omi2_count = prepare_omi2(omi2_path, args.num_samples, output_omi2)
    gsm8k_count = prepare_gsm8k_test(gsm8k_path, output_gsm8k)

    print(f"\nOpenMathInstruct-2: {omi2_count} samples -> {output_omi2}")
    print(f"GSM8K test set: {gsm8k_count} samples -> {output_gsm8k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare OpenMathInstruct-2 and GSM8K data for experiment"
    )
    parser.add_argument(
        "--omi2-path", type=str,
        default="~/data/datasets/openmathinstruct2_1m",
        help="Path to OpenMathInstruct-2 dataset",
    )
    parser.add_argument(
        "--gsm8k-path", type=str,
        default="~/data/datasets/gsm8k",
        help="Path to GSM8K dataset",
    )
    parser.add_argument(
        "--num-samples", type=int, default=2500,
        help="Number of OMI2 samples to subsample (default: 2500)",
    )
    parser.add_argument(
        "--output-omi2", type=str,
        default="~/data/datasets/omi2_subset.jsonl",
        help="Output path for OMI2 subset JSONL",
    )
    parser.add_argument(
        "--output-gsm8k-test", type=str,
        default="~/data/datasets/gsm8k_test.jsonl",
        help="Output path for GSM8K test JSONL",
    )
    main(parser.parse_args())
