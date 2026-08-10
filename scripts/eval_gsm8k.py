#!/usr/bin/env python3
"""Evaluate a model (optionally with LoRA adapter) on GSM8K test set."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import structlog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = structlog.get_logger()


def extract_answer(text: str) -> str | None:
    """Extract numeric answer from model response.

    Tries #### pattern, then \\boxed{}, then last number in text.
    Strips <think>...</think> blocks from Qwen3 reasoning models.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", cleaned)
    if match:
        return match.group(1).replace(",", "")

    match = re.search(r"\\boxed\{([^}]+)\}", cleaned)
    if match:
        inner = match.group(1).strip().replace(",", "")
        if re.fullmatch(r"[+-]?\d+\.?\d*", inner):
            return inner

    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", cleaned)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def answers_match(predicted: str | None, expected: str) -> bool:
    """Compare predicted and expected answers numerically."""
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(expected)) < 1e-6
    except ValueError:
        return predicted.strip() == expected.strip()


def load_test_data(path: str) -> list[dict]:
    """Load JSONL test data."""
    samples = []
    with open(Path(path).expanduser()) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    log.info("loaded_test_data", num_samples=len(samples), path=path)
    return samples


def format_prompts(samples: list[dict], tokenizer) -> list[str]:
    """Format test samples as chat prompts."""
    prompts = []
    for s in samples:
        messages = [{"role": "user", "content": s["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)
    return prompts


def generate_vllm(
    model_path: str,
    adapter_path: str | None,
    prompts: list[str],
) -> list[str]:
    """Generate responses using vLLM."""
    from vllm import LLM, SamplingParams

    kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "dtype": "bfloat16",
    }

    lora_request = None
    if adapter_path:
        from vllm.lora.request import LoRARequest
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = 64
        lora_request = LoRARequest("adapter", 1, adapter_path)

    llm = LLM(**kwargs)
    sampling_params = SamplingParams(
        temperature=0, max_tokens=1024,
    )

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]


def generate_hf(
    model_path: str,
    adapter_path: str | None,
    prompts: list[str],
    tokenizer,
    batch_size: int,
) -> list[str]:
    """Generate responses using HuggingFace transformers."""
    log.info("loading_hf_model", model_path=model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        log.info("loading_adapter", adapter_path=adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            responses.append(text)

        log.info(
            "hf_generate_progress",
            completed=min(i + batch_size, len(prompts)),
            total=len(prompts),
        )

    return responses


def main(args: argparse.Namespace) -> None:
    model_path = str(Path(args.model_path).expanduser())
    adapter_path = (
        str(Path(args.adapter_path).expanduser())
        if args.adapter_path else None
    )
    output_path = Path(args.output).expanduser()

    samples = load_test_data(args.test_data)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
    )
    prompts = format_prompts(samples, tokenizer)

    log.info(
        "eval_start",
        model_path=model_path,
        adapter_path=adapter_path,
        num_samples=len(samples),
    )

    # Try vLLM first, fall back to HF generate
    try:
        log.info("attempting_vllm")
        responses = generate_vllm(model_path, adapter_path, prompts)
        log.info("vllm_generation_complete")
    except (ImportError, Exception) as exc:
        log.info("vllm_unavailable_or_failed", error=str(exc))
        log.info("falling_back_to_hf_generate")
        responses = generate_hf(
            model_path, adapter_path, prompts, tokenizer, args.batch_size,
        )

    correct = 0
    per_sample_results = []

    for sample, response in zip(samples, responses):
        predicted = extract_answer(response)
        expected = sample["expected_answer"]
        is_correct = answers_match(predicted, expected)
        if is_correct:
            correct += 1

        per_sample_results.append({
            "sample_id": sample.get("sample_id", ""),
            "correct": is_correct,
            "predicted": predicted or "",
            "expected": expected,
        })

    total = len(samples)
    accuracy = correct / total if total > 0 else 0.0

    results = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "results": per_sample_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(
        "eval_complete",
        accuracy=accuracy,
        correct=correct,
        total=total,
        output=str(output_path),
    )

    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on GSM8K test set"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="~/data/models/Qwen3.5-4B",
        help="Path to base model",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
        help="Path to LoRA adapter (optional)",
    )
    parser.add_argument(
        "--test-data", type=str, required=True,
        help="Path to GSM8K test JSONL file",
    )
    parser.add_argument(
        "--output", type=str,
        default="~/data/results/eval_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for HF generate (default: 8)",
    )
    main(parser.parse_args())
