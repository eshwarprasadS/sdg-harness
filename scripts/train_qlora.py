#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3.5-4B on a JSONL dataset."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import structlog
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

log = structlog.get_logger()


def load_and_split_data(data_path: str):
    """Load JSONL data and split into train/eval."""
    ds = load_dataset("json", data_files=data_path, split="train")
    log.info("loaded_training_data", num_samples=len(ds), path=data_path)

    split = ds.train_test_split(test_size=0.1, seed=42)
    log.info(
        "data_split",
        train=len(split["train"]), eval=len(split["test"]),
    )
    return split["train"], split["test"]


def main(args: argparse.Namespace) -> None:
    model_path = str(Path(args.model_path).expanduser())
    output_dir = str(Path(args.output).expanduser())
    data_path = str(Path(args.data).expanduser())

    log.info(
        "training_start",
        model_path=model_path,
        data=data_path,
        output=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    log.info("loading_model", model_path=model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    train_ds, eval_ds = load_and_split_data(data_path)

    def formatting_func(row):
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["solution"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # SFTConfig only accepts warmup_steps (not warmup_ratio), compute from total steps
    gradient_accumulation_steps = 4
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * gradient_accumulation_steps))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)

    training_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field=None,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    log.info("starting_trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    log.info("training_complete", **metrics)

    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete. Adapter saved to: {output_dir}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune a model on JSONL data"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="~/data/models/Qwen3.5-4B",
        help="Path to base model",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--output", type=str,
        default="~/data/checkpoints/experiment",
        help="Output directory for adapter",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device batch size (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    main(parser.parse_args())
