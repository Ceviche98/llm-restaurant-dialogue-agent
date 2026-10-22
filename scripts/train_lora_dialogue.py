#!/usr/bin/env python3
"""Configure and launch Supervised Fine-Tuning (SFT) or LoRA PEFT on Gemma-3-270M-it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restaurant_agent import BENCHMARK_RESULTS, LoRATrainingSpec, SFTTrainingSpec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure or launch Full SFT / LoRA PEFT on Gemma-3-270M-it (SmolTalk)."
    )
    parser.add_argument(
        "--mode",
        choices=["lora", "sft-8bit", "sft-unquantized", "report-benchmarks"],
        default="report-benchmarks",
        help="Training configuration mode or benchmark report display",
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA adapter rank r")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA scaling factor alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    if args.mode == "report-benchmarks":
        print(json.dumps(BENCHMARK_RESULTS, indent=2))
        return

    if args.mode == "lora":
        spec = LoRATrainingSpec(r=args.rank, lora_alpha=args.alpha, num_train_epochs=args.epochs)
        print("LoRA PEFT Configuration:")
        print(json.dumps({"peft_config": spec.to_peft_kwargs(), "sft_config": spec.to_sft_kwargs()}, indent=2))
    elif args.mode == "sft-8bit":
        spec_sft = SFTTrainingSpec(num_train_epochs=args.epochs)
        print("8-Bit Constrained SFT Configuration (8GB VRAM):")
        print(json.dumps(spec_sft.to_dict(), indent=2))
    elif args.mode == "sft-unquantized":
        spec_unq = SFTTrainingSpec(
            output_dir="sft_unquantized_output",
            max_length=512,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            optim="adamw_torch_fused",
            num_train_epochs=args.epochs,
        )
        print("Unquantized bfloat16 SFT Configuration (24GB L4 Ceiling):")
        print(json.dumps(spec_unq.to_dict(), indent=2))


if __name__ == "__main__":
    main()
