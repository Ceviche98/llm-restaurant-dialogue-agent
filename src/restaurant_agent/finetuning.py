"""Parameter-Efficient Fine-Tuning (PEFT / LoRA & Full SFT) for conversational LLMs (Lab 6)."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List


DEFAULT_BASE_MODEL = "google/gemma-3-270m-it"
DEFAULT_DATASET_NAME = "HuggingFaceTB/smoltalk"
DEFAULT_DATASET_CONFIG = "everyday-conversations"


@dataclass
class SFTTrainingSpec:
    """Configuration recipe for Supervised Fine-Tuning (8-bit constrained vs unquantized bfloat16)."""

    output_dir: str = "sft_output"
    max_length: int = 256
    dataset_text_field: str = "messages"
    packing: bool = False
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    fp16: bool = False
    bf16: bool = True
    optim: str = "adamw_bnb_8bit"
    gradient_checkpointing: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 100
    per_device_eval_batch_size: int = 1
    eval_accumulation_steps: int = 4
    prediction_loss_only: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class LoRATrainingSpec:
    """Configuration recipe for Low-Rank Adaptation (PEFT / LoRA) on causal dialogue LLMs."""

    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    output_dir: str = "lora_output"
    max_length: int = 256
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    bf16: bool = True
    optim: str = "paged_adamw_32bit"
    gradient_checkpointing: bool = True
    eval_steps: int = 100
    logging_steps: int = 10

    def to_peft_kwargs(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": list(self.target_modules),
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
        }

    def to_sft_kwargs(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "max_length": self.max_length,
            "dataset_text_field": "messages",
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "bf16": self.bf16,
            "optim": self.optim,
            "gradient_checkpointing": self.gradient_checkpointing,
            "eval_strategy": "steps",
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "save_strategy": "epoch",
        }


def normalize_answer(s: str) -> str:
    """Normalize text for lexical token-overlap evaluation (lowercase, strip punc/articles)."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute normalized token-level F1 score between prediction and reference."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common_tokens = pred_counter & gold_counter
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def evaluate_predictions(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """Compute corpus-level average Token F1 (0-100 scale) and n-gram BLEU score."""
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length.")
    if not predictions:
        return {"average_f1": 0.0, "bleu": 0.0, "total": 0, "f1_scores": []}

    f1_scores = [compute_f1(pred, gold) for pred, gold in zip(predictions, ground_truths)]
    avg_f1 = (sum(f1_scores) / len(f1_scores)) * 100.0

    try:
        import sacrebleu

        bleu_score = float(sacrebleu.corpus_bleu(predictions, [ground_truths]).score)
    except ImportError:
        bleu_score = 0.0

    return {
        "average_f1": round(avg_f1, 4),
        "bleu": round(bleu_score, 4),
        "total": len(predictions),
        "f1_scores": f1_scores,
    }


BENCHMARK_RESULTS = {
    "rag_hybriddialogue": {
        "dataset": "HybridDialogue (N=243 test turns)",
        "backbone": "Qwen/Qwen3-1.7B",
        "retriever": "facebook/contriever-msmarco (2020 Wikipedia dump, top-3)",
        "baseline_no_rag": {"f1": 23.20, "bleu": 1.92},
        "dense_rag_top3": {"f1": 29.77, "bleu": 4.28},
    },
    "smoltalk_finetuning": {
        "dataset": "HuggingFaceTB/smoltalk (everyday-conversations, N=119 test)",
        "backbone": "google/gemma-3-270m-it",
        "zero_shot_baseline": {"f1": 22.97, "bleu": 4.34},
        "sft_8bit_8gb_vram": {"f1": 31.69, "bleu": 11.66, "final_val_loss": 1.0330},
        "lora_r16_alpha32_8gb_vram": {"f1": 31.88, "bleu": 10.25, "final_val_loss": 1.1439},
        "sft_unquantized_bf16_24gb_l4": {"f1": 32.70, "bleu": 11.68, "final_train_loss": 1.1362},
    },
}
