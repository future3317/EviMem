"""Command-line entry points for LoRA SFT and verifier-shaped GRPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import build_grpo_dataset, build_sft_dataset, load_examples_jsonl
from .rewards import CachedVerifierOracle, load_reward_records_jsonl
from .trainers import (
    GRPOJobConfig,
    LoraSpec,
    SFTJobConfig,
    build_grpo_trainer,
    build_sft_trainer,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HF model ID or local model path")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--resume-from-checkpoint")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evimem-train",
        description="Train an EviMem next-action controller with maintained HF libraries.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sft = commands.add_parser("sft", help="LoRA next-action imitation training")
    _common(sft)
    sft.add_argument("--max-length", type=int, default=2048)

    grpo = commands.add_parser("grpo", help="Verifier-shaped constrained GRPO")
    _common(grpo)
    grpo.add_argument("--rewards-jsonl", required=True, type=Path)
    grpo.add_argument("--num-generations", type=int, default=4)
    grpo.add_argument("--max-completion-length", type=int, default=256)
    grpo.add_argument("--beta", type=float, default=0.02)
    grpo.add_argument("--epsilon", type=float, default=0.2)
    return parser


def _lora(args: argparse.Namespace) -> LoraSpec:
    return LoraSpec(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )


def _load_optional_examples(path: Path | None):
    if path is None:
        return None
    return load_examples_jsonl(path)


def _run_sft(args: argparse.Namespace) -> None:
    train = build_sft_dataset(load_examples_jsonl(args.train_jsonl))
    eval_examples = _load_optional_examples(args.eval_jsonl)
    evaluation = build_sft_dataset(eval_examples) if eval_examples is not None else None
    config = SFTJobConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate or 2e-4,
        epochs=args.epochs or 3.0,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        seed=args.seed,
        bf16=not args.no_bf16,
    )
    trainer = build_sft_trainer(
        model=args.model,
        train_dataset=train,
        eval_dataset=evaluation,
        config=config,
        lora=_lora(args),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))


def _run_grpo(args: argparse.Namespace) -> None:
    train_examples = load_examples_jsonl(args.train_jsonl)
    train = build_grpo_dataset(train_examples)
    eval_examples = _load_optional_examples(args.eval_jsonl)
    evaluation = build_grpo_dataset(eval_examples) if eval_examples is not None else None
    reward_records = load_reward_records_jsonl(args.rewards_jsonl)
    oracle = CachedVerifierOracle(reward_records, examples=train_examples)
    config = GRPOJobConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate or 1e-6,
        epochs=args.epochs or 1.0,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        epsilon=args.epsilon,
        seed=args.seed,
        bf16=not args.no_bf16,
    )
    trainer = build_grpo_trainer(
        model=args.model,
        train_dataset=train,
        eval_dataset=evaluation,
        reward_oracle=oracle,
        config=config,
        lora=_lora(args),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "sft":
        _run_sft(args)
    else:
        _run_grpo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
