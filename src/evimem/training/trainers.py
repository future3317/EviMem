"""Maintained-library trainer factories for LoRA SFT and constrained GRPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rewards import ActionRewardOracle, VerifierRewardAdapter


@dataclass(frozen=True)
class LoraSpec:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class SFTJobConfig:
    output_dir: str
    learning_rate: float = 2e-4
    epochs: float = 3.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class GRPOJobConfig:
    output_dir: str
    learning_rate: float = 1e-6
    epochs: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 4
    max_completion_length: int = 256
    beta: float = 0.02
    epsilon: float = 0.2
    logging_steps: int = 5
    save_steps: int = 100
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True


def build_lora_config(spec: LoraSpec | None = None):
    from peft import LoraConfig, TaskType

    selected = spec or LoraSpec()
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=selected.rank,
        lora_alpha=selected.alpha,
        lora_dropout=selected.dropout,
        target_modules=list(selected.target_modules),
        bias="none",
    )


def build_sft_args(config: SFTJobConfig):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        completion_only_loss=True,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy="steps",
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=True,
    )


def build_grpo_args(config: GRPOJobConfig):
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        beta=config.beta,
        epsilon=config.epsilon,
        scale_rewards="group",
        loss_type="grpo",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy="steps",
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
    )


def build_sft_trainer(
    *,
    model: str | Any,
    train_dataset: Any,
    config: SFTJobConfig,
    eval_dataset: Any | None = None,
    processing_class: Any | None = None,
    lora: LoraSpec | None = None,
):
    from trl import SFTTrainer

    return SFTTrainer(
        model=model,
        args=build_sft_args(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=build_lora_config(lora),
    )


def build_grpo_trainer(
    *,
    model: str | Any,
    train_dataset: Any,
    reward_oracle: ActionRewardOracle,
    config: GRPOJobConfig,
    eval_dataset: Any | None = None,
    processing_class: Any | None = None,
    lora: LoraSpec | None = None,
):
    from trl import GRPOTrainer

    return GRPOTrainer(
        model=model,
        reward_funcs=VerifierRewardAdapter(reward_oracle),
        args=build_grpo_args(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=build_lora_config(lora),
    )

