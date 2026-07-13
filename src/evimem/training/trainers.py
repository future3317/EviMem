"""Maintained-library configuration for supervised retriever and manager training."""

from __future__ import annotations

from dataclasses import dataclass


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
class ManagerSFTConfig:
    output_dir: str
    learning_rate: float = 2e-4
    epochs: float = 3.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42
    bf16: bool = True


@dataclass(frozen=True)
class RetrieverTrainingConfig:
    output_dir: str
    learning_rate: float = 2e-5
    epochs: float = 3.0
    per_device_batch_size: int = 16
    warmup_ratio: float = 0.1
    seed: int = 42
    bf16: bool = True


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


def build_manager_training_args(config: ManagerSFTConfig):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy="steps",
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        bf16=config.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=True,
    )


def build_retriever_training_args(config: RetrieverTrainingConfig):
    from sentence_transformers import SentenceTransformerTrainingArguments

    return SentenceTransformerTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        warmup_ratio=config.warmup_ratio,
        report_to="none",
        seed=config.seed,
        bf16=config.bf16,
    )
