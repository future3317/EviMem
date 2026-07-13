"""Supervised and verifier-shaped RL interfaces for the EviMem controller."""

from .codec import ActionCodec, ActionDecodingError, PolicyPromptRenderer
from .dataset import (
    OracleActionExample,
    OracleTrajectoryCompiler,
    build_grpo_dataset,
    build_sft_dataset,
    load_examples_jsonl,
    split_examples_by_document,
    write_examples_jsonl,
)
from .policy import GenerationConfig, LearnedController, TransformersTextGenerator
from .rewards import (
    ActionRewardOracle,
    CachedVerifierOracle,
    CertifiedRewardRecord,
    VerifierRewardAdapter,
    load_reward_records_jsonl,
    write_reward_records_jsonl,
)
from .trainers import (
    GRPOJobConfig,
    LoraSpec,
    SFTJobConfig,
    build_grpo_args,
    build_grpo_trainer,
    build_lora_config,
    build_sft_args,
    build_sft_trainer,
)

__all__ = [
    "ActionCodec",
    "ActionDecodingError",
    "ActionRewardOracle",
    "CachedVerifierOracle",
    "CertifiedRewardRecord",
    "GRPOJobConfig",
    "GenerationConfig",
    "LearnedController",
    "LoraSpec",
    "OracleActionExample",
    "OracleTrajectoryCompiler",
    "PolicyPromptRenderer",
    "SFTJobConfig",
    "TransformersTextGenerator",
    "VerifierRewardAdapter",
    "build_grpo_args",
    "build_grpo_dataset",
    "build_grpo_trainer",
    "build_lora_config",
    "build_sft_args",
    "build_sft_dataset",
    "build_sft_trainer",
    "load_examples_jsonl",
    "load_reward_records_jsonl",
    "split_examples_by_document",
    "write_examples_jsonl",
    "write_reward_records_jsonl",
]
