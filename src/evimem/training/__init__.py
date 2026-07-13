"""Supervised learning interfaces for EviMem retrieval and memory management."""

from .dataset import (
    ManagerTrainingExample,
    RetrievalTrainingExample,
    load_manager_examples_jsonl,
    require_official_training_splits,
    write_examples_jsonl,
)
from .manager import (
    ManagerActionCodec,
    ManagerDecodingError,
    ManagerInput,
    StructuredMemoryManager,
    TextGenerator,
)
from .trainers import (
    LoraSpec,
    ManagerSFTConfig,
    RetrieverTrainingConfig,
    build_lora_config,
    build_manager_training_args,
    build_retriever_training_args,
)

__all__ = [
    "LoraSpec",
    "ManagerActionCodec",
    "ManagerDecodingError",
    "ManagerInput",
    "ManagerSFTConfig",
    "ManagerTrainingExample",
    "RetrievalTrainingExample",
    "RetrieverTrainingConfig",
    "StructuredMemoryManager",
    "TextGenerator",
    "build_lora_config",
    "build_manager_training_args",
    "build_retriever_training_args",
    "load_manager_examples_jsonl",
    "require_official_training_splits",
    "write_examples_jsonl",
]
