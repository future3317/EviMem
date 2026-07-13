"""Learned controller adapter with fail-closed structured action decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from evimem.controller import ActionType, CurationAction
from evimem.controller.state import ControllerState

from .codec import ActionCodec, ActionDecodingError, PolicyPromptRenderer


class TextGenerator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


class TransformersTextGenerator:
    """Thin inference wrapper around maintained Transformers APIs."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: GenerationConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        config: GenerationConfig | None = None,
        device_map: str | dict[str, int | str] = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> TransformersTextGenerator:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        return cls(model=model, tokenizer=tokenizer, config=config)

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(self.model.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "top_p": self.config.top_p,
        }
        if self.config.do_sample:
            kwargs["temperature"] = self.config.temperature
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, **kwargs)
        completion = output_ids[0, input_ids.shape[-1] :]
        return str(self.tokenizer.decode(completion, skip_special_tokens=True)).strip()


class LearnedController:
    """ControllerPolicy implementation that can only return legal actions."""

    def __init__(self, generator: TextGenerator) -> None:
        self.generator = generator
        self.last_raw_completion: str | None = None
        self.last_decode_error: str | None = None

    def choose_action(
        self,
        state: ControllerState,
        legal_actions: frozenset[ActionType],
    ) -> CurationAction:
        messages = PolicyPromptRenderer.render_messages(state, legal_actions)
        raw = self.generator.generate(messages)
        self.last_raw_completion = raw
        try:
            action = ActionCodec.decode(raw, legal_actions=legal_actions)
            self.last_decode_error = None
            return action
        except ActionDecodingError as exc:
            self.last_decode_error = str(exc)
            for safe_action in (ActionType.DEFER_FOR_REVIEW, ActionType.STOP_NO_RECORD):
                if safe_action in legal_actions:
                    return CurationAction(
                        type=safe_action,
                        rationale_code={"reason": "invalid_model_action"},
                    )
            raise RuntimeError("model action invalid and no fail-closed action is legal") from exc

