from __future__ import annotations

import json

import pytest

from evimem.controller import (
    ActionType,
    CurationAction,
    HeuristicController,
    SequentialCurationEngine,
)
from evimem.training import (
    ActionCodec,
    ActionDecodingError,
    CachedVerifierOracle,
    CertifiedRewardRecord,
    GRPOJobConfig,
    LearnedController,
    OracleActionExample,
    OracleTrajectoryCompiler,
    PolicyPromptRenderer,
    SFTJobConfig,
    VerifierRewardAdapter,
    build_grpo_args,
    build_grpo_dataset,
    build_grpo_trainer,
    build_lora_config,
    build_sft_args,
    build_sft_dataset,
    build_sft_trainer,
    split_examples_by_document,
)
from evimem.training.cli import build_parser

from .test_evimem_controller import _executor, _state


class _Generator:
    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return self.completion


def _example() -> OracleActionExample:
    state = _state()
    action = CurationAction(
        type=ActionType.RETRIEVE_TABLE,
        arguments={"query": "d33 350 pC/N"},
        rationale_code={"target_slot": "value"},
    )
    return OracleActionExample(
        example_id="example-1",
        episode_id="episode-1",
        document_id=state.candidate.doi,
        state=state,
        legal_actions=tuple(sorted(_executor().legal_actions(state), key=lambda item: item.value)),
        target_action=action,
        source="oracle_evidence_path",
    )


def test_action_codec_is_canonical_strict_and_legality_aware() -> None:
    action = _example().target_action
    encoded = ActionCodec.encode(action)
    assert ActionCodec.decode(encoded, legal_actions={ActionType.RETRIEVE_TABLE}) == action
    assert ActionCodec.canonical_key(action) == (
        '{"arguments":{"query":"d33 350 pC/N"},"type":"RETRIEVE_TABLE"}'
    )
    with pytest.raises(ActionDecodingError, match="illegal action"):
        ActionCodec.decode(encoded, legal_actions={ActionType.DEFER_FOR_REVIEW})
    with pytest.raises(ActionDecodingError, match="invalid action JSON"):
        ActionCodec.decode(f"Use this action: {encoded}")


def test_prompt_contains_only_policy_state_and_legal_action_schema() -> None:
    example = _example()
    messages = PolicyPromptRenderer.render_messages(example.state, example.legal_actions)
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {
        "action_schema",
        "legal_actions",
        "policy_visible_state",
        "task",
    }
    serialized = messages[1]["content"]
    assert "gold_evidence_refs" not in serialized
    assert "gold_certificate_id" not in serialized
    assert "publication_store" not in serialized


def test_learned_controller_returns_legal_action_and_fails_closed() -> None:
    example = _example()
    valid = _Generator(ActionCodec.encode(example.target_action))
    assert (
        LearnedController(valid).choose_action(example.state, frozenset(example.legal_actions))
        == example.target_action
    )

    invalid = LearnedController(_Generator("not-json"))
    action = invalid.choose_action(example.state, frozenset(example.legal_actions))
    assert action.type == ActionType.DEFER_FOR_REVIEW
    assert invalid.last_decode_error is not None


def test_sft_and_grpo_records_keep_oracle_targets_out_of_prompts() -> None:
    example = _example()
    sft = build_sft_dataset([example])[0]
    grpo = build_grpo_dataset([example])[0]
    assert ActionCodec.decode(sft["completion"][0]["content"]) == example.target_action
    assert "completion" not in grpo
    assert "target_action" not in json.dumps(grpo, sort_keys=True)


def test_document_split_never_separates_steps_from_one_paper() -> None:
    first = _example()
    second = first.model_copy(update={"example_id": "example-2"})
    splits = split_examples_by_document([first, second], seed=7)
    populated = [name for name, values in splits.items() if values]
    assert len(populated) == 1
    assert {item.example_id for item in splits[populated[0]]} == {"example-1", "example-2"}


def test_trajectory_compiler_replays_state_and_result_hashes() -> None:
    executor = _executor()
    initial = _state()
    outcome = SequentialCurationEngine(executor=executor, max_steps=4).run(
        run_id="episode-training",
        initial_state=initial,
        policy=HeuristicController(),
    )
    examples = OracleTrajectoryCompiler.compile(
        initial_state=initial,
        trajectory=outcome.trajectory,
        executor=executor,
        source="heuristic",
    )
    assert len(examples) == len(outcome.trajectory.steps)
    assert examples[-1].target_action.type == ActionType.REQUEST_PUBLICATION


def test_grpo_reward_comes_from_external_certified_oracle() -> None:
    action = _example().target_action
    example = _example()
    oracle = CachedVerifierOracle(
        [
            CertifiedRewardRecord.create(
                example=example,
                action_rewards={ActionCodec.canonical_key(action): 2.75},
                verifier_version="test-verifier-1",
            )
        ],
        examples=[example],
    )
    reward = VerifierRewardAdapter(oracle)
    assert reward(
        [ActionCodec.encode(action), "invalid"],
        example_id=["example-1", "example-1"],
        legal_actions=[
            [ActionType.RETRIEVE_TABLE.value],
            [ActionType.RETRIEVE_TABLE.value],
        ],
    ) == [2.75, -3.0]


def test_certified_reward_artifact_rejects_tampering_and_illegal_actions() -> None:
    example = _example()
    record = CertifiedRewardRecord.create(
        example=example,
        action_rewards={ActionCodec.canonical_key(example.target_action): 1.0},
        verifier_version="test-verifier-1",
    )
    tampered = record.model_dump(mode="json")
    tampered["action_rewards"] = {ActionCodec.canonical_key(example.target_action): 9.0}
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        CertifiedRewardRecord.model_validate(tampered)

    illegal = CurationAction(
        type=ActionType.INSPECT_CAPTION,
        arguments={"reference_id": "figure-1"},
    )
    illegal_record = CertifiedRewardRecord.create(
        example=example,
        action_rewards={ActionCodec.canonical_key(illegal): 1.0},
        verifier_version="test-verifier-1",
    )
    with pytest.raises(ValueError, match="illegal action"):
        CachedVerifierOracle([illegal_record], examples=[example])


def test_trl_and_peft_configs_match_single_gpu_constrained_training(tmp_path) -> None:
    lora = build_lora_config()
    assert lora.r == 16
    assert lora.task_type.value == "CAUSAL_LM"

    sft = build_sft_args(SFTJobConfig(output_dir=str(tmp_path / "sft"), bf16=False))
    assert sft.completion_only_loss is True
    assert sft.report_to == []

    grpo = build_grpo_args(GRPOJobConfig(output_dir=str(tmp_path / "grpo"), bf16=False))
    assert grpo.num_generations == 4
    assert grpo.scale_rewards == "group"
    assert grpo.loss_type == "grpo"
    assert grpo.remove_unused_columns is False


def test_training_cli_requires_explicit_model_data_and_output(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "grpo",
            "--model",
            "controller-sft",
            "--train-jsonl",
            str(tmp_path / "train.jsonl"),
            "--rewards-jsonl",
            str(tmp_path / "rewards.jsonl"),
            "--output-dir",
            str(tmp_path / "grpo"),
        ]
    )
    assert args.command == "grpo"
    assert args.num_generations == 4
    assert args.output_dir == tmp_path / "grpo"


def test_trainer_factories_delegate_to_trl_and_peft(monkeypatch, tmp_path) -> None:
    import trl

    calls = []

    class _Trainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            calls.append(kwargs)

    monkeypatch.setattr(trl, "SFTTrainer", _Trainer)
    monkeypatch.setattr(trl, "GRPOTrainer", _Trainer)

    sft = build_sft_trainer(
        model="controller-base",
        train_dataset=[],
        config=SFTJobConfig(output_dir=str(tmp_path / "sft"), bf16=False),
    )
    oracle = CachedVerifierOracle(
        [
            CertifiedRewardRecord.create(
                example=_example(),
                action_rewards={ActionCodec.canonical_key(_example().target_action): 1.0},
                verifier_version="test-verifier-1",
            )
        ],
        examples=[_example()],
    )
    grpo = build_grpo_trainer(
        model="controller-sft",
        train_dataset=[],
        reward_oracle=oracle,
        config=GRPOJobConfig(output_dir=str(tmp_path / "grpo"), bf16=False),
    )

    assert sft.kwargs["model"] == "controller-base"
    assert sft.kwargs["peft_config"].r == 16
    assert grpo.kwargs["model"] == "controller-sft"
    assert isinstance(grpo.kwargs["reward_funcs"], VerifierRewardAdapter)
    assert len(calls) == 2
