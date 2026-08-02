"""Exact, observable joint-acquisition--retention separation audit.

The construction is intentionally small enough to enumerate exactly.  It is a
synthetic delayed-full-pool POMDP, not an adapter for a materials dataset.  In
particular, it keeps the immutable archive conceptually complete while making
the policy-facing post-reveal state the only channel through which a later
acquisition decision may use a calibration witness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations
from typing import Literal

Action = str
MemoryToken = int | None
PolicyName = Literal["joint", "nonadaptive", "myopic", "full_history"]


@dataclass(frozen=True)
class ExactSeparationInstance:
    """Registered finite-pool instance for the separation audit.

    ``state_access_cost == 0`` is the homogeneous full-history null: the
    policy can read the archive without consuming a bounded decision state.
    ``probe_protocol_supported`` fails closed by removing the witness from the
    legal action set.  An uninformative but supported witness remains legal,
    yet cannot improve a future boundary decision.
    """

    world_count: int = 3
    budget: int = 3
    memory_capacity: int = 1
    state_access_cost: float = 1.0
    probe_informative: bool = True
    probe_protocol_supported: bool = True

    def __post_init__(self) -> None:
        if self.world_count != 3:
            raise ValueError("The registered proof instance has exactly three worlds.")
        if self.budget != 3:
            raise ValueError("The registered proof instance has query budget B=3.")
        if self.memory_capacity < 0:
            raise ValueError("memory_capacity must be non-negative.")
        if self.state_access_cost < 0:
            raise ValueError("state_access_cost must be non-negative.")

    @property
    def actions(self) -> tuple[Action, ...]:
        return (
            "probe",
            "safe",
            "group-0-a",
            "group-0-b",
            "group-1-a",
            "group-1-b",
            "group-2-a",
            "group-2-b",
        )

    @property
    def legal_actions(self) -> tuple[Action, ...]:
        if self.probe_protocol_supported:
            return self.actions
        return tuple(action for action in self.actions if action != "probe")

    @property
    def effective_memory_capacity(self) -> int:
        """Return the policy-facing capacity after the access-cost null."""

        if self.state_access_cost == 0:
            return self.budget
        return self.memory_capacity


@dataclass(frozen=True)
class ExactPolicyPath:
    """One complete-world trace under a deterministic policy."""

    world: int
    actions: tuple[Action, ...]
    retained_states: tuple[MemoryToken, ...]
    terminal_confirmations: float
    final_causal_confirmations: float
    false_stable: float
    false_unstable: float


@dataclass(frozen=True)
class ExactPolicyEvaluation:
    """Exact world-average metrics and traces for one declared policy class."""

    policy: PolicyName
    terminal_confirmations: float
    final_causal_confirmations: float
    false_stable: float
    false_unstable: float
    memory_turnovers: float
    paths: tuple[ExactPolicyPath, ...]


@dataclass(frozen=True)
class ExactSeparationReport:
    """Primary comparison and registered null/safety evaluations."""

    joint: ExactPolicyEvaluation
    nonadaptive: ExactPolicyEvaluation
    myopic: ExactPolicyEvaluation
    full_history: ExactPolicyEvaluation
    zero_memory: ExactPolicyEvaluation
    zero_access_cost: ExactPolicyEvaluation
    uninformative_witness: ExactPolicyEvaluation
    unsupported_witness: ExactPolicyEvaluation

    @property
    def joint_nonadaptive_gap(self) -> float:
        return self.joint.terminal_confirmations - self.nonadaptive.terminal_confirmations


def _group_world(action: Action) -> int | None:
    if not action.startswith("group-"):
        return None
    return int(action.split("-")[1])


def _terminal_confirmation(action: Action, world: int) -> float:
    if action == "safe":
        return 1.0
    return float(_group_world(action) == world)


def _final_causal_confirmation(action: Action) -> float:
    # ``probe`` is a calibration witness.  Every selected candidate is stable
    # on the selected-history hull before an omitted world-specific competitor
    # performs the delayed full-pool adjudication.
    return float(action != "probe")


def _belief_from_memory(instance: ExactSeparationInstance, memory: MemoryToken) -> tuple[int, ...]:
    if memory is not None and instance.probe_informative:
        return (memory,)
    return tuple(range(instance.world_count))


def _mean_terminal(action: Action, belief: tuple[int, ...]) -> float:
    return sum(_terminal_confirmation(action, world) for world in belief) / len(belief)


def _retention_choices(
    instance: ExactSeparationInstance,
    previous: MemoryToken,
    action: Action,
    observed_world: int | None,
) -> tuple[MemoryToken, ...]:
    if instance.effective_memory_capacity == 0:
        return (None,)
    choices = {previous}
    if action == "probe" and instance.probe_informative and observed_world is not None:
        choices.add(observed_world)
    return tuple(sorted(choices, key=lambda value: -1 if value is None else value))


def _joint_value(
    instance: ExactSeparationInstance,
    remaining: tuple[Action, ...],
    rounds_left: int,
    memory: MemoryToken,
    cache: dict[tuple[tuple[Action, ...], int, MemoryToken], float],
) -> float:
    key = (remaining, rounds_left, memory)
    if key in cache:
        return cache[key]
    if rounds_left == 0:
        return 0.0

    belief = _belief_from_memory(instance, memory)
    best_value = float("-inf")
    for action in remaining:
        next_remaining = tuple(candidate for candidate in remaining if candidate != action)
        if action == "probe" and instance.probe_informative:
            branch_values = []
            for world in belief:
                continuation = max(
                    _joint_value(instance, next_remaining, rounds_left - 1, retained, cache)
                    for retained in _retention_choices(instance, memory, action, world)
                )
                branch_values.append(continuation)
            value = sum(branch_values) / len(branch_values)
        else:
            continuation = max(
                _joint_value(instance, next_remaining, rounds_left - 1, retained, cache)
                for retained in _retention_choices(instance, memory, action, None)
            )
            value = _mean_terminal(action, belief) + continuation
        best_value = max(best_value, value)
    cache[key] = best_value
    return best_value


def _joint_action_and_retention(
    instance: ExactSeparationInstance,
    remaining: tuple[Action, ...],
    rounds_left: int,
    memory: MemoryToken,
    world: int,
    cache: dict[tuple[tuple[Action, ...], int, MemoryToken], float],
) -> tuple[Action, MemoryToken]:
    """Recover a deterministic lexicographic optimal action for one world."""

    belief = _belief_from_memory(instance, memory)
    candidates: list[tuple[float, Action, MemoryToken]] = []
    for action in remaining:
        next_remaining = tuple(candidate for candidate in remaining if candidate != action)
        observed_world = world if action == "probe" and instance.probe_informative else None
        for retained in _retention_choices(instance, memory, action, observed_world):
            if action == "probe" and instance.probe_informative:
                branch_values = []
                for branch_world in belief:
                    branch_choices = _retention_choices(instance, memory, action, branch_world)
                    branch_values.append(
                        max(
                            _joint_value(instance, next_remaining, rounds_left - 1, choice, cache)
                            for choice in branch_choices
                        )
                    )
                value = sum(branch_values) / len(branch_values)
                # The actual retention must be optimal on this observed branch.
                value = _joint_value(instance, next_remaining, rounds_left - 1, retained, cache)
            else:
                value = _mean_terminal(action, belief) + _joint_value(
                    instance, next_remaining, rounds_left - 1, retained, cache
                )
            candidates.append((value, action, retained))

    best_value = _joint_value(instance, remaining, rounds_left, memory, cache)
    eligible = [candidate for candidate in candidates if abs(candidate[0] - best_value) < 1e-12]
    if not eligible:
        # For an informative probe the full expected action value is compared
        # above, whereas each world takes its own optimal retention branch.
        action = "probe"
        if action in remaining and instance.probe_informative:
            next_remaining = tuple(candidate for candidate in remaining if candidate != action)
            retained = max(
                _retention_choices(instance, memory, action, world),
                key=lambda choice: _joint_value(
                    instance, next_remaining, rounds_left - 1, choice, cache
                ),
            )
            return action, retained
        raise RuntimeError("No exact joint action attained the Bellman value.")
    return sorted(eligible, key=lambda item: (item[1], -1 if item[2] is None else item[2]))[0][1:]


def _path_from_actions(
    policy: PolicyName,
    world: int,
    actions: tuple[Action, ...],
    retained_states: tuple[MemoryToken, ...],
) -> ExactPolicyPath:
    terminal = sum(_terminal_confirmation(action, world) for action in actions)
    final_causal = sum(_final_causal_confirmation(action) for action in actions)
    return ExactPolicyPath(
        world=world,
        actions=actions,
        retained_states=retained_states,
        terminal_confirmations=terminal,
        final_causal_confirmations=final_causal,
        false_stable=final_causal - terminal,
        false_unstable=0.0,
    )


def _summarize_paths(
    policy: PolicyName, paths: tuple[ExactPolicyPath, ...]
) -> ExactPolicyEvaluation:
    count = len(paths)
    turnovers = []
    for path in paths:
        previous: MemoryToken = None
        changes = 0
        for state in path.retained_states:
            if state != previous:
                changes += 1
            previous = state
        turnovers.append(changes)
    return ExactPolicyEvaluation(
        policy=policy,
        terminal_confirmations=sum(path.terminal_confirmations for path in paths) / count,
        final_causal_confirmations=sum(path.final_causal_confirmations for path in paths) / count,
        false_stable=sum(path.false_stable for path in paths) / count,
        false_unstable=sum(path.false_unstable for path in paths) / count,
        memory_turnovers=sum(turnovers) / count,
        paths=paths,
    )


def evaluate_joint_policy(instance: ExactSeparationInstance) -> ExactPolicyEvaluation:
    """Solve the registered bounded-state Bellman recursion exactly."""

    cache: dict[tuple[tuple[Action, ...], int, MemoryToken], float] = {}
    paths = []
    for world in range(instance.world_count):
        remaining = instance.legal_actions
        memory: MemoryToken = None
        actions = []
        retained_states = []
        for rounds_left in range(instance.budget, 0, -1):
            action, memory = _joint_action_and_retention(
                instance, remaining, rounds_left, memory, world, cache
            )
            actions.append(action)
            retained_states.append(memory)
            remaining = tuple(candidate for candidate in remaining if candidate != action)
        paths.append(_path_from_actions("joint", world, tuple(actions), tuple(retained_states)))
    return _summarize_paths("joint", tuple(paths))


def enumerate_nonadaptive_sequence_values(
    instance: ExactSeparationInstance,
) -> tuple[tuple[tuple[Action, ...], float], ...]:
    """Enumerate every state-blind, deterministic acquisition sequence.

    The declared comparator may optimize the sequence globally, but it cannot
    condition a later acquisition on a revealed witness or retained state.  On
    this registered instance its only observable decision inputs are round and
    remaining actions, so every deterministic policy is one ordered sequence.
    """

    values = []
    for actions in permutations(instance.legal_actions, instance.budget):
        terminal = (
            sum(
                sum(_terminal_confirmation(action, world) for action in actions)
                for world in range(instance.world_count)
            )
            / instance.world_count
        )
        values.append((actions, terminal))
    return tuple(values)


def evaluate_nonadaptive_policy(instance: ExactSeparationInstance) -> ExactPolicyEvaluation:
    """Optimize over the declared nonadaptive, state-blind acquisition class."""

    best_actions: tuple[Action, ...] | None = None
    best_terminal = float("-inf")
    for actions, terminal in enumerate_nonadaptive_sequence_values(instance):
        if terminal > best_terminal + 1e-12 or (
            abs(terminal - best_terminal) < 1e-12
            and (best_actions is None or actions < best_actions)
        ):
            best_actions = actions
            best_terminal = terminal
    assert best_actions is not None
    paths = tuple(
        _path_from_actions("nonadaptive", world, best_actions, (None,) * instance.budget)
        for world in range(instance.world_count)
    )
    return _summarize_paths("nonadaptive", paths)


def evaluate_myopic_policy(instance: ExactSeparationInstance) -> ExactPolicyEvaluation:
    """Apply a terminal-myopic policy with no lookahead incentive for the witness."""

    paths = []
    for world in range(instance.world_count):
        remaining = instance.legal_actions
        memory: MemoryToken = None
        actions = []
        retained_states = []
        for _ in range(instance.budget):
            belief = _belief_from_memory(instance, memory)
            action = sorted(
                remaining,
                key=lambda candidate: (-_mean_terminal(candidate, belief), candidate),
            )[0]
            # Myopic acquisition never selects the zero-reward probe here, so
            # it cannot acquire the world code that would make retention useful.
            memory = None
            actions.append(action)
            retained_states.append(memory)
            remaining = tuple(candidate for candidate in remaining if candidate != action)
        paths.append(_path_from_actions("myopic", world, tuple(actions), tuple(retained_states)))
    return _summarize_paths("myopic", tuple(paths))


def evaluate_exact_joint_separation(
    instance: ExactSeparationInstance = ExactSeparationInstance(),
) -> ExactSeparationReport:
    """Evaluate the primary comparison plus all registered null/safety cases."""

    joint = evaluate_joint_policy(instance)
    nonadaptive = evaluate_nonadaptive_policy(instance)
    myopic = evaluate_myopic_policy(instance)
    full_history_instance = replace(
        instance, memory_capacity=instance.budget, state_access_cost=1.0
    )
    full_history = replace(evaluate_joint_policy(full_history_instance), policy="full_history")
    zero_memory = evaluate_joint_policy(replace(instance, memory_capacity=0))
    zero_access_cost = evaluate_joint_policy(
        replace(instance, memory_capacity=0, state_access_cost=0.0)
    )
    uninformative_witness = evaluate_joint_policy(replace(instance, probe_informative=False))
    unsupported_witness = evaluate_joint_policy(replace(instance, probe_protocol_supported=False))
    return ExactSeparationReport(
        joint=joint,
        nonadaptive=nonadaptive,
        myopic=myopic,
        full_history=full_history,
        zero_memory=zero_memory,
        zero_access_cost=zero_access_cost,
        uninformative_witness=uninformative_witness,
        unsupported_witness=unsupported_witness,
    )
