"""Exploratory fixed-trace test of CHIC training-subset selection.

This is deliberately a development experiment, not a confirmatory v5 run.  It
uses only MatPES development systems.  Every strategy sees
the same fixed reveal trace; the selected subset changes only the one-step
update of a small residual MLP.  Evaluation outcomes are consumed by the
evaluator and never by the subset selector before reveal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry

from matmem.chic import joint_nonnegative_gradient_match

STRATEGIES = (
    "full_history",
    "fifo",
    "random",
    "diversity",
    "hard_example",
    "grad_match",
    "chic",
)


@dataclass(frozen=True)
class ExperimentConfig:
    max_systems: int = 8
    minimum_candidates: int = 12
    maximum_budget: int = 6
    capacity: int = 2
    hidden_dimension: int = 32
    pretrain_epochs: int = 120
    pretrain_learning_rate: float = 0.01
    online_learning_rate: float = 0.01
    decision_temperature_ev_per_atom: float = 0.05
    seed: int = 20270720


class ResidualMLP(torch.nn.Module):
    """Source-preserving predictor with a trainable nonlinear correction."""

    def __init__(self, input_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.correction = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dimension, 1),
        )
        torch.nn.init.zeros_(self.correction[-1].weight)
        torch.nn.init.zeros_(self.correction[-1].bias)

    def forward(self, standardized_features: torch.Tensor, source_energy: torch.Tensor) -> torch.Tensor:
        return source_energy + self.correction(standardized_features).squeeze(-1)

    def hidden_and_prediction(
        self,
        standardized_features: torch.Tensor,
        source_energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.correction[1](self.correction[0](standardized_features))
        prediction = source_energy + self.correction[2](hidden).squeeze(-1)
        return hidden, prediction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _feature(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [row["source_formation_energy_ev_per_atom"], *row["source_environment_embedding"]],
        dtype=np.float64,
    )


def _initial_entries(rows: list[dict[str, Any]]) -> list[ComputedEntry]:
    return [
        ComputedEntry(
            row["composition"],
            row["corrected_total_energy_ev"],
            entry_id=row["entry_id"],
        )
        for row in rows
    ]


def _hull_formation_energy(diagram: PhaseDiagram, composition: dict[str, float]) -> float:
    from pymatgen.core import Composition

    parsed = Composition(composition)
    total_per_atom = float(diagram.get_hull_energy_per_atom(parsed))
    fake = ComputedEntry(parsed, total_per_atom * parsed.num_atoms)
    return float(diagram.get_form_energy_per_atom(fake))


def _flatten_gradients(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool = False,
) -> np.ndarray:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=False,
    )
    return np.concatenate([gradient.detach().cpu().numpy().reshape(-1) for gradient in gradients])


def _apply_flat_gradient(
    model: ResidualMLP,
    gradient: np.ndarray,
    *,
    learning_rate: float,
) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            update = torch.as_tensor(
                gradient[offset : offset + size],
                dtype=parameter.dtype,
                device=parameter.device,
            ).reshape_as(parameter)
            parameter.subtract_(learning_rate * update)
            offset += size
    if offset != len(gradient):
        raise AssertionError("flat CHIC gradient does not match model parameters")


def _predict(
    model: ResidualMLP,
    rows: list[dict[str, Any]],
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> torch.Tensor:
    features = np.stack([_feature(row) for row in rows])
    standardized = torch.as_tensor(
        (features - feature_mean) / feature_scale,
        dtype=torch.float64,
    )
    source = torch.as_tensor(
        [row["source_formation_energy_ev_per_atom"] for row in rows],
        dtype=torch.float64,
    )
    return model(standardized, source)


def _hidden_and_prediction(
    model: ResidualMLP,
    rows: list[dict[str, Any]],
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = np.stack([_feature(row) for row in rows])
    standardized = torch.as_tensor(
        (features - feature_mean) / feature_scale,
        dtype=torch.float64,
    )
    source = torch.as_tensor(
        [row["source_formation_energy_ev_per_atom"] for row in rows],
        dtype=torch.float64,
    )
    return model.hidden_and_prediction(standardized, source)


def _last_layer_proxy_gradients(
    model: ResidualMLP,
    archive_rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return analytic last-layer loss gradients with no full backward pass."""

    with torch.no_grad():
        hidden, prediction = _hidden_and_prediction(
            model,
            archive_rows,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )
    target = torch.as_tensor(
        [
            outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"]
            for row in archive_rows
        ],
        dtype=torch.float64,
    )
    residual = prediction - target
    augmented = torch.column_stack(
        (hidden, torch.ones(len(hidden), dtype=hidden.dtype, device=hidden.device))
    )
    gradients = residual[:, None] * augmented
    losses = 0.5 * residual**2
    return gradients.cpu().numpy(), losses.cpu().numpy()


def _sample_gradients(
    model: ResidualMLP,
    archive_rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    parameters = tuple(model.parameters())
    gradients: list[np.ndarray] = []
    losses: list[float] = []
    selected = tuple(range(len(archive_rows))) if indices is None else indices
    for index in selected:
        row = archive_rows[index]
        prediction = _predict(
            model, [row], feature_mean=feature_mean, feature_scale=feature_scale
        )[0]
        target = torch.tensor(
            outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"],
            dtype=torch.float64,
        )
        loss = 0.5 * (prediction - target) ** 2
        gradients.append(_flatten_gradients(loss, parameters))
        losses.append(float(loss.detach()))
    return np.stack(gradients), np.asarray(losses)


def _sample_losses(
    model: ResidualMLP,
    archive_rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    prediction = _predict(
        model,
        archive_rows,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    ).detach().cpu().numpy()
    target = np.asarray(
        [
            outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"]
            for row in archive_rows
        ]
    )
    return 0.5 * (prediction - target) ** 2


def _decision_gradient(
    model: ResidualMLP,
    remaining_rows: list[dict[str, Any]],
    exact_entries: list[ComputedEntry],
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, int, float]:
    with torch.no_grad():
        hidden, predicted = _hidden_and_prediction(
            model,
            remaining_rows,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )
    prediction_gradients = torch.column_stack(
        (hidden, torch.ones(len(hidden), dtype=hidden.dtype, device=hidden.device))
    ).cpu().numpy()
    diagram = PhaseDiagram(exact_entries)
    predicted_e = predicted.detach().cpu().numpy()
    margins = np.empty(len(remaining_rows), dtype=np.float64)
    for candidate_index, row in enumerate(remaining_rows):
        # The causal competing set contains only exact revealed/reference
        # phases.  Therefore d margin / d candidate prediction is exactly one;
        # solving a fresh LP for every candidate would be redundant.
        hull_energy = _hull_formation_energy(diagram, row["composition"])
        margins[candidate_index] = predicted_e[candidate_index] - hull_energy
    logits = -(margins - float(np.min(margins))) / temperature
    weights = np.exp(logits)
    weights /= weights.sum()
    target = weights @ prediction_gradients
    effective_candidates = float(1.0 / np.sum(weights**2))
    return target, len(remaining_rows), effective_candidates


def _diversity_indices(features: np.ndarray, capacity: int) -> tuple[int, ...]:
    if capacity >= len(features):
        return tuple(range(len(features)))
    center = features.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(features - center, axis=1)))
    selected = [first]
    while len(selected) < capacity:
        distances = np.min(
            np.linalg.norm(features[:, None, :] - features[selected][None, :, :], axis=2),
            axis=1,
        )
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    return tuple(selected)


def _selected_update(
    *,
    strategy: str,
    model: ResidualMLP,
    archive_rows: list[dict[str, Any]],
    remaining_rows: list[dict[str, Any]],
    exact_entries: list[ComputedEntry],
    outcomes: dict[str, dict[str, Any]],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    config: ExperimentConfig,
    system: str,
    round_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    capacity = min(config.capacity, len(archive_rows))
    metadata: dict[str, Any] = {
        "sample_forward_evaluations": 0,
        "sample_gradient_evaluations": 0,
        "pool_prediction_gradient_evaluations": 0,
        "proxy_gradient_evaluations": 0,
        "selector_fallback": None,
    }
    if strategy == "full_history":
        indices = tuple(range(len(archive_rows)))
    elif strategy == "fifo":
        indices = tuple(range(len(archive_rows) - capacity, len(archive_rows)))
    elif strategy == "random":
        indices = tuple(
            sorted(
                range(len(archive_rows)),
                key=lambda index: _stable_hash(
                    str(config.seed), system, str(round_index), archive_rows[index]["pair_id"]
                ),
            )[:capacity]
        )
    elif strategy == "diversity":
        archive_features = np.stack([(_feature(row) - feature_mean) / feature_scale for row in archive_rows])
        indices = _diversity_indices(archive_features, capacity)
    elif strategy == "hard_example":
        sample_losses = _sample_losses(
            model,
            archive_rows,
            outcomes,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )
        metadata["sample_forward_evaluations"] = len(archive_rows)
        indices = tuple(
            sorted(
                range(len(archive_rows)),
                key=lambda index: (-sample_losses[index], archive_rows[index]["pair_id"]),
            )[:capacity]
        )
    else:
        proxy_gradients, _ = _last_layer_proxy_gradients(
            model,
            archive_rows,
            outcomes,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )
        metadata["sample_forward_evaluations"] = len(archive_rows)
        metadata["proxy_gradient_evaluations"] = len(archive_rows)
        full_proxy_gradient = proxy_gradients.mean(axis=0)
        target = full_proxy_gradient
        matching_gradients = proxy_gradients
        if strategy == "chic" and remaining_rows:
            decision_gradient, feasible, effective_candidates = _decision_gradient(
                model,
                remaining_rows,
                exact_entries,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                temperature=config.decision_temperature_ev_per_atom,
            )
            metadata["sample_forward_evaluations"] += len(remaining_rows)
            metadata["proxy_gradient_evaluations"] += len(remaining_rows)
            metadata["feasible_hull_margin_count"] = feasible
            decision_norm = float(np.linalg.norm(decision_gradient))
            metadata["decision_gradient_norm"] = decision_norm
            metadata["softmin_effective_candidate_count"] = effective_candidates
            if decision_norm > 1e-12:
                direction = decision_gradient / decision_norm
                # This realizes ||e||^2 + (<d_hat,e>)^2 exactly.  The decision
                # term is normalized, so no dataset-tuned mixing weight enters.
                stretch = math.sqrt(2.0) - 1.0

                def transform(values: np.ndarray) -> np.ndarray:
                    return values + stretch * np.outer(values @ direction, direction)

                target = transform(full_proxy_gradient[None, :])[0]
                matching_gradients = transform(proxy_gradients)
                metadata["matching_metric"] = (
                    "euclidean_plus_unit_decision_projection"
                )
            else:
                metadata["selector_fallback"] = "no_hull_decision_gradient"
        matched = joint_nonnegative_gradient_match(
            target,
            matching_gradients,
            max_items=capacity,
        )
        indices = matched.selected_indices
        selection_weights = np.asarray(matched.weights)
        if not indices:
            metadata["selector_fallback"] = metadata["selector_fallback"] or "nnls_empty"
            indices = (len(archive_rows) - 1,)
            selection_weights = np.ones(1, dtype=np.float64)
        metadata["gradient_match_initial_norm"] = matched.initial_error_norm
        metadata["gradient_match_residual_norm"] = matched.residual_norm
    selected_gradients, _ = _sample_gradients(
        model,
        archive_rows,
        outcomes,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        indices=indices,
    )
    metadata["sample_gradient_evaluations"] = len(indices)
    if strategy in {"grad_match", "chic"}:
        update = selection_weights @ selected_gradients
    else:
        update = selected_gradients.mean(axis=0)
    metadata["selected_indices"] = indices
    metadata["selected_pair_ids"] = [archive_rows[index]["pair_id"] for index in indices]
    metadata["selected_size"] = len(indices)
    metadata["update_gradient_norm"] = float(np.linalg.norm(update))
    return np.asarray(update, dtype=np.float64), metadata


def _pretrain(
    rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    config: ExperimentConfig,
) -> ResidualMLP:
    torch.manual_seed(config.seed)
    model = ResidualMLP(len(feature_mean), config.hidden_dimension).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain_learning_rate)
    counts = defaultdict(int)
    for row in rows:
        counts[row["chemical_system"]] += 1
    weights = torch.as_tensor(
        [1.0 / counts[row["chemical_system"]] for row in rows], dtype=torch.float64
    )
    weights /= weights.sum()
    targets = torch.as_tensor(
        [outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"] for row in rows],
        dtype=torch.float64,
    )
    for _ in range(config.pretrain_epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(
            model, rows, feature_mean=feature_mean, feature_scale=feature_scale
        )
        loss = torch.sum(weights * (prediction - targets) ** 2)
        loss.backward()
        optimizer.step()
    return model


def _evaluate_round(
    model: ResidualMLP,
    remaining_rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    diagram: PhaseDiagram,
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> dict[str, float]:
    predicted = _predict(
        model,
        remaining_rows,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    ).detach().cpu().numpy()
    targets = np.asarray(
        [outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"] for row in remaining_rows]
    )
    predicted_margins = []
    actual_margins = []
    for row, point in zip(remaining_rows, predicted, strict=True):
        hull_energy = _hull_formation_energy(diagram, row["composition"])
        predicted_margins.append(float(point - hull_energy))
        outcome = outcomes[row["pair_id"]]
        actual_entry = ComputedEntry(
            outcome["composition"],
            outcome["target_corrected_total_energy_ev"],
            entry_id=row["pair_id"],
        )
        actual_margins.append(
            float(diagram.get_e_above_hull(actual_entry, allow_negative=True))
        )
    predicted_margins = np.asarray(predicted_margins)
    actual_margins = np.asarray(actual_margins)
    selected = int(np.argmin(predicted_margins))
    return {
        "mae_ev_per_atom": float(np.mean(np.abs(predicted - targets))),
        "hull_misclassification": float(
            np.mean((predicted_margins <= 0) != (actual_margins <= 1e-8))
        ),
        "action_regret_ev_per_atom": float(
            actual_margins[selected] - np.min(actual_margins)
        ),
    }


def run(
    *,
    task_path: Path,
    development_vault_path: Path,
    output_path: Path,
    config: ExperimentConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError("CHIC exploratory output already exists")
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("CHIC exploratory output must remain outside Git")
    torch.set_num_threads(1)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    vault = json.loads(development_vault_path.read_text(encoding="utf-8"))
    if any(row["split"] != "development" for row in vault["target_outcomes"]):
        raise ValueError("exploratory CHIC runner accepts only development outcomes")
    outcomes = {row["pair_id"]: row for row in vault["target_outcomes"]}
    rows = task["development_pairs"]
    if set(outcomes) != {row["pair_id"] for row in rows}:
        raise ValueError("development task/vault join is not exact")
    release_id = task["release_id"]
    ordered_systems = sorted(
        {row["chemical_system"] for row in rows},
        key=lambda system: _stable_hash(release_id, "chic-fit-split-v1", system),
    )
    fit_count = max(1, len(ordered_systems) // 3)
    fit_systems = set(ordered_systems[:fit_count])
    trajectory_systems = sorted(
        (
            system for system in ordered_systems[fit_count:]
            if system not in fit_systems
            and sum(item["chemical_system"] == system for item in rows)
            >= config.minimum_candidates
        ),
        key=lambda system: _stable_hash(release_id, "chic-exploratory-v1", system),
    )[: config.max_systems]
    if set(trajectory_systems) & fit_systems:
        raise AssertionError("CHIC fit and development systems overlap")
    fit_rows = [row for row in rows if row["chemical_system"] in fit_systems]
    feature_matrix = np.stack([_feature(row) for row in fit_rows])
    feature_mean = feature_matrix.mean(axis=0)
    feature_scale = feature_matrix.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    started = time.perf_counter()
    base_model = _pretrain(
        fit_rows,
        outcomes,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        config=config,
    )
    pretrain_seconds = time.perf_counter() - started
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[row["chemical_system"]].append(row)
    system_results: dict[str, Any] = {}
    for system in trajectory_systems:
        ordered = sorted(
            by_system[system],
            key=lambda row: _stable_hash(
                release_id, "matpes-fixed-reveal-v1", row["pair_id"]
            ),
        )
        budget = min(config.maximum_budget, len(ordered) // 2)
        initial_rows = task["development_initial_phase_entries"][system]
        strategy_results: dict[str, Any] = {}
        for strategy in STRATEGIES:
            model = deepcopy(base_model)
            exact_entries = _initial_entries(initial_rows)
            archive_rows: list[dict[str, Any]] = []
            rounds: list[dict[str, Any]] = []
            for round_index in range(budget):
                remaining_rows = ordered[round_index:]
                diagram = PhaseDiagram(exact_entries)
                metrics = _evaluate_round(
                    model,
                    remaining_rows,
                    outcomes,
                    diagram,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                )
                revealed = ordered[round_index]
                archive_rows.append(revealed)
                outcome = outcomes[revealed["pair_id"]]
                exact_entries.append(
                    ComputedEntry(
                        outcome["composition"],
                        outcome["target_corrected_total_energy_ev"],
                        entry_id=revealed["pair_id"],
                    )
                )
                selection_started = time.perf_counter()
                update, selection = _selected_update(
                    strategy=strategy,
                    model=model,
                    archive_rows=archive_rows,
                    remaining_rows=ordered[round_index + 1 :],
                    exact_entries=exact_entries,
                    outcomes=outcomes,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    config=config,
                    system=system,
                    round_index=round_index + 1,
                )
                selection_seconds = time.perf_counter() - selection_started
                update_started = time.perf_counter()
                _apply_flat_gradient(
                    model, update, learning_rate=config.online_learning_rate
                )
                update_seconds = time.perf_counter() - update_started
                rounds.append(
                    {
                        "round_index": round_index + 1,
                        "revealed_pair_id": revealed["pair_id"],
                        **metrics,
                        "archive_size": len(archive_rows),
                        "selection_seconds": selection_seconds,
                        "update_seconds": update_seconds,
                        **selection,
                    }
                )
            strategy_results[strategy] = {
                "rounds": rounds,
                "mean_mae_ev_per_atom": float(
                    np.mean([row["mae_ev_per_atom"] for row in rounds])
                ),
                "mean_hull_misclassification": float(
                    np.mean([row["hull_misclassification"] for row in rounds])
                ),
                "mean_action_regret_ev_per_atom": float(
                    np.mean([row["action_regret_ev_per_atom"] for row in rounds])
                ),
                "total_selection_seconds": float(
                    sum(row["selection_seconds"] for row in rounds)
                ),
                "total_update_seconds": float(
                    sum(row["update_seconds"] for row in rounds)
                ),
                "total_gradient_evaluations": int(
                    sum(
                        row["sample_gradient_evaluations"]
                        for row in rounds
                    )
                ),
                "total_proxy_gradient_evaluations": int(
                    sum(row["proxy_gradient_evaluations"] for row in rounds)
                ),
            }
        system_results[system] = {
            "candidate_count": len(ordered),
            "budget": budget,
            "fixed_reveal_pair_ids": [row["pair_id"] for row in ordered[:budget]],
            "strategies": strategy_results,
        }

    aggregates: dict[str, Any] = {}
    for strategy in STRATEGIES:
        aggregates[strategy] = {
            metric: float(
                np.mean(
                    [
                        system_results[system]["strategies"][strategy][metric]
                        for system in trajectory_systems
                    ]
                )
            )
            for metric in (
                "mean_mae_ev_per_atom",
                "mean_hull_misclassification",
                "mean_action_regret_ev_per_atom",
                "total_selection_seconds",
                "total_update_seconds",
                "total_gradient_evaluations",
                "total_proxy_gradient_evaluations",
            )
        }
    output = {
        "schema_version": 1,
        "status": "exploratory_development_systems_only_not_confirmatory",
        "estimand": "fixed_reveal_training_subset_effect",
        "task_sha256": _sha256(task_path),
        "development_vault_sha256": _sha256(development_vault_path),
        "script_sha256": _sha256(Path(__file__)),
        "evaluation_systems_accessed": False,
        "config": asdict(config),
        "fit_system_count": len(fit_systems),
        "fit_pair_count": len(fit_rows),
        "development_systems": trajectory_systems,
        "pretrain_seconds": pretrain_seconds,
        "aggregates": aggregates,
        "systems": system_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"output={output_path.resolve()}")
    for strategy in STRATEGIES:
        values = aggregates[strategy]
        print(
            strategy,
            f"mae={values['mean_mae_ev_per_atom']:.6f}",
            f"hull={values['mean_hull_misclassification']:.6f}",
            f"regret={values['mean_action_regret_ev_per_atom']:.6f}",
            f"grad_evals={values['total_gradient_evaluations']:.1f}",
            f"proxy_evals={values['total_proxy_gradient_evaluations']:.1f}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--development-vault", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-systems", type=int, default=8)
    parser.add_argument("--minimum-candidates", type=int, default=12)
    parser.add_argument("--maximum-budget", type=int, default=6)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--hidden-dimension", type=int, default=32)
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--pretrain-learning-rate", type=float, default=0.01)
    parser.add_argument("--online-learning-rate", type=float, default=0.01)
    parser.add_argument("--decision-temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20270720)
    args = parser.parse_args()
    config = ExperimentConfig(
        max_systems=args.max_systems,
        minimum_candidates=args.minimum_candidates,
        maximum_budget=args.maximum_budget,
        capacity=args.capacity,
        hidden_dimension=args.hidden_dimension,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_learning_rate=args.pretrain_learning_rate,
        online_learning_rate=args.online_learning_rate,
        decision_temperature_ev_per_atom=args.decision_temperature,
        seed=args.seed,
    )
    if (
        config.max_systems < 1
        or config.minimum_candidates < 2
        or config.maximum_budget < 1
        or config.capacity < 1
        or config.hidden_dimension < 1
        or config.pretrain_epochs < 1
        or config.pretrain_learning_rate <= 0
        or config.online_learning_rate <= 0
        or config.decision_temperature_ev_per_atom <= 0
    ):
        raise ValueError("CHIC exploratory configuration is invalid")
    run(
        task_path=args.task,
        development_vault_path=args.development_vault,
        output_path=args.output,
        config=config,
    )


if __name__ == "__main__":
    main()
