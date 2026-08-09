"""Summarize E52 pre-reveal final-hull membership probability calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal_system_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[str(row["system"])].append(row)
    system_count = len(by_system)
    return np.asarray(
        [1.0 / (system_count * len(by_system[str(row["system"])])) for row in rows],
        dtype=float,
    )


def _normalized_weights(weights: np.ndarray | None, *, size: int) -> np.ndarray | None:
    if weights is None:
        return None
    normalized = np.asarray(weights, dtype=float)
    if normalized.shape != (size,):
        raise ValueError("weights must provide one value per row")
    if np.any(normalized < 0.0) or not np.any(normalized > 0.0):
        raise ValueError("weights must include a positive value and no negative values")
    return normalized / np.sum(normalized)


def _safe_ranking_metric(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    ap: bool,
    weights: np.ndarray | None = None,
) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    function = average_precision_score if ap else roc_auc_score
    return float(function(labels, probabilities, sample_weight=weights))


def _metrics(
    rows: list[dict[str, Any]],
    *,
    bin_count: int,
    weights: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    probabilities = np.asarray([row["probability"] for row in rows], dtype=float)
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    normalized_weights = _normalized_weights(weights, size=len(rows))
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    indices = np.minimum((probabilities * bin_count).astype(int), bin_count - 1)
    ece = 0.0
    for index in range(bin_count):
        mask = indices == index
        if np.any(mask):
            if normalized_weights is None:
                bin_weights = None
                bin_mass = float(np.mean(mask))
            else:
                bin_weights = normalized_weights[mask]
                bin_mass = float(np.sum(bin_weights))
            ece += bin_mass * abs(
                float(np.average(probabilities[mask], weights=bin_weights))
                - float(np.average(labels[mask], weights=bin_weights))
            )
    return {
        "record_count": len(rows),
        "positive_count": int(np.sum(labels)),
        "brier_score": float(np.average((probabilities - labels) ** 2, weights=normalized_weights)),
        "bernoulli_nll": float(
            -np.average(
                labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped),
                weights=normalized_weights,
            )
        ),
        "expected_calibration_error": ece,
        "roc_auc": _safe_ranking_metric(
            labels, probabilities, ap=False, weights=normalized_weights
        ),
        "average_precision": _safe_ranking_metric(
            labels, probabilities, ap=True, weights=normalized_weights
        ),
    }


def _reliability(
    rows: list[dict[str, Any]],
    *,
    bin_count: int,
    weights: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    normalized_weights = _normalized_weights(weights, size=len(rows))
    result: list[dict[str, Any]] = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        mask = np.asarray(
            [
                lower <= row["probability"] < upper
                or (index == bin_count - 1 and row["probability"] == 1.0)
                for row in rows
            ],
            dtype=bool,
        )
        in_bin = [row for row, include in zip(rows, mask, strict=True) if include]
        bin_weights = None if normalized_weights is None else normalized_weights[mask]
        by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in in_bin:
            by_system[row["system"]].append(row)
        system_probability = [
            float(np.mean([row["probability"] for row in system_rows]))
            for system_rows in by_system.values()
        ]
        system_frequency = [
            float(np.mean([row["label"] for row in system_rows]))
            for system_rows in by_system.values()
        ]
        result.append(
            {
                "lower": lower,
                "upper": upper,
                "record_count": len(in_bin),
                "system_count": len(by_system),
                "mean_predicted_probability": (
                    None
                    if not in_bin
                    else float(
                        np.average([row["probability"] for row in in_bin], weights=bin_weights)
                    )
                ),
                "empirical_frequency": (
                    None
                    if not in_bin
                    else float(np.average([row["label"] for row in in_bin], weights=bin_weights))
                ),
                "equal_system_mean_predicted_probability": (
                    None if not system_probability else float(np.mean(system_probability))
                ),
                "equal_system_empirical_frequency": (
                    None if not system_frequency else float(np.mean(system_frequency))
                ),
            }
        )
    return result


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    bin_count: int,
    bootstrap_count: int,
    seed: int,
    equal_system: bool = False,
) -> dict[str, dict[str, float] | None]:
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)
    systems = sorted(by_system)
    if bootstrap_count < 1 or len(systems) < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap_count):
        sampled = rng.choice(systems, size=len(systems), replace=True)
        sampled_rows = [row for system in sampled for row in by_system[str(system)]]
        sampled_weights = None
        if equal_system:
            sampled_weights = np.concatenate(
                [
                    np.full(len(by_system[str(system)]), 1.0 / (len(systems) * len(by_system[str(system)])))
                    for system in sampled
                ]
            )
        metrics = _metrics(sampled_rows, bin_count=bin_count, weights=sampled_weights)
        for name, value in metrics.items():
            if name.endswith("count") or value is None:
                continue
            draws[name].append(float(value))
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for name, values in draws.items()
        if values
    }


def _decision_top_k(rows: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    if k < 1:
        raise ValueError("decision top-k must include the selected action")
    by_state: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[(str(row["system"]), str(row["state_checksum"]))].append(row)
    selected_rows: list[dict[str, Any]] = []
    for state_rows in by_state.values():
        selected = [row for row in state_rows if row["selected"]]
        if len(selected) != 1:
            raise ValueError("each decision state must have exactly one selected row")
        unselected = sorted(
            (row for row in state_rows if not row["selected"]),
            key=lambda row: (-float(row["probability"]), str(row["pair_id"])),
        )
        selected_rows.extend(selected + unselected[: k - 1])
    return selected_rows


def _population_summary(
    rows: list[dict[str, Any]],
    *,
    bin_count: int,
    bootstrap_count: int,
    seed: int,
) -> dict[str, Any]:
    equal_system_weights = _equal_system_weights(rows)
    return {
        "metrics": _metrics(rows, bin_count=bin_count),
        "cluster_bootstrap_95": _cluster_bootstrap(
            rows,
            bin_count=bin_count,
            bootstrap_count=bootstrap_count,
            seed=seed,
        ),
        "reliability_bins": _reliability(rows, bin_count=bin_count),
        "equal_system_metrics": _metrics(
            rows, bin_count=bin_count, weights=equal_system_weights
        ),
        "equal_system_cluster_bootstrap_95": _cluster_bootstrap(
            rows,
            bin_count=bin_count,
            bootstrap_count=bootstrap_count,
            seed=seed,
            equal_system=True,
        ),
        "equal_system_reliability_bins": _reliability(
            rows, bin_count=bin_count, weights=equal_system_weights
        ),
    }


def _extract(input_paths: tuple[Path, ...]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_states: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for path in input_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        input_hashes[str(path.resolve())] = _sha256(path)
        task_sha = str(payload["task_sha256"])
        for system, system_result in payload["systems"].items():
            for policy, strategy in system_result["strategies"].items():
                labels = strategy.get("oracle_pool_final_labels_by_pair_id")
                if labels is None:
                    continue
                for event in strategy.get("policy_decision_rounds", ()):
                    diagnostics = event.get("selection_diagnostics") or {}
                    probabilities = diagnostics.get("final_stability_probabilities")
                    if probabilities is None:
                        continue
                    candidate_ids = tuple(diagnostics.get("candidate_pair_ids", ()))
                    if set(candidate_ids) != set(probabilities):
                        raise ValueError(f"candidate/probability mismatch in {path}")
                    if not set(candidate_ids) <= set(labels):
                        raise ValueError(f"candidate/label mismatch in {path}")
                    selected_id = str(event["selected_pair_id"])
                    if selected_id != str(diagnostics.get("selected_pair_id")):
                        raise ValueError(f"selected-action diagnostic mismatch in {path}")
                    state_key = (
                        task_sha,
                        str(system),
                        str(policy),
                        str(event["pre_reveal_state_checksum"]),
                    )
                    state = {
                        "candidate_ids": candidate_ids,
                        "probabilities": {key: float(value) for key, value in probabilities.items()},
                        "labels": {key: bool(labels[key]) for key in candidate_ids},
                        "selected_id": selected_id,
                    }
                    prior = seen_states.get(state_key)
                    if prior is not None:
                        if prior != state:
                            raise ValueError(f"conflicting duplicate state {state_key}")
                        continue
                    seen_states[state_key] = state
                    for pair_id in candidate_ids:
                        probability = float(probabilities[pair_id])
                        if not 0.0 <= probability <= 1.0:
                            raise ValueError(f"invalid membership probability {probability}")
                        grouped[(task_sha, str(policy))].append(
                            {
                                "system": str(system),
                                "state_checksum": state_key[-1],
                                "pair_id": pair_id,
                                "probability": probability,
                                "label": int(bool(labels[pair_id])),
                                "selected": pair_id == selected_id,
                            }
                        )
    return grouped, input_hashes


def summarize(
    *,
    input_paths: tuple[Path, ...],
    output: Path,
    bin_count: int = 10,
    bootstrap_count: int = 2000,
    seed: int = 20260809,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("calibration summaries must remain outside Git")
    if not input_paths:
        raise ValueError("no result files supplied")
    if bin_count < 2:
        raise ValueError("bin count must be at least two")
    grouped, input_hashes = _extract(input_paths)
    if not grouped:
        raise ValueError("no membership-probability diagnostics found")
    groups: dict[str, Any] = {}
    for group_index, ((task_sha, policy), rows) in enumerate(sorted(grouped.items())):
        selected_rows = [row for row in rows if row["selected"]]
        top3_rows = _decision_top_k(rows)
        systems = sorted({row["system"] for row in rows})
        groups[f"{task_sha}:{policy}"] = {
            "task_sha256": task_sha,
            "policy": policy,
            "system_count": len(systems),
            "unique_state_count": len({row["state_checksum"] for row in rows}),
            "all_candidates": _population_summary(
                rows,
                bin_count=bin_count,
                bootstrap_count=bootstrap_count,
                seed=seed + group_index,
            ),
            "selected_actions": _population_summary(
                selected_rows,
                bin_count=bin_count,
                bootstrap_count=bootstrap_count,
                seed=seed + 1000 + group_index,
            ),
            "top3_candidates": _population_summary(
                top3_rows,
                bin_count=bin_count,
                bootstrap_count=bootstrap_count,
                seed=seed + 2000 + group_index,
            ),
        }
    result = {
        "schema_version": 2,
        "status": "e52_final_hull_membership_calibration_complete",
        "deduplication_unit": "task_sha256, system, policy, pre_reveal_state_checksum",
        "bootstrap_unit": "exact_chemical_system",
        "bin_count": bin_count,
        "bootstrap_count": bootstrap_count,
        "seed": seed,
        "input_sha256": input_hashes,
        "groups": groups,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-count", type=int, default=10)
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    result = summarize(
        input_paths=tuple(args.input),
        output=args.output,
        bin_count=args.bin_count,
        bootstrap_count=args.bootstrap_count,
        seed=args.seed,
    )
    print(json.dumps({"status": result["status"], "groups": list(result["groups"])}, indent=2))


if __name__ == "__main__":
    main()
