"""Summarize a frozen MAD-1.5 policy probe at exact-system grain.

This is a development-only evaluator.  It reads completed runner output,
never opens the raw vault directly, and writes summaries outside Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _ci(values: np.ndarray, *, seed: int, repetitions: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_pvalue(values: np.ndarray, *, seed: int, repetitions: int) -> float:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(repetitions, len(values)))
    null_means = (signs * values[None, :]).mean(axis=1)
    observed = abs(float(values.mean()))
    return float((np.count_nonzero(np.abs(null_means) >= observed) + 1) / (repetitions + 1))


def run(
    *,
    input_path: Path,
    output_path: Path,
    policy_a: str,
    policy_b: str,
    bootstrap_repetitions: int = 20_000,
    sign_flip_repetitions: int = 100_000,
    seed: int = 20260730,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("MAD probe summary cannot overwrite an existing output")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    systems = payload["query_systems"]
    rows: list[dict[str, Any]] = []
    for system in systems:
        strategies = payload["systems"][system]["strategies"]
        left = strategies[policy_a]
        right = strategies[policy_b]
        rows.append(
            {
                "chemical_system": system,
                "oracle_diff": float(right["oracle_pool_confirmed_discoveries"])
                - float(left["oracle_pool_confirmed_discoveries"]),
                "final_causal_diff": float(right["final_causal_confirmed_discoveries"])
                - float(left["final_causal_confirmed_discoveries"]),
                "wall_seconds_diff": float(right["wall_seconds"]) - float(left["wall_seconds"]),
            }
        )
    oracle = np.asarray([row["oracle_diff"] for row in rows], dtype=float)
    final_causal = np.asarray([row["final_causal_diff"] for row in rows], dtype=float)
    if len(oracle) < 2 or not np.isfinite(oracle).all() or not np.isfinite(final_causal).all():
        raise ValueError("MAD probe differences must contain at least two finite systems")

    def summarize(values: np.ndarray) -> dict[str, Any]:
        interval = _ci(values, seed=seed, repetitions=bootstrap_repetitions)
        return {
            "mean": float(values.mean()),
            "bootstrap_95pct": [interval[0], interval[1]],
            "sign_flip_pvalue_mc": _sign_flip_pvalue(
                values, seed=seed + 1, repetitions=sign_flip_repetitions
            ),
            "wins": int(np.count_nonzero(values > 0)),
            "ties": int(np.count_nonzero(values == 0)),
            "losses": int(np.count_nonzero(values < 0)),
        }

    result = {
        "status": "development_only_mad_probe_summary",
        "input_path": str(input_path.resolve()),
        "input_sha256": __import__("hashlib").sha256(input_path.read_bytes()).hexdigest(),
        "system_count": len(rows),
        "policy_a": policy_a,
        "policy_b": policy_b,
        "seed": seed,
        "bootstrap_repetitions": bootstrap_repetitions,
        "sign_flip_repetitions": sign_flip_repetitions,
        "oracle_difference_policy_b_minus_a": summarize(oracle),
        "final_causal_difference_policy_b_minus_a": summarize(final_causal),
        "mean_wall_seconds_difference_policy_b_minus_a": float(
            np.asarray([row["wall_seconds_diff"] for row in rows], dtype=float).mean()
        ),
        "systems": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "system_count", "oracle_difference_policy_b_minus_a",
        "final_causal_difference_policy_b_minus_a",
        "mean_wall_seconds_difference_policy_b_minus_a",
    )}, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-a", default="source_margin")
    parser.add_argument("--policy-b", default="independent_confirmation_source_rollout")
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--sign-flip-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        policy_a=args.policy_a,
        policy_b=args.policy_b,
        bootstrap_repetitions=args.bootstrap_repetitions,
        sign_flip_repetitions=args.sign_flip_repetitions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
