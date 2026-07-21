"""Summarize paired exact-system MatPES confirmatory results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratum(system: str) -> str:
    count = len(system.split("-"))
    return "binary" if count == 2 else "ternary" if count == 3 else "quaternary_or_higher"


def _exact_sign_flip_two_sided(differences: np.ndarray) -> float:
    scaled = np.rint(np.asarray(differences, dtype=float) * 1_000_000).astype(np.int64)
    counts: dict[int, int] = {0: 1}
    for value in scaled:
        updated: dict[int, int] = defaultdict(int)
        for total, count in counts.items():
            updated[total + int(value)] += count
            updated[total - int(value)] += count
        counts = dict(updated)
    observed = abs(int(np.sum(scaled)))
    extreme = sum(count for total, count in counts.items() if abs(total) >= observed)
    return float(extreme / (2 ** len(scaled)))


def _paired_summary(
    differences: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
    bootstrap = values[indices].mean(axis=1)
    return {
        "system_count": len(values),
        "paired_mean_difference": float(values.mean()),
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wins": int(np.sum(values > 0)),
        "ties": int(np.sum(values == 0)),
        "losses": int(np.sum(values < 0)),
        "exact_two_sided_sign_flip_p": _exact_sign_flip_two_sided(values),
    }


def summarize(
    *,
    result_path: Path,
    output_path: Path,
    primary_policy: str = "delta_hull_active_search",
    comparison_policies: tuple[str, ...] = (
        "source_margin",
        "ridge_margin",
        "ridge_predicted_final_margin",
    ),
    metric: str = "oracle_pool_confirmed_discoveries",
    bootstrap_seed: int = 20270721,
    bootstrap_replicates: int = 50_000,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("confirmatory summary must remain outside Git")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("split") != "confirmatory" or not result.get("evaluation_systems_accessed"):
        raise ValueError("summary requires an opened confirmatory result")
    systems = sorted(result["systems"])
    summaries: dict[str, Any] = {}
    for comparison in comparison_policies:
        differences = np.asarray(
            [
                result["systems"][system]["strategies"][primary_policy][metric]
                - result["systems"][system]["strategies"][comparison][metric]
                for system in systems
            ],
            dtype=float,
        )
        strata: dict[str, Any] = {}
        for stratum in ("binary", "ternary", "quaternary_or_higher"):
            selected = np.asarray(
                [value for system, value in zip(systems, differences, strict=True) if _stratum(system) == stratum]
            )
            if len(selected):
                strata[stratum] = _paired_summary(
                    selected,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_replicates=bootstrap_replicates,
                )
        summaries[comparison] = {
            **_paired_summary(
                differences,
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            ),
            "by_stratum": strata,
        }
    output = {
        "schema_version": 1,
        "status": "opened_confirmatory_summary",
        "result_sha256": _sha256(result_path),
        "primary_policy": primary_policy,
        "metric": metric,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "comparisons": summaries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-policy", default="delta_hull_active_search")
    parser.add_argument(
        "--comparison-policies",
        nargs="+",
        default=("source_margin", "ridge_margin", "ridge_predicted_final_margin"),
    )
    parser.add_argument("--metric", default="oracle_pool_confirmed_discoveries")
    parser.add_argument("--bootstrap-seed", type=int, default=20270721)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    args = parser.parse_args()
    summarize(
        result_path=args.result,
        output_path=args.output,
        primary_policy=args.primary_policy,
        comparison_policies=tuple(args.comparison_policies),
        metric=args.metric,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
