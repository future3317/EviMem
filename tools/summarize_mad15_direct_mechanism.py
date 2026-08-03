"""Summarize the registered four-policy MAD-1.5 direct comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

POLICIES = (
    "source_margin",
    "delta_hull_active_search",
    "ungated_source_rollout",
    "independent_confirmation_source_rollout",
)
METRICS = {
    "D": "causal_discoveries",
    "F": "final_causal_confirmed_discoveries",
    "T": "oracle_pool_confirmed_discoveries",
    "D_minus_F": "within_campaign_revocations",
    "F_minus_T": "unqueried_competitor_invalidations",
    "wall_seconds": "wall_seconds",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("paired values must be finite and nonempty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    bootstrap = values[indices].mean(axis=1)
    signs = np.random.default_rng(seed + 1).choice(
        (-1.0, 1.0), size=(100_000, len(values))
    )
    observed = abs(float(values.mean()))
    randomized = np.abs((signs * values[None, :]).mean(axis=1))
    return {
        "mean": float(values.mean()),
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "sign_flip_p": float((np.sum(randomized >= observed) + 1) / 100_001),
        "wins": int(np.sum(values > 0)),
        "ties": int(np.sum(values == 0)),
        "losses": int(np.sum(values < 0)),
        "system_count": int(len(values)),
        "bootstrap_replicates": 20_000,
        "sign_flip_draws": 100_000,
        "seed": seed,
    }


def _load(path: Path, budget: int) -> dict[str, Any]:
    if not path.exists():
        failure = path.with_suffix(".failure.json")
        if failure.exists():
            raise RuntimeError(f"MAD direct output failed: {failure}")
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload.get("active_policies", ())) != POLICIES:
        raise ValueError(f"unexpected MAD direct roster in {path}")
    if int(payload["config"]["query_budget"]) != budget:
        raise ValueError(f"wrong MAD direct budget in {path}")
    return payload


def _rows(payloads: list[dict[str, Any]], policy: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for system, result in payload["systems"].items():
            if system in rows:
                raise ValueError(f"duplicate MAD exact system {system}")
            rows[system] = result["strategies"][policy]
    return rows


def _contrast(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], seed: int
) -> dict[str, Any]:
    systems = sorted(left)
    if set(systems) != set(right):
        raise ValueError("MAD direct paired systems differ")
    result: dict[str, Any] = {"system_count": len(systems)}
    for offset, (name, field) in enumerate(METRICS.items()):
        values = np.asarray([float(left[s][field]) - float(right[s][field]) for s in systems])
        result[name] = _paired(values, seed + offset)
    return result


def summarize(*, inputs: tuple[Path, ...], manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("MAD summaries must remain outside Git")
    if len(inputs) != 6:
        raise ValueError("MAD direct summary requires B=1..6")
    payloads = [_load(path, index) for index, path in enumerate(inputs, start=1)]
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected_systems = tuple(manifest_payload["selected_systems"])
    if any(tuple(payload["query_systems"]) != expected_systems for payload in payloads):
        raise ValueError("MAD direct outputs do not share the frozen query manifest")
    policies = {tuple(payload["active_policies"]) for payload in payloads}
    if policies != {POLICIES}:
        raise ValueError("MAD direct policy rosters differ")
    curves = {policy: {metric: [0.0] for metric in METRICS.values()} for policy in POLICIES}
    pairwise: dict[str, Any] = {}
    for budget, payload in enumerate(payloads, start=1):
        rows = {policy: _rows([payload], policy) for policy in POLICIES}
        for policy in POLICIES:
            for field in METRICS.values():
                curves[policy][field].append(
                    float(np.mean([float(rows[policy][s][field]) for s in expected_systems]))
                )
        pairwise[str(budget)] = {
            f"{left}_minus_{right}": _contrast(rows[left], rows[right], 20260810 + budget)
            for left, right in (
                ("delta_hull_active_search", "source_margin"),
                ("ungated_source_rollout", "delta_hull_active_search"),
                ("independent_confirmation_source_rollout", "ungated_source_rollout"),
                ("independent_confirmation_source_rollout", "delta_hull_active_search"),
            )
        }
    auc = {
        policy: {
            field: float(np.trapezoid(np.asarray(values)))
            for field, values in metrics.items()
        }
        for policy, metrics in curves.items()
    }
    result = {
        "status": "complete_mad15_direct_mechanism_development_summary",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "inputs": [str(path.resolve()) for path in inputs],
        "input_sha256": {str(path): _sha256(path) for path in inputs},
        "system_count": len(expected_systems),
        "budgets": list(range(7)),
        "policies": list(POLICIES),
        "curves": curves,
        "auc": auc,
        "direct_pairwise_by_budget": pairwise,
        "semantics": "MAD atomization-energy convex-hull proxy, not formation-energy hull",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(inputs=tuple(args.input), manifest=args.manifest, output=args.output)
    print(json.dumps({"status": result["status"], "system_count": result["system_count"]}))


if __name__ == "__main__":
    main()
