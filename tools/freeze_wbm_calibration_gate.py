"""Freeze the GP configuration and probability-loss margins from calibration only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
SRC_ROOT = TOOLS_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmem import paired_system_improvement_bootstrap  # noqa: E402

BOOTSTRAP_SEED = 20270718
BOOTSTRAP_ITERATIONS = 10000
METRICS = (
    "boundary_weighted_causal_brier",
    "boundary_weighted_causal_log_loss",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(config["posterior"], sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def _calibration_systems(manifest: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, pool in manifest["selection"]["pools"].items():
        result[name] = str(pool["chemical_complexity_stratum"])
    if list(sorted(result.values())).count("binary") != 4 or list(sorted(result.values())).count("ternary") != 4:
        raise ValueError("margin calibration requires exactly four binary and four ternary systems")
    return result


def _finite_full_history_sanity(runs: list[dict[str, Any]], systems: set[str]) -> dict[str, Any]:
    full = {row["pool"]: row for row in runs if row["strategy"] == "full_history"}
    if set(full) != systems:
        raise ValueError("calibration summary must include exactly one full-history run per system")
    problems: list[str] = []
    for system, row in sorted(full.items()):
        if row["budget"] != 12 or len(row["prequential_rounds"]) != 12:
            problems.append(f"{system}: expected 12 full-history prequential rounds")
            continue
        for metric in (
            "boundary_weighted_causal_crps",
            "boundary_weighted_causal_brier",
            "boundary_weighted_causal_log_loss",
        ):
            value = row["prequential"].get(metric)
            if value is None or not math.isfinite(float(value)):
                problems.append(f"{system}: non-finite {metric}")
        brier = row["prequential"].get("boundary_weighted_causal_brier")
        log_loss = row["prequential"].get("boundary_weighted_causal_log_loss")
        crps = row["prequential"].get("boundary_weighted_causal_crps")
        if brier is not None and not 0.0 <= float(brier) <= 1.0:
            problems.append(f"{system}: Brier is outside [0, 1]")
        if log_loss is not None and float(log_loss) < 0.0:
            problems.append(f"{system}: log loss is negative")
        if crps is not None and float(crps) < 0.0:
            problems.append(f"{system}: CRPS is negative")
    return {"passed": not problems, "problems": problems, "system_count": len(full)}


def freeze_gate(
    *, config: dict[str, Any], manifest: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    """Return a fully determined calibration-only GP and margin manifest."""

    systems_by_stratum = _calibration_systems(manifest)
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("calibration summary has no runs")
    allowed = {"fifo", "gp_variance_one_swap", "full_history"}
    unexpected = {str(row["strategy"]) for row in runs} - allowed
    if unexpected:
        raise ValueError(f"margin calibration must not execute other selectors: {sorted(unexpected)}")
    systems = set(systems_by_stratum)
    sanity = _finite_full_history_sanity(runs, systems)
    if not sanity["passed"]:
        raise ValueError("full-history GP calibration sanity failed: " + "; ".join(sanity["problems"]))
    margins: dict[str, Any] = {}
    for metric in METRICS:
        values = {
            strategy: {
                row["pool"]: float(row["prequential"][metric])
                for row in runs
                if row["strategy"] == strategy
            }
            for strategy in ("fifo", "gp_variance_one_swap")
        }
        if any(set(by_system) != systems for by_system in values.values()):
            raise ValueError(f"{metric} requires FIFO and GP-variance values for every calibration system")
        bootstrap = paired_system_improvement_bootstrap(
            values["fifo"],
            values["gp_variance_one_swap"],
            seed=BOOTSTRAP_SEED,
            iterations=BOOTSTRAP_ITERATIONS,
        )
        g_lb = max(0.0, float(bootstrap["ci95_low"]))
        gp_variance_macro_loss = sum(values["gp_variance_one_swap"].values()) / len(systems)
        margin = min(0.10 * gp_variance_macro_loss, 0.20 * g_lb)
        margins[metric] = {
            "fifo_macro_loss": sum(values["fifo"].values()) / len(systems),
            "gp_variance_macro_loss": gp_variance_macro_loss,
            "gp_variance_improvement_over_fifo": bootstrap,
            "margin": margin,
            "rule": "min(0.10 * gp_variance_macro_loss, 0.20 * max(improvement_ci95_low, 0))",
        }
    return {
        "schema_version": "wbm-gp-and-noninferiority-calibration-freeze-v1",
        "scope": "disjoint_calibration_only_no_evaluation_results_accessed",
        "gp_config": config["posterior"],
        "gp_config_sha256": _config_sha256(config),
        "gp_parameter_status": "frozen_on_disjoint_calibration_systems_v1",
        "full_history_prequential_sanity": sanity,
        "margin_rule": "min(0.10 * gp_variance_macro_loss, 0.20 * max(improvement_ci95_low, 0))",
        "brier_margin": margins["boundary_weighted_causal_brier"]["margin"],
        "log_loss_margin": margins["boundary_weighted_causal_log_loss"]["margin"],
        "metrics": margins,
        "calibration_system_ids": sorted(systems),
        "calibration_strata": systems_by_stratum,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "evaluation_results_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(repo_root):
        parser.error("calibration freeze output must remain outside the repository")
    if args.output.exists():
        raise FileExistsError("calibration freeze manifest is immutable")
    result = freeze_gate(
        config=json.loads(args.config.read_text(encoding="utf-8")),
        manifest=json.loads(args.calibration_manifest.read_text(encoding="utf-8")),
        summary=json.loads(args.calibration_summary.read_text(encoding="utf-8")),
    )
    result["calibration_manifest_sha256"] = _sha256(args.calibration_manifest)
    result["calibration_summary_sha256"] = _sha256(args.calibration_summary)
    result["config_sha256"] = _sha256(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"brier_margin={result['brier_margin']:.12g}")
    print(f"log_loss_margin={result['log_loss_margin']:.12g}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
