"""Split a trusted JARVIS--MP oracle vault into calibration and sealed evaluation files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    if len(calibration) + len(evaluation) != len(rows):
        raise ValueError("oracle vault contains an unregistered split")
    ids = [row["pair_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("oracle vault pair IDs are not unique")
    return calibration, evaluation


def split(source: Path, output_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir.resolve().is_relative_to(repo_root):
        raise ValueError("oracle vault outputs must remain outside Git")
    paths = {
        "calibration": output_dir / "calibration-oracle-vault.json",
        "evaluation": output_dir / "sealed-evaluation-oracle-vault.json",
        "audit": output_dir / "oracle-vault-split-audit.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("split oracle vault already exists; never overwrite it")
    payload = json.loads(source.read_text(encoding="utf-8"))
    calibration, evaluation = _partition(payload["target_outcomes"])
    common = {
        "schema_version": 1,
        "release_id": payload["release_id"],
        "task_manifest_sha256": payload["task_manifest_sha256"],
        "source_oracle_vault_sha256": _sha256(source),
    }
    calibration_payload = common | {
        "access_contract": "calibration outcomes only; contains no evaluation row",
        "target_outcomes": calibration,
    }
    evaluation_payload = common | {
        "access_contract": "sealed evaluation; forbidden until calibration freeze passes",
        "target_outcomes": evaluation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["calibration"].write_text(
        json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["evaluation"].write_text(
        json.dumps(evaluation_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = common | {
        "calibration_row_count": len(calibration),
        "evaluation_row_count": len(evaluation),
        "calibration_vault_sha256": _sha256(paths["calibration"]),
        "sealed_evaluation_vault_sha256": _sha256(paths["evaluation"]),
        "split_overlap": sorted(
            {row["pair_id"] for row in calibration}
            & {row["pair_id"] for row in evaluation}
        ),
        "evaluation_results_accessed_by_calibration_runner": False,
    }
    paths["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"calibration_rows={len(calibration)}")
    print(f"evaluation_rows={len(evaluation)}")
    print(f"calibration_vault={paths['calibration'].resolve()}")
    print(f"sealed_evaluation_vault={paths['evaluation'].resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    split(args.source, args.output_dir)


if __name__ == "__main__":
    main()
