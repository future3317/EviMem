"""Attach frozen source-structure embeddings to an oracle-isolated MatPES task."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pymatgen.core import Structure

from matmem.frozen_structure_encoder import FrozenCHGNetCrystalEncoder
from matmem.matpes_data import (
    MATPES_PBE_STEM,
    MATPES_SPLITS,
    compact_matpes_configuration,
    iter_matpes_jsonl,
)


class StructureEncoder(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def encode(self, structures: Sequence[Structure], *, batch_size: int) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    *,
    task_path: Path,
    pbe_root: Path,
    output_path: Path,
    device: str,
    batch_size: int,
    encoder: StructureEncoder | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("MatPES embedding augmenter cannot overwrite an output")
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("MatPES representation cache must remain outside Git")
    if batch_size < 1:
        raise ValueError("MatPES embedding batch size must be positive")

    task = json.loads(task_path.read_text(encoding="utf-8"))
    rows = list(task["development_pairs"])
    rows_by_id = {str(row["pair_id"]): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("MatPES task contains duplicate pair IDs")
    selected_ids = set(rows_by_id)
    active_encoder = encoder or FrozenCHGNetCrystalEncoder(device=device)
    embeddings: dict[str, tuple[float, ...]] = {}
    isolated_atom_counts: dict[str, int] = {}
    pending_ids: list[str] = []
    pending_structures: list[Structure] = []

    def flush() -> None:
        if not pending_ids:
            return
        values = active_encoder.encode(pending_structures, batch_size=batch_size)
        if values.shape[0] != len(pending_ids):
            raise RuntimeError("source encoder returned the wrong number of rows")
        for pair_id, value in zip(pending_ids, values, strict=True):
            embeddings[pair_id] = tuple(float(item) for item in value)
        pending_ids.clear()
        pending_structures.clear()

    for split in MATPES_SPLITS:
        path = pbe_root / f"{MATPES_PBE_STEM}-{split}.jsonl"
        for source_row in iter_matpes_jsonl(path):
            pair_id = str(source_row["matpes_id"])
            if pair_id not in selected_ids:
                continue
            if pair_id in embeddings or pair_id in pending_ids:
                raise ValueError("selected MatPES source ID appears more than once")
            compact = compact_matpes_configuration(source_row, split=split)
            if compact.exact_geometry_sha256 != rows_by_id[pair_id]["source_structure_sha256"]:
                raise ValueError("MatPES source structure changed after task construction")
            structure = Structure.from_dict(source_row["structure"])
            center_index, *_ = structure.get_neighbor_list(
                r=6.0,
                sites=structure.sites,
                numerical_tol=1e-8,
            )
            isolated_atom_counts[pair_id] = len(
                set(range(len(structure))) - set(int(value) for value in center_index)
            )
            pending_ids.append(pair_id)
            pending_structures.append(structure)
            if len(pending_ids) >= batch_size:
                flush()
    flush()
    missing = sorted(selected_ids - set(embeddings))
    if missing:
        raise ValueError(f"MatPES embedding scan missed {len(missing)} selected IDs")
    dimensions = {len(value) for value in embeddings.values()}
    if len(dimensions) != 1 or not dimensions or dimensions == {0}:
        raise ValueError("MatPES frozen structure embedding dimension is inconsistent")

    for row in rows:
        row["source_local_environment_embedding"] = embeddings[row["pair_id"]]
        row["source_local_environment_isolated_atom_count"] = isolated_atom_counts[
            row["pair_id"]
        ]
    metadata = dict(active_encoder.metadata)
    metadata.update(
        {
            "dimension": dimensions.pop(),
            "task_source_sha256": _sha256(task_path),
            "augmenter_script_sha256": _sha256(Path(__file__)),
            "selected_pair_count": len(rows),
            "uses_target_outcome": False,
            "isolated_source_structure_count": sum(
                value > 0 for value in isolated_atom_counts.values()
            ),
            "isolated_source_atom_count": sum(isolated_atom_counts.values()),
            "graph_cutoff_angstrom": 6.0,
        }
    )
    task["development_pairs"] = rows
    task["representation_id"] = (
        str(task["representation_id"]) + "+frozen-chgnet-crystal-fea-v1"
    )
    task["local_environment_representation"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "selected_pair_count": len(rows),
        "embedding_dimension": metadata["dimension"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "isolated_source_structure_count": metadata[
            "isolated_source_structure_count"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--pbe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(
        task_path=args.task,
        pbe_root=args.pbe_root,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
