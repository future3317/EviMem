"""Create the frozen periodic SOAP cache for the oracle-blind small WBM pools."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from build_wbm_small_pool_manifest import (
    STEP3_ANOMALOUS_STRUCTURE_IDS,
    _extract_initial_structure,
    _fix_step3_alignment,
    _read_cleaned_ids,
    _source_to_wbm_id,
    read_observable_candidates,
)

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmem.identity import (  # noqa: E402
    StructureStage,
    WBMStructureSourceField,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _initial_structures_by_id(
    cse_root: Path, structures_root: Path, cleaned_ids: Path, selected: set[str]
) -> dict[str, dict]:
    """Recover only pre-query WBM initial structures for selected IDs."""

    # Reuse the audited alignment path so the selected hashes are explicitly
    # bound to the initial-structure source rather than to relaxed CSE entries.
    candidates = read_observable_candidates(
        cse_root=cse_root, structures_root=structures_root, cleaned_ids=_read_cleaned_ids(cleaned_ids)
    )
    selected_hashes = {
        item.query_id: item.exact_structure_sha256
        for item in candidates
        if item.query_id in selected
    }
    if set(selected_hashes) != selected:
        raise ValueError("selected pool IDs are absent from cleaned observable candidates")
    records: dict[str, dict] = {}
    anomalies = set(STEP3_ANOMALOUS_STRUCTURE_IDS)
    for step in range(1, 6):
        structures = json.loads(
            bz2.decompress(
                (structures_root / f"wbm-structures-step-{step}.json.bz2").read_bytes()
            )
        )
        for source_id, structure_record in structures.items():
            if source_id in anomalies:
                continue
            query_id = _fix_step3_alignment(_source_to_wbm_id(source_id))
            if query_id not in selected:
                continue
            structure = _extract_initial_structure(structure_record, query_id=query_id)
            checksum = "sha256:" + hashlib.sha256(
                json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if checksum != selected_hashes[query_id]:
                raise ValueError(f"initial-structure checksum mismatch for {query_id}")
            records[query_id] = structure
    if set(records) != selected:
        raise ValueError("could not recover every selected WBM initial structure")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--cse-root", type=Path, required=True)
    parser.add_argument("--structures-root", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.output_npz.resolve().is_relative_to(repo_root) or args.output_manifest.resolve().is_relative_to(repo_root):
        parser.error("SOAP cache outputs must remain outside the repository")
    pool_manifest = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    pool_by_id = {
        candidate["query_id"]: name
        for name, pool in pool_manifest["selection"]["pools"].items()
        for candidate in pool["candidates"]
    }
    records = _initial_structures_by_id(
        args.cse_root, args.structures_root, args.cleaned_ids, set(pool_by_id)
    )
    from dscribe.descriptors import SOAP
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    structures = {
        query_id: Structure.from_dict(structure)
        for query_id, structure in records.items()
    }
    species = tuple(
        sorted(
            {
                str(element)
                for structure in structures.values()
                for element in structure.composition.elements
            }
        )
    )
    soap = SOAP(species=species, r_cut=5.0, n_max=8, l_max=6, periodic=True, sparse=False)
    identifiers = tuple(sorted(records))
    vectors = []
    for query_id in identifiers:
        atoms = AseAtomsAdaptor.get_atoms(structures[query_id])
        vector = np.asarray(soap.create(atoms), dtype=np.float64).mean(axis=0)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"invalid SOAP vector for {query_id}")
        vectors.append(vector / norm)
    matrix = np.vstack(vectors)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, query_ids=np.asarray(identifiers), vectors=matrix)
    manifest = {
        "scope": "oracle_blind_wbm_initial_structure_soap_cache",
        "structure_stage": StructureStage.INITIAL.value,
        "causal_available_before_query": True,
        "structure_source_pattern": "wbm-structures-step-{1..5}.json.bz2",
        "structure_source_field": WBMStructureSourceField.ORIGINAL.value,
        "pool_manifest_sha256": _sha256_file(args.pool_manifest),
        "cache_sha256": _sha256_file(args.output_npz),
        "record_count": len(identifiers), "vector_dimension": int(matrix.shape[1]),
        "species": list(species), "periodic": True, "cutoff_angstrom": 5.0, "n_max": 8, "l_max": 6,
        "dscribe_version": importlib.metadata.version("dscribe"), "pool_by_id": pool_by_id,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"records={len(identifiers)} dimensions={matrix.shape[1]} cache={args.output_npz}")


if __name__ == "__main__":
    main()
