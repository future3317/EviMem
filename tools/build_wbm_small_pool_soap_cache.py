"""Create the frozen periodic SOAP cache for the oracle-blind small WBM pools."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from build_wbm_small_pool_manifest import _read_cleaned_ids, read_observable_candidates


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _raw_entries_by_id(
    cse_root: Path, structures_root: Path, cleaned_ids: Path, selected: set[str]
) -> dict[str, dict]:
    # Reuse the audited alignment path, then retain only selected CSE structures.
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
    # CSE sequence has the same audited source order; recover records by checksum.
    # Invert once: scanning all five CSE files must not be O(all_records * pool_size).
    query_ids_by_hash: dict[str, list[str]] = {}
    for query_id, checksum in selected_hashes.items():
        query_ids_by_hash.setdefault(checksum, []).append(query_id)
    records: dict[str, dict] = {}
    for step in range(1, 6):
        payload = json.loads(bz2.decompress((cse_root / f"step_{step}.json.bz2").read_bytes()))
        for entry in payload["entries"]:
            structure = entry.get("structure")
            checksum = "sha256:" + hashlib.sha256(
                json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for query_id in query_ids_by_hash.get(checksum, ()):
                records[query_id] = entry
    if set(records) != selected:
        raise ValueError("could not recover every selected CSE structure by its frozen checksum")
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
    records = _raw_entries_by_id(
        args.cse_root, args.structures_root, args.cleaned_ids, set(pool_by_id)
    )
    from dscribe.descriptors import SOAP
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    species = tuple(sorted({element for entry in records.values() for element in entry["composition"]}))
    soap = SOAP(species=species, r_cut=5.0, n_max=8, l_max=6, periodic=True, sparse=False)
    identifiers = tuple(sorted(records))
    vectors = []
    for query_id in identifiers:
        atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(records[query_id]["structure"]))
        vector = np.asarray(soap.create(atoms), dtype=np.float64).mean(axis=0)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"invalid SOAP vector for {query_id}")
        vectors.append(vector / norm)
    matrix = np.vstack(vectors)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, query_ids=np.asarray(identifiers), vectors=matrix)
    manifest = {
        "scope": "oracle_blind_wbm_small_pool_soap_cache",
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
