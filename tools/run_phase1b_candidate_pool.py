"""Build the unlabeled SciMem-Update pilot pool from licensed Phase 1B sources.

Raw public releases and Crossref responses stay outside the repository. The
committed JSONL is a standard Label Studio task import and contains neither
compiled operations nor update gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from evimem.benchmark import DataView, SciFactAdapter, SciRexAdapter, ViewSample, sha256_file
from evimem.phase1b import (
    CandidateSide,
    UpdatePilotCandidate,
    crossref_update_candidates,
    validate_candidate_pool,
)
from evimem.phase1b.candidates import sha256_json

CROSSREF_ENDPOINT = "https://api.crossref.org/v1/works"
USER_AGENT = "EviMemPhase1B/0.1 (mailto:future3317@users.noreply.github.com)"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _claim_text(sample: ViewSample) -> str:
    claim = sample.claim
    if claim is None:
        return sample.query_text
    parts = [claim.subject, claim.relation]
    if claim.object:
        parts.extend(["for", claim.object])
    if claim.value is not None:
        parts.extend(["=", str(claim.value)])
    if claim.unit:
        parts.append(claim.unit)
    if claim.condition:
        parts.append(f"under {json.dumps(claim.condition, ensure_ascii=False, sort_keys=True)}")
    return " ".join(parts)


def _side(sample: ViewSample, *, claim_text: str | None = None) -> CandidateSide:
    evidence = sample.evidence
    locator = ";".join(
        f"{span.source_field}:{span.start}-{span.end}" for span in evidence
    ) or "structured annotation without text span"
    checksum_payload = {
        "sample_id": sample.sample_id,
        "claim": claim_text or _claim_text(sample),
        "evidence": [span.model_dump(mode="json") for span in evidence],
        "source_row_checksum": sample.metadata.get("source_row_checksum"),
    }
    return CandidateSide(
        claim_text=claim_text or _claim_text(sample),
        source_document_id=sample.source_document_id,
        evidence_locator=locator,
        evidence_checksum=sha256_json(checksum_payload),
    )


def _load_scirex_train(root: Path) -> tuple[list[ViewSample], list[Path]]:
    path = root / "release_data" / "train.jsonl"
    adapter = SciRexAdapter()
    samples: list[ViewSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        samples.extend(adapter.convert_views(json.loads(line), split="train"))
    return samples, [path]


def _load_scifact_train(root: Path) -> tuple[list[ViewSample], list[Path]]:
    corpus_path = root / "data" / "corpus.jsonl"
    claims_path = root / "data" / "claims_train.jsonl"
    corpus = {
        str(item["doc_id"]): item
        for item in (
            json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()
        )
    }
    adapter = SciFactAdapter()
    samples: list[ViewSample] = []
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        claim = json.loads(line)
        for document_id, evidence_sets in claim.get("evidence", {}).items():
            source = corpus[str(document_id)]
            for evidence_index, annotation in enumerate(evidence_sets):
                samples.extend(
                    adapter.convert_views(
                        {
                            "id": f"{claim['id']}:{evidence_index}",
                            "document_id": document_id,
                            "claim": claim["claim"],
                            "label": annotation["label"],
                            "abstract_sentences": source["abstract"],
                            "evidence_sentence_ids": annotation["sentences"],
                        },
                        split="train",
                    )
                )
    return samples, [claims_path, corpus_path]


def _scirex_stratum(left: ViewSample, right: ViewSample) -> str:
    assert left.claim is not None and right.claim is not None
    if left.claim.canonical_key() == right.claim.canonical_key():
        return "possible_equivalent_not_gold"
    if left.claim.canonical_key(include_value=False) == right.claim.canonical_key(
        include_value=False
    ):
        return "possible_same_scope_conflict_not_gold"
    if (
        left.claim.subject == right.claim.subject
        or left.claim.object == right.claim.object
        or left.claim.relation == right.claim.relation
    ):
        return "possible_related_scope_not_gold"
    return "unrelated_hard_negative_candidate_not_gold"


def build_scirex_candidates(samples: list[ViewSample], *, count: int = 160) -> list[UpdatePilotCandidate]:
    eligible = sorted(
        (
            sample
            for sample in samples
            if sample.split == "train" and sample.view == DataView.RETRIEVAL and sample.claim
        ),
        key=lambda item: item.sample_id,
    )
    pools: dict[str, list[tuple[ViewSample, ViewSample]]] = {}
    for left, right in combinations(eligible, 2):
        stratum = _scirex_stratum(left, right)
        pools.setdefault(stratum, []).append((left, right))

    quotas = {
        "possible_equivalent_not_gold": 30,
        "possible_same_scope_conflict_not_gold": 40,
        "possible_related_scope_not_gold": 50,
        "unrelated_hard_negative_candidate_not_gold": 40,
    }
    selected: list[tuple[str, ViewSample, ViewSample]] = []
    used: set[tuple[str, str]] = set()
    for stratum, quota in quotas.items():
        for left, right in pools.get(stratum, ())[:quota]:
            key = (left.sample_id, right.sample_id)
            if key not in used:
                used.add(key)
                selected.append((stratum, left, right))

    if len(selected) < count:
        for stratum in quotas:
            for left, right in pools.get(stratum, ()):
                key = (left.sample_id, right.sample_id)
                if key in used:
                    continue
                used.add(key)
                selected.append((stratum, left, right))
                if len(selected) == count:
                    break
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"SciREX produced {len(selected)} of {count} requested pairs")

    output = []
    for index, (stratum, left, right) in enumerate(selected):
        output.append(
            UpdatePilotCandidate(
                pair_id=f"scirex:{index:04d}",
                source_dataset="SciREX",
                split="train",
                left=_side(left),
                right=_side(right),
                sampling_stratum=stratum,
                license_components=("annotations:Apache-2.0", "source_text:Apache-2.0"),
            )
        )
    return output


def build_scifact_candidates(
    samples: list[ViewSample],
    *,
    leaked_documents: set[str],
    count: int = 160,
) -> list[UpdatePilotCandidate]:
    eligible = [
        sample
        for sample in samples
        if sample.split == "train"
        and sample.source_document_id not in leaked_documents
        and sample.evidence
        and sample.claim is not None
    ]
    by_label: dict[str, list[ViewSample]] = {}
    for sample in sorted(eligible, key=lambda item: item.sample_id):
        label = str(sample.target.get("veracity_label") or "UNKNOWN").upper()
        by_label.setdefault(label, []).append(sample)
    selected = by_label.get("SUPPORT", ())[: count // 2] + by_label.get("CONTRADICT", ())[
        : count // 2
    ]
    if len(selected) < count:
        used = {item.sample_id for item in selected}
        selected.extend(item for item in eligible if item.sample_id not in used)
        selected = selected[:count]
    if len(selected) != count:
        raise RuntimeError(f"SciFact produced {len(selected)} of {count} requested pairs")

    output = []
    for index, sample in enumerate(selected):
        label = str(sample.target.get("veracity_label") or "UNKNOWN").upper()
        evidence_text = " ".join(span.text for span in sample.evidence)
        right = CandidateSide(
            claim_text=evidence_text,
            source_document_id=sample.source_document_id,
            evidence_locator=";".join(
                f"{span.source_field}:{span.start}-{span.end}" for span in sample.evidence
            ),
            evidence_checksum=sha256_json(
                [span.model_dump(mode="json") for span in sample.evidence]
            ),
        )
        output.append(
            UpdatePilotCandidate(
                pair_id=f"scifact:{index:04d}",
                source_dataset="SciFact",
                split="train",
                left=_side(sample, claim_text=sample.query_text),
                right=right,
                sampling_stratum=f"native_{label.lower()}_candidate_not_update_gold",
                license_components=("annotations:CC-BY-4.0", "source_text:ODC-By-1.0"),
            )
        )
    return output


def _fetch_crossref(update_type: str, raw_dir: Path) -> tuple[dict[str, Any], str, str]:
    query = urllib.parse.urlencode({"filter": f"update-type:{update_type}", "rows": 100})
    url = f"{CROSSREF_ENDPOINT}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"crossref_{update_type}.json"
    path.write_bytes(payload)
    checksum = "sha256:" + hashlib.sha256(payload).hexdigest()
    return json.loads(payload), checksum, url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scirex", type=Path, required=True)
    parser.add_argument("--scifact", type=Path, required=True)
    parser.add_argument("--leakage-report", type=Path, required=True)
    parser.add_argument("--raw-crossref-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    scirex_samples, scirex_files = _load_scirex_train(args.scirex)
    scifact_samples, scifact_files = _load_scifact_train(args.scifact)
    leakage = json.loads(args.leakage_report.read_text(encoding="utf-8"))
    leaked_documents = set(leakage["source_document_cross_split"])

    retraction_response, retraction_checksum, retraction_url = _fetch_crossref(
        "retraction", args.raw_crossref_dir
    )
    correction_response, correction_checksum, correction_url = _fetch_crossref(
        "correction", args.raw_crossref_dir
    )
    crossref = crossref_update_candidates(
        retraction_response, response_checksum=retraction_checksum, limit=20
    ) + crossref_update_candidates(
        correction_response, response_checksum=correction_checksum, limit=20
    )
    deduplicated = {item.pair_id: item for item in crossref}
    crossref = list(deduplicated.values())
    if len(crossref) != 40:
        raise RuntimeError(f"Crossref produced {len(crossref)} of 40 requested pairs")

    candidates = (
        build_scirex_candidates(scirex_samples)
        + build_scifact_candidates(scifact_samples, leaked_documents=leaked_documents)
        + crossref
    )
    forbidden = {f"doi:{item}" for item in leaked_documents} | leaked_documents
    validation = validate_candidate_pool(candidates, forbidden_document_ids=forbidden)

    root = args.repository_root.resolve()
    annotation_path = root / "annotation" / "scimem_update_pilot_unlabeled.jsonl"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        "".join(
            json.dumps(item.label_studio_task(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in candidates
        ),
        encoding="utf-8",
    )

    source_counts = Counter(item.source_dataset for item in candidates)
    stratum_counts = Counter(item.sampling_stratum for item in candidates)
    update_types = Counter(
        item.source_level_update.update_type
        for item in candidates
        if item.source_level_update is not None
    )
    update_sources = Counter(
        item.source_level_update.source
        for item in candidates
        if item.source_level_update is not None
    )
    possible_conflict = [
        item.pair_id
        for item in candidates
        if item.sampling_stratum
        in {
            "possible_same_scope_conflict_not_gold",
            "native_contradict_candidate_not_update_gold",
        }
    ]
    source_level_only = [item.pair_id for item in crossref]
    distribution = {
        "candidate_count": len(candidates),
        "candidate_is_gold": False,
        "human_annotation_completed": False,
        "source_distribution": dict(sorted(source_counts.items())),
        "sampling_strata_not_gold": dict(sorted(stratum_counts.items())),
        "crossref_update_types": dict(sorted(update_types.items())),
        "crossref_metadata_sources": dict(sorted(update_sources.items())),
        "possible_genuine_conflict_pair_ids_not_gold": possible_conflict,
        "source_level_only_pair_ids": source_level_only,
        "validation": validation,
    }
    _write_json(root / "reports" / "phase1b" / "candidate_distribution.json", distribution)

    license_report = {
        "passed_for_candidate_pool": True,
        "full_text_copied": False,
        "sources": {
            "SciREX": {
                "components": ["annotations", "source_text"],
                "status": "confirmed",
                "spdx": "Apache-2.0",
                "manifest": "configs/datasets.json",
            },
            "SciFact": {
                "components": ["annotations", "source_text"],
                "status": "confirmed",
                "spdx": ["CC-BY-4.0", "ODC-By-1.0"],
                "manifest": "configs/datasets.json",
            },
            "Crossref/Retraction Watch": {
                "components": ["DOI", "title", "update-to factual metadata"],
                "status": "confirmed_for_selected_factual_metadata",
                "source_url": "https://www.crossref.org/documentation/retrieve-metadata/rest-api/",
                "retraction_watch_url": "https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/",
                "license_basis": (
                    "Crossref states that almost none of its REST metadata is subject to "
                    "copyright and it may be used for any purpose. Abstracts are excluded."
                ),
                "attribution": "Cite Crossref and Retraction Watch; retain DOI/update provenance.",
                "abstracts_included": False,
                "full_text_included": False,
            },
        },
        "blocked_sources_excluded": [
            "QASPER",
            "Evidence Inference 2.0",
            "MeasEval",
            "BioRED",
        ],
        "ood_sources_excluded": ["POLYIE"],
    }
    _write_json(root / "reports" / "phase1b" / "source_license_report.json", license_report)

    source_files = sorted({path.resolve() for path in scirex_files + scifact_files if path.exists()})
    manifest = {
        "schema_version": "evimem.scimem_update_pilot_manifest.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "phase": "1B",
        "annotation_status": "unlabeled",
        "candidate_is_gold": False,
        "compiled_operation_present": False,
        "candidate_count": len(candidates),
        "annotation_format": "Label Studio JSONL task import",
        "annotation_file": "annotation/scimem_update_pilot_unlabeled.jsonl",
        "annotation_file_checksum": sha256_file(annotation_path),
        "source_snapshots": {
            "dataset_files": [
                {"name": path.name, "sha256": sha256_file(path)} for path in source_files
            ],
            "crossref_api": [
                {
                    "source": retraction_url,
                    "update_type": "retraction",
                    "response_checksum": retraction_checksum,
                    "raw_response_committed": False,
                },
                {
                    "source": correction_url,
                    "update_type": "correction",
                    "response_checksum": correction_checksum,
                    "raw_response_committed": False,
                },
            ],
        },
        "safety": {
            "no_copyrighted_full_text": True,
            "crossref_source_status_is_not_claim_supersession": True,
            "claim_level_supersession_requires_human_evidence_annotation": True,
            "annotators_cannot_see_compiled_operation": True,
        },
        "validation": validation,
    }
    _write_json(root / "data_manifests" / "scimem_update_pilot_manifest.json", manifest)
    print(json.dumps({"candidate_count": len(candidates), "sources": source_counts}, default=dict))


if __name__ == "__main__":
    main()
