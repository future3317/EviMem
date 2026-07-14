from __future__ import annotations

import pytest

from evimem.phase1b import crossref_update_candidates, validate_candidate_pool


def crossref_response() -> dict:
    return {
        "message": {
            "items": [
                {
                    "DOI": "10.1000/update",
                    "title": ["Correction notice"],
                    "update-to": [
                        {
                            "DOI": "10.1000/original",
                            "type": "correction",
                            "source": "retraction-watch",
                            "updated": {"date-time": "2026-01-02T00:00:00Z"},
                        }
                    ],
                }
            ]
        }
    }


def test_crossref_maps_only_source_level_status() -> None:
    candidates = crossref_update_candidates(
        crossref_response(), response_checksum="sha256:" + "a" * 64, limit=1
    )
    assert len(candidates) == 1
    update = candidates[0].source_level_update
    assert update is not None
    assert update.status_scope == "source_level_only"
    assert update.claim_level_status == "awaiting_human_evidence_annotation"
    dumped = candidates[0].model_dump()
    assert "compiled_operation" not in dumped
    assert "SUPERSEDE" not in candidates[0].model_dump_json()


def test_label_studio_export_hides_sampling_operation() -> None:
    candidate = crossref_update_candidates(
        crossref_response(), response_checksum="sha256:" + "a" * 64, limit=1
    )[0]
    task = candidate.label_studio_task()
    assert "compiled_operation" not in task["data"]
    assert task["data"]["source_level_update_notice"].startswith("Document-level")


def test_pilot_split_leakage_gate_rejects_eval_documents() -> None:
    candidate = crossref_update_candidates(
        crossref_response(), response_checksum="sha256:" + "a" * 64, limit=1
    )[0]
    pool = [candidate.model_copy(update={"pair_id": f"pair:{index}"}) for index in range(300)]
    with pytest.raises(ValueError, match="forbidden eval documents"):
        validate_candidate_pool(pool, forbidden_document_ids={"doi:10.1000/original"})
