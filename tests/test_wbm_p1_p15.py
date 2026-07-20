from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "audit_wbm_p1_p15.py")
)


def _soap_payload(ids: list[str]) -> dict[str, object]:
    return {
        "pool_by_id": dict.fromkeys(ids, "pool"),
        "cache_sha256": "sha256:x",
        "structure_stage": "initial",
        "causal_available_before_query": True,
        "structure_source_field": "org",
    }


def _pool() -> dict[str, object]:
    pools = {}
    for pool_index in range(8):
        name = f"A{pool_index}-B{pool_index}"
        pools[name] = {
            "chemical_system": [f"A{pool_index}", f"B{pool_index}"],
            "candidates": [
                {
                    "query_id": f"q-{pool_index}-{item}",
                    "exact_structure_sha256": f"sha256:org-{pool_index}-{item}",
                }
                for item in range(16)
            ],
        }
    return {"selection": {"pools": pools}}


def _parity(pool: dict[str, object]) -> dict[str, object]:
    rows = []
    for pool_name, item in pool["selection"]["pools"].items():
        for candidate in item["candidates"]:
            rows.append(
                {
                    "query_id": candidate["query_id"],
                    "exact_chemsys": pool_name,
                    "canonical_structure_id": (
                        "byte-identical:" + candidate["exact_structure_sha256"]
                    ),
                    "modern_corrected_formation_energy_ev_per_atom": -1.0,
                    "parity_corrected_formation_energy_ev_per_atom": -1.0,
                    "initial_e_above_hull_modern_ev_per_atom": 0.0,
                    "initial_e_above_hull_parity_ev_per_atom": 0.0,
                    "stable_label_modern": True,
                    "stable_label_parity": True,
                    "phase_membership_modern": True,
                    "phase_membership_parity": True,
                }
            )
    return {"rows": rows}


def test_engineering_p1_accepts_historical_replay_but_not_claim_grade() -> None:
    pool = _pool()
    parity = _parity(pool)
    ids = [row["query_id"] for row in parity["rows"]]
    vectors = np.zeros((128, 2))
    vectors[:, 0] = 1.0
    report = MODULE["validate_engineering_p1"](
        pool_payload=pool,
        parity_payload=parity,
        soap_payload=_soap_payload(ids),
        soap_query_ids=ids,
        soap_vectors=vectors,
        soap_cache_sha256="sha256:x",
    )
    assert report["engineering_p1_passed"] is True
    assert report["frozen_parity_replay_passed"] is True
    assert report["official_energy_reproduction_claim_permitted"] is False
    assert report["claim_grade_identity_passed"] is False
    assert report["soap_structure_stage"] == "initial"


def test_engineering_p1_fails_on_label_or_soap_mismatch() -> None:
    pool = _pool()
    parity = _parity(pool)
    ids = [row["query_id"] for row in parity["rows"]]
    parity["rows"][0]["stable_label_modern"] = False
    vectors = np.zeros((128, 2))
    vectors[:, 0] = 1.0
    report = MODULE["validate_engineering_p1"](
        pool_payload=pool,
        parity_payload=parity,
        soap_payload=_soap_payload(ids),
        soap_query_ids=ids,
        soap_vectors=vectors,
        soap_cache_sha256="sha256:x",
    )
    assert report["engineering_p1_passed"] is False
    assert report["frozen_parity_replay_passed"] is True
    with pytest.raises(ValueError, match="unit-normalized"):
        MODULE["validate_engineering_p1"](
            pool_payload=pool,
            parity_payload=_parity(pool),
            soap_payload=_soap_payload(ids),
            soap_query_ids=ids,
            soap_vectors=np.zeros((128, 2)),
            soap_cache_sha256="sha256:x",
        )


def test_engineering_p1_rejects_relaxed_structure_soap() -> None:
    pool = _pool()
    parity = _parity(pool)
    ids = [row["query_id"] for row in parity["rows"]]
    vectors = np.zeros((128, 2))
    vectors[:, 0] = 1.0
    payload = _soap_payload(ids)
    payload["structure_stage"] = "relaxed"
    with pytest.raises(ValueError, match="initial structures"):
        MODULE["validate_engineering_p1"](
            pool_payload=pool,
            parity_payload=parity,
            soap_payload=payload,
            soap_query_ids=ids,
            soap_vectors=vectors,
            soap_cache_sha256="sha256:x",
        )


def test_engineering_p1_rejects_mislabeled_initial_structure_field() -> None:
    pool = _pool()
    parity = _parity(pool)
    ids = [row["query_id"] for row in parity["rows"]]
    vectors = np.zeros((128, 2))
    vectors[:, 0] = 1.0
    payload = _soap_payload(ids)
    payload["structure_source_field"] = "opt"
    with pytest.raises(ValueError, match="org initial-structure field"):
        MODULE["validate_engineering_p1"](
            pool_payload=pool,
            parity_payload=parity,
            soap_payload=payload,
            soap_query_ids=ids,
            soap_vectors=vectors,
            soap_cache_sha256="sha256:x",
        )
