from __future__ import annotations

import bz2
import json
from pathlib import Path

import pytest

from evimem.matmem import (
    WBMRawOracleVault,
    raw_wbm_records_from_payload,
    validate_raw_wbm_release,
)


def _write_step(path: Path, count: int) -> None:
    payload = {
        "entries": [
            {
                "composition": {"Li": 1, "O": 1},
                "structure": {"lattice": {"a": index + 1}, "sites": []},
                "energy": -float(index),
            }
            for index in range(count)
        ]
    }
    path.write_bytes(bz2.compress(json.dumps(payload).encode("utf-8")))


def test_raw_release_validation_decodes_all_steps_and_fails_on_wrong_count(tmp_path: Path) -> None:
    counts = (1, 2, 3, 4, 5)
    for step, count in enumerate(counts, start=1):
        _write_step(tmp_path / f"step_{step}.json.bz2", count)
    report = validate_raw_wbm_release(tmp_path, expected_counts=counts)
    assert report.entry_counts == counts
    assert report.raw_entry_total == sum(counts)
    assert len(report.file_checksums) == 5

    with pytest.raises(ValueError, match="count mismatch"):
        validate_raw_wbm_release(tmp_path, expected_counts=(1, 2, 99, 4, 5))


def test_raw_oracle_isolation_never_places_energy_in_observable() -> None:
    records = raw_wbm_records_from_payload(
        {
            "entries": [
                {
                    "composition": {"Li": 2, "O": 1},
                    "structure": {"lattice": {"a": 3}, "sites": []},
                    "energy": -7.5,
                }
            ]
        },
        step=2,
    )
    observable, _ = records[0]
    assert "energy" not in observable.model_dump_json()
    assert observable.source_record_locator == "raw-wbm-step-2-index-0"

    vault = WBMRawOracleVault(records)
    assert vault.observable(observable.source_record_locator) == observable
    outcome = vault.reveal(observable.source_record_locator)
    assert outcome.total_energy_ev == -7.5
    with pytest.raises(ValueError, match="already been revealed"):
        vault.reveal(observable.source_record_locator)


def test_raw_payload_rejects_nonfinite_oracle_energy() -> None:
    with pytest.raises(ValueError, match="finite numeric total energy"):
        raw_wbm_records_from_payload(
            {
                "entries": [
                    {
                        "composition": {"Li": 1},
                        "structure": {"lattice": {}, "sites": []},
                        "energy": float("nan"),
                    }
                ]
            },
            step=1,
        )
