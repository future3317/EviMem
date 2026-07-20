from __future__ import annotations

import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "split_jarvis_mp_oracle_vault.py")
)


def test_oracle_partition_is_exact_and_disjoint() -> None:
    calibration, evaluation = MODULE["_partition"](
        [
            {"pair_id": "a", "split": "calibration"},
            {"pair_id": "b", "split": "evaluation"},
        ]
    )
    assert [row["pair_id"] for row in calibration] == ["a"]
    assert [row["pair_id"] for row in evaluation] == ["b"]


def test_oracle_partition_rejects_unregistered_split_and_duplicates() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        MODULE["_partition"]([{"pair_id": "a", "split": "development"}])
    with pytest.raises(ValueError, match="unique"):
        MODULE["_partition"](
            [
                {"pair_id": "a", "split": "calibration"},
                {"pair_id": "a", "split": "evaluation"},
            ]
        )
