from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_summary_tool():
    path = Path(__file__).parents[1] / "tools" / "summarize_matpes_confirmatory.py"
    spec = importlib.util.spec_from_file_location("summarize_matpes_confirmatory", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_sign_flip_uses_the_complete_randomization_distribution() -> None:
    tool = _load_summary_tool()
    assert tool._exact_sign_flip_two_sided(np.asarray([1.0, 1.0, 1.0])) == 0.25
    assert tool._exact_sign_flip_two_sided(np.asarray([1.0, -1.0])) == 1.0


def test_paired_summary_keeps_exact_system_as_the_unit() -> None:
    tool = _load_summary_tool()
    summary = tool._paired_summary(
        np.asarray([1.0, 0.0, -1.0, 2.0]),
        bootstrap_seed=7,
        bootstrap_replicates=100,
    )
    assert summary["system_count"] == 4
    assert summary["wins"] == 2
    assert summary["ties"] == 1
    assert summary["losses"] == 1
    assert summary["sign_flip_method"] == "exact_complete_enumeration"


def test_paired_summary_uses_deterministic_monte_carlo_for_continuous_metrics() -> None:
    tool = _load_summary_tool()
    values = np.asarray([0.125, -0.25, 0.375, -0.0625])
    first = tool._paired_summary(
        values,
        bootstrap_seed=19,
        bootstrap_replicates=1_000,
    )
    second = tool._paired_summary(
        values,
        bootstrap_seed=19,
        bootstrap_replicates=1_000,
    )
    assert first["sign_flip_method"] == "deterministic_monte_carlo"
    assert first["two_sided_sign_flip_p"] == second["two_sided_sign_flip_p"]
