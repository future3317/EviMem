from __future__ import annotations

import bz2
import json
import runpy
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

LOADER_MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "audit_wbm_p1_p15.py")
)


def _raw_entry() -> dict[str, object]:
    structure = Structure(Lattice.cubic(3.0), ["Li"], [[0.0, 0.0, 0.0]])
    return ComputedStructureEntry(
        structure,
        -1.0,
        parameters={"is_hubbard": False},
    ).as_dict()


def test_exact_system_loader_streams_entries_without_json_load(
    tmp_path: Path, monkeypatch
) -> None:
    """The loader must not materialize a full WBM step as one Python list."""

    for step in range(1, 6):
        with bz2.open(tmp_path / f"step_{step}.json.bz2", "wt", encoding="utf-8") as handle:
            json.dump({"entries": [_raw_entry()]}, handle)

    loader = LOADER_MODULE["_load_exact_system_universe"]
    monkeypatch.setitem(loader.__globals__, "STEP_COUNTS", (1, 1, 1, 1, 1))

    def _forbidden_json_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("streaming WBM loader must not call json.load")

    monkeypatch.setattr(json, "load", _forbidden_json_load)
    loaded = loader(
        tmp_path,
        {"wbm-1-1"},
        {("Li",)},
    )

    assert list(loaded) == [("Li",)]
    assert [str(entry.entry_id) for entry in loaded[("Li",)]] == ["wbm-1-1"]
    assert loaded[("Li",)][0].parameters["run_type"] == "GGA"
