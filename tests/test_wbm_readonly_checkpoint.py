from __future__ import annotations

from pathlib import Path

from tools.create_wbm_readonly_checkpoint import _current_code_tree


def test_code_tree_digest_changes_with_uncommitted_source(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "method.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first, records = _current_code_tree(tmp_path)
    assert [item["path"] for item in records] == ["src/method.py"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second, _ = _current_code_tree(tmp_path)
    assert first != second
