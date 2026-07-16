from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def test_package_has_no_legacy_project_imports() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "evimem"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src"):
                violations.append(f"{path.relative_to(source_root)}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src"):
                        violations.append(f"{path.relative_to(source_root)}:{node.lineno}")
    assert violations == []


def test_package_has_no_hardcoded_legacy_repository_paths() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "evimem"
    forbidden = (
        "piepaper",
        "src.evipgce",
        "src/evipgce",
        "src\\evipgce",
        "e:\\code\\piepaper",
        "e:/code/piepaper",
    )
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        lowered = path.read_text(encoding="utf-8").lower()
        if any(value in lowered for value in forbidden):
            violations.append(str(path.relative_to(source_root)))
    assert violations == []


def test_runtime_data_directories_are_not_tracked_source() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ("dataset", "artifacts", "outputs", "checkpoints", "runs", "wandb")
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", *forbidden],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert tracked == []


def test_learning_components_cannot_import_publication_writers() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "evimem"
    violations: list[str] = []
    for component in ("training", "benchmark"):
        for path in (package_root / component).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
                elif isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                if any("publication" in module or module == "sqlite3" for module in modules):
                    violations.append(f"{path.relative_to(package_root)}:{node.lineno}")
    assert violations == []
