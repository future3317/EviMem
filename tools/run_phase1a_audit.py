"""Generate Phase 1A reports from pinned upstream checkouts outside the repository.

This script performs no training and does not copy source datasets into the repo.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evimem.benchmark import (
    DatasetRegistry,
    EvidenceInferenceAdapter,
    MeasEvalAdapter,
    QasperAdapter,
    RejectedConversion,
    SciFactAdapter,
    SciRexAdapter,
    ViewSample,
    sha256_file,
    source_snapshot,
    write_reports,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rejection(dataset: str, row_id: str, split: str, reason: Exception, path: Path):
    return RejectedConversion(
        dataset_name=dataset,
        source_row_id=row_id,
        split=split,
        reason=f"{type(reason).__name__}: {reason}",
        source_checksum=sha256_file(path),
    )


def load_scirex(root: Path) -> tuple[list[ViewSample], list[RejectedConversion], list[Path]]:
    samples: list[ViewSample] = []
    rejected: list[RejectedConversion] = []
    files: list[Path] = []
    adapter = SciRexAdapter()
    release = root / "release_data"
    if release.exists():
        for split in ("train", "dev", "test"):
            path = release / f"{split}.jsonl"
            files.append(path)
            for row in (
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ):
                try:
                    samples.extend(adapter.convert_views(row, split=split))
                except (TypeError, ValueError, KeyError, IndexError) as exc:
                    rejected.append(
                        _rejection(adapter.dataset_name, str(row.get("doc_id")), split, exc, path)
                    )
    else:
        for split in ("train", "dev", "test"):
            for path in sorted((root / "docs" / f"{split}_docs").glob("*.json")):
                files.append(path)
                row = _json(path)
                try:
                    samples.extend(adapter.convert_views(row, split=split))
                except (TypeError, ValueError, KeyError, IndexError) as exc:
                    rejected.append(_rejection(adapter.dataset_name, path.stem, split, exc, path))
        files.append(root / "LICENSE")
    return samples, rejected, files


def load_qasper(root: Path) -> tuple[list[ViewSample], list[RejectedConversion], list[Path]]:
    samples: list[ViewSample] = []
    rejected: list[RejectedConversion] = []
    files: list[Path] = []
    adapter = QasperAdapter()
    for split in ("train", "dev", "test"):
        path = root / f"qasper-{split}-v0.3.json"
        files.append(path)
        for document_id, article in _json(path).items():
            for qa_index, qa in enumerate(article.get("qas", ())):
                for answer_index, annotation in enumerate(qa.get("answers", ())):
                    isolated_qa = {**qa, "answers": [annotation]}
                    row = {"document_id": document_id, **article, "qas": [isolated_qa]}
                    try:
                        samples.extend(adapter.convert_views(row, split=split))
                    except (TypeError, ValueError, KeyError, IndexError) as exc:
                        annotation_id = annotation.get("annotation_id", answer_index)
                        row_id = f"{document_id}:{qa.get('question_id', qa_index)}:{annotation_id}"
                        rejected.append(
                            _rejection(adapter.dataset_name, row_id, split, exc, path)
                        )
    files.extend(path for path in (root / "train-dev.tgz", root / "test.tgz") if path.exists())
    return samples, rejected, files


def load_scifact(root: Path) -> tuple[list[ViewSample], list[RejectedConversion], list[Path]]:
    corpus_path = root / "data" / "corpus.jsonl"
    corpus = {
        str(item["doc_id"]): item
        for item in (json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines())
    }
    samples: list[ViewSample] = []
    rejected: list[RejectedConversion] = []
    files = [corpus_path]
    adapter = SciFactAdapter()
    for split in ("train", "dev", "test"):
        path = root / "data" / f"claims_{split}.jsonl"
        files.append(path)
        for claim in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()):
            for document_id, evidence_sets in claim.get("evidence", {}).items():
                source = corpus.get(str(document_id))
                if source is None:
                    rejected.append(
                        _rejection(
                            adapter.dataset_name,
                            f"{claim['id']}:{document_id}",
                            split,
                            ValueError("evidence document missing from corpus"),
                            path,
                        )
                    )
                    continue
                for evidence_index, annotation in enumerate(evidence_sets):
                    row = {
                        "id": f"{claim['id']}:{evidence_index}",
                        "document_id": document_id,
                        "claim": claim["claim"],
                        "label": annotation["label"],
                        "abstract_sentences": source["abstract"],
                        "evidence_sentence_ids": annotation["sentences"],
                    }
                    try:
                        samples.extend(adapter.convert_views(row, split=split))
                    except (TypeError, ValueError, KeyError, IndexError) as exc:
                        rejected.append(
                            _rejection(
                                adapter.dataset_name,
                                f"{claim['id']}:{document_id}:{evidence_index}",
                                split,
                                exc,
                                path,
                            )
                        )
    return samples, rejected, files


def load_measeval(root: Path) -> tuple[list[ViewSample], list[RejectedConversion], list[Path]]:
    samples: list[ViewSample] = []
    rejected: list[RejectedConversion] = []
    files: list[Path] = []
    adapter = MeasEvalAdapter()
    for split in ("train", "trial", "eval"):
        for path in sorted((root / "data" / split / "tsv").glob("*.tsv")):
            text_path = root / "data" / split / "text" / f"{path.stem}.txt"
            if not text_path.exists():
                text_path = root / "data" / split / "txt" / f"{path.stem}.txt"
            files.extend((path, text_path))
            source_text = text_path.read_text(encoding="utf-8")
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    groups[row["annotSet"]].append(row)
            for annotation_set, annotation_rows in groups.items():
                converted: dict[str, Any] = {
                    "id": annotation_set,
                    "document_id": path.stem,
                    "text": source_text,
                }
                mapping = {
                    "MeasuredEntity": "measured_entity",
                    "MeasuredProperty": "measured_property",
                    "Quantity": "quantity",
                }
                for annotation in annotation_rows:
                    field = mapping.get(annotation["annotType"])
                    if field and field not in converted:
                        converted[field] = annotation["text"]
                        converted[f"{field}_start"] = int(annotation["startOffset"])
                        if field == "quantity" and annotation.get("other"):
                            try:
                                converted["unit"] = json.loads(annotation["other"]).get("unit")
                            except json.JSONDecodeError:
                                pass
                try:
                    samples.extend(adapter.convert_views(converted, split=split))
                except (TypeError, ValueError, KeyError, IndexError) as exc:
                    rejected.append(
                        _rejection(
                            adapter.dataset_name,
                            f"{path.stem}:{annotation_set}",
                            split,
                            exc,
                            path,
                        )
                    )
    return samples, rejected, files


def load_evidence_fixture(
    root: Path,
) -> tuple[list[ViewSample], list[RejectedConversion], list[Path]]:
    path = root / "evidence_inference.json"
    rows = _json(path)
    adapter = EvidenceInferenceAdapter()
    samples: list[ViewSample] = []
    rejected: list[RejectedConversion] = []
    for row in rows:
        split = row.pop("split")
        try:
            samples.extend(adapter.convert_views(row, split=split))
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            rejected.append(_rejection(adapter.dataset_name, str(row.get("id")), split, exc, path))
    return samples, rejected, [path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scirex", type=Path)
    parser.add_argument("--qasper", type=Path)
    parser.add_argument("--scifact", type=Path)
    parser.add_argument("--measeval", type=Path)
    parser.add_argument("--evidence-fixture", type=Path)
    args = parser.parse_args()
    registry = DatasetRegistry.load(args.manifest)
    jobs = (
        ("SciREX", "scirex", load_scirex),
        ("QASPER", "qasper", load_qasper),
        ("SciFact", "scifact", load_scifact),
        ("Evidence Inference 2.0", "evidence_fixture", load_evidence_fixture),
        ("MeasEval", "measeval", load_measeval),
    )
    for dataset_name, argument, loader in jobs:
        root = getattr(args, argument)
        if root is None:
            continue
        samples, rejected, files = loader(root)
        write_reports(
            args.output / dataset_name.lower().replace(" ", "_").replace("2.0", "2_0"),
            spec=registry.get(dataset_name),
            samples=samples,
            rejected=rejected,
            snapshot=source_snapshot(files),
        )
        print(f"{dataset_name}: {len(samples)} samples, {len(rejected)} rejected")


if __name__ == "__main__":
    main()
