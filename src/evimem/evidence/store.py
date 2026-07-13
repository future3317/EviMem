"""Release-aware evidence access and immutable-ref validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from evimem.contracts import EvidenceRef, TextSpanLocator, evidence_ref_from_block

from .doi import normalize_doi
from .release import EvidenceReleaseManager, block_checksum


@dataclass(frozen=True)
class EvidenceSearchResult:
    block: dict[str, Any]
    evidence_ref: EvidenceRef
    score: float


class EvidenceBlockStore:
    """Read-only view over one or more immutable releases."""

    def __init__(self, manager: EvidenceReleaseManager):
        self.manager = manager

    def blocks_for_doi(
        self,
        release_id: str,
        doi: str,
        *,
        domain_name: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(self.manager.load_by_doi(release_id, doi, domain_name=domain_name))

    @staticmethod
    def ref_from_block(block: dict[str, Any], *, release_id: str, doi: str) -> EvidenceRef:
        text = str(block.get("text", block.get("content", "")) or "")
        return evidence_ref_from_block(
            block,
            release_id=release_id,
            document_id=f"doi:{normalize_doi(doi)}",
            locator=TextSpanLocator(
                block_id=str(block["block_id"]),
                start=0,
                end=len(text),
            ),
            quote=text,
        )

    def refs_for_doi(self, release_id: str, doi: str) -> tuple[EvidenceRef, ...]:
        return tuple(
            self.ref_from_block(block, release_id=release_id, doi=doi)
            for block in self.blocks_for_doi(release_id, doi)
        )

    def validate_ref(self, ref: EvidenceRef) -> dict[str, Any]:
        release = self.manager.get_release(ref.release_id)
        rows = self.manager.load_blocks(release.release_id)
        matches = rows[rows["block_id"].astype(str) == ref.block_id]
        if len(matches) != 1:
            raise ValueError("evidence ref block_id is absent or non-unique in its release")
        block = matches.iloc[0].to_dict()
        if block_checksum(block) != ref.checksum:
            raise ValueError("evidence ref checksum does not match immutable release block")
        doi = normalize_doi(str(block.get("doi", "")))
        if ref.document_id != f"doi:{doi}":
            raise ValueError("evidence ref document_id does not match release block DOI")
        text = str(block.get("text", block.get("content", "")) or "")
        if ref.quote and ref.quote not in text:
            raise ValueError("evidence quote is not present in the referenced block")
        if isinstance(ref.locator, TextSpanLocator):
            if not 0 <= ref.locator.start <= ref.locator.end <= len(text):
                raise ValueError("evidence text-span locator is outside the block")
            if ref.quote and text[ref.locator.start : ref.locator.end] != ref.quote:
                raise ValueError("evidence quote does not match its text-span locator")
        block["evidence_release_id"] = ref.release_id
        block["evidence_block_checksum"] = ref.checksum
        return block

    def search(
        self,
        *,
        release_id: str,
        doi: str,
        query: str,
        limit: int = 8,
    ) -> tuple[EvidenceSearchResult, ...]:
        tokens = tuple(
            dict.fromkeys(token.casefold() for token in re.findall(r"[\w.+/%-]+", query) if token)
        )
        results: list[EvidenceSearchResult] = []
        for block in self.blocks_for_doi(release_id, doi):
            text = str(block.get("text", block.get("content", "")) or "")
            folded = text.casefold()
            matched = sum(token in folded for token in tokens)
            score = matched / max(1, len(tokens))
            if score or not tokens:
                results.append(
                    EvidenceSearchResult(
                        block=block,
                        evidence_ref=self.ref_from_block(block, release_id=release_id, doi=doi),
                        score=score,
                    )
                )
        results.sort(key=lambda item: (-item.score, item.evidence_ref.block_id))
        return tuple(results[: max(1, limit)])

