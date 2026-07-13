from __future__ import annotations

from evimem.core import EvidenceReleaseManager
from evimem.core.contracts import evidence_ref_from_block


def test_release_is_checksumed_and_produces_canonical_refs(tmp_path) -> None:
    manager = EvidenceReleaseManager(tmp_path / "evidence")
    manager.create_release(
        [
            {
                "doi": "10.1000/example",
                "source": "fixture",
                "block_id": "block-1",
                "text": "PZT has d33 = 350 pC/N.",
            }
        ],
        release_id="release-1",
    )

    manager.set_current_release("release-1")
    assert manager.get_current_release().release_id == "release-1"
    assert manager.verify_release("release-1")["ok"] is True

    blocks = manager.load_by_doi("release-1", "https://doi.org/10.1000/example")
    assert len(blocks) == 1
    assert blocks[0]["evidence_release_id"] == "release-1"
    assert blocks[0]["evidence_block_checksum"].startswith("sha256:")

    ref = evidence_ref_from_block(
        blocks[0],
        release_id="release-1",
        document_id="doi:10.1000/example",
    )
    assert ref.release_id == "release-1"
    assert ref.block_id == "block-1"
    assert ref.checksum == blocks[0]["evidence_block_checksum"]


def test_release_ids_cannot_escape_release_root(tmp_path) -> None:
    manager = EvidenceReleaseManager(tmp_path / "evidence")
    try:
        manager.create_release(
            [{"doi": "10.1000/x", "source": "fixture"}],
            release_id="../escape",
        )
    except ValueError as exc:
        assert "path separators" in str(exc)
    else:
        raise AssertionError("unsafe release ID was accepted")

