from __future__ import annotations

import pytest
from sqlalchemy import event

from evimem.publication import PublicationCommitService, PublicationStore

from .evimem_helpers import certificate


def test_publication_transaction_rolls_back_certificate_on_observation_failure(tmp_path) -> None:
    store = PublicationStore(tmp_path / "publication.sqlite")
    service = PublicationCommitService(store, policy_version="phase0-test")

    def fail_observation_insert(conn, cursor, statement, parameters, context, executemany):
        if "INSERT INTO published_observations" in statement:
            raise RuntimeError("injected observation failure")

    event.listen(store._engine, "before_cursor_execute", fail_observation_insert)
    try:
        with pytest.raises(RuntimeError, match="injected observation failure"):
            service.commit(
                doi="10.1000/example",
                domain_name="piezoelectric",
                certificate=certificate(),
            )
    finally:
        event.remove(store._engine, "before_cursor_execute", fail_observation_insert)

    assert store.count_observations() == 0
    assert store.count_certificates() == 0
    assert store.count_commits() == 0
