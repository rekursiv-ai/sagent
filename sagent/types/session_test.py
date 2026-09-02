"""Tests for ``types.session``: the id/dir binding."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.types.session import Session


def test_at_names_the_session_for_its_directory(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "abc123"
    session = Session.at(str(runs))
    assert session.id == "abc123"
    assert session.dir == runs


def test_at_accepts_a_path(tmp_path: Path) -> None:
    assert Session.at(tmp_path / "abc123").id == "abc123"


def test_ephemeral_has_an_id_but_no_directory() -> None:
    session = Session.ephemeral()
    assert session.id
    assert session.dir is None


def test_ephemeral_ids_are_distinct() -> None:
    assert Session.ephemeral().id != Session.ephemeral().id


def test_a_directory_that_disagrees_with_its_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="named for its session"):
        _ = Session(id="abc123", dir=tmp_path / "other")


def test_an_empty_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _ = Session(id="")


def test_an_unpersisted_session_needs_no_directory_agreement() -> None:
    assert Session(id="abc123").dir is None


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
