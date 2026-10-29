"""Tests for where sessions live on disk and how a listing finds them."""

from __future__ import annotations

from pathlib import Path

import logging
import os
import uuid

from treekle import json_dumps

import pytest

from sagent.lib.userdirs import data_dir
from sagent.session import projects
from sagent.session.projects import (
    child_journal,
    cwd_slug,
    find_sessions,
    latest_session,
    list_sessions,
    new_session_path,
    project_dir,
    scope_dir,
)
from sagent.session.types import (
    Attachment,
    ContextClear,
    SessionRecord,
    TurnContext,
    UserMessage,
    record_classes,
)


def _journal(
    directory: Path,
    *records: SessionRecord,
    mtime: float = 1_000.0,
) -> Path:
    """Write ``records`` as the journal of the session ``directory``."""
    path = directory / "session.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = record_classes()
    path.write_text(
        "".join(json_dumps(r, classes=classes, sort=False) + "\n" for r in records),
    )
    os.utime(path, (mtime, mtime))
    return path


# ---- cwd_slug --------------------------------------------------------------


def test_a_slug_turns_separators_into_underscores() -> None:
    """Not under ``/home``: the slug resolves symlinks, and some hosts link ``/home``."""
    assert cwd_slug(Path("/Me/Proj9")) == "_Me_Proj9"


def test_a_slug_escapes_every_other_character_as_hex() -> None:
    assert cwd_slug(Path("/a-b")) == "_a-2d-b"
    assert cwd_slug(Path("/a_b")) == "_a-5f-b"
    assert cwd_slug(Path("/a.b")) == "_a-2e-b"
    assert cwd_slug(Path("/é")) == "_-c3a9-"


def test_distinct_directories_never_share_a_slug() -> None:
    names = ["/a_b", "/a-b", "/a.b", "/a/b", "/a b"]
    assert len({cwd_slug(Path(n)) for n in names}) == len(names)


def test_a_slug_names_the_resolved_directory(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert cwd_slug(tmp_path / "link") == cwd_slug(tmp_path / "real")
    assert cwd_slug(tmp_path / "real" / "..") == cwd_slug(tmp_path)


def test_a_long_slug_is_cut_and_ends_in_a_stable_hash() -> None:
    long = Path("/" + "x" * 300)
    slug = cwd_slug(long)
    head, _, digest = slug.rpartition("-")
    assert head == ("_" + "x" * 300)[:200]
    assert digest == "cfa5c7ebdcf3307e"  # FNV-1a 64 of the path.
    assert cwd_slug(long) == slug
    assert cwd_slug(Path("/" + "x" * 301)) != slug


def test_a_slug_at_the_limit_is_kept_whole() -> None:
    assert cwd_slug(Path("/abcd"), max_len=5) == "_abcd"
    assert cwd_slug(Path("/abcde"), max_len=5).startswith("_abcd-")


# ---- directories -----------------------------------------------------------


def test_a_project_lives_under_the_per_user_root() -> None:
    root = data_dir() / "rekursiv-ai" / "sagent" / "projects"
    assert project_dir(Path("/w")) == root / "_w"


def test_a_project_root_may_be_given(tmp_path: Path) -> None:
    assert project_dir(Path("/w"), root=tmp_path) == tmp_path / "_w"


def test_a_scope_nests_under_the_root(tmp_path: Path) -> None:
    root = data_dir() / "rekursiv-ai" / "sagent" / "projects"
    assert scope_dir("slack/T123") == root / "slack" / "T123"
    assert scope_dir("t", root=tmp_path) == tmp_path / "t"


@pytest.mark.parametrize(
    "scope",
    ["", "/abs", "a/../b", "..", "./a", "a//b", "a/", "a\\b", "C:/x", "a\x00b"],
)
def test_a_scope_that_could_leave_the_root_is_rejected(scope: str) -> None:
    with pytest.raises(ValueError, match="Not a relative scope"):
        scope_dir(scope)


def test_a_scope_may_hold_a_colon_past_its_first_part(tmp_path: Path) -> None:
    assert scope_dir("slack/T:1", root=tmp_path) == tmp_path / "slack" / "T:1"


def test_a_child_is_a_session_inside_its_parents_whatever_its_name() -> None:
    assert child_journal(Path("/p/abc/journal.jsonl"), "helper") == Path(
        "/p/abc/helper/session.jsonl",
    )


def test_a_child_is_a_session_inside_its_parents() -> None:
    assert child_journal(Path("/p/abc/session.jsonl"), "helper") == Path(
        "/p/abc/helper/session.jsonl",
    )


# ---- new_session_path ------------------------------------------------------


def test_a_new_session_is_a_private_directory_holding_an_empty_journal(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "projects" / "here"
    path = new_session_path(directory)
    assert (path.name, path.parent.parent) == ("session.jsonl", directory)
    assert len(path.parent.name) == 12
    int(path.parent.name, 16)
    assert path.read_bytes() == b""
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert directory.stat().st_mode & 0o777 == 0o700
    assert new_session_path(directory) != path


def test_a_colliding_id_is_replaced_by_a_fresh_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An old sagent's session holds the id too, as ``<id>/session.jsonl``."""
    taken = uuid.UUID(int=1 << 120)
    fresh = uuid.UUID(int=2 << 120)
    old = tmp_path / taken.hex[:12] / "session.jsonl"
    old.parent.mkdir()
    old.write_text("live")
    minted = iter([taken, fresh])
    monkeypatch.setattr(uuid, "uuid4", lambda: next(minted))
    with caplog.at_level(logging.WARNING, logger=projects.__name__):
        path = new_session_path(tmp_path)
    assert path.parent.name == fresh.hex[:12]
    assert old.read_text() == "live"
    assert caplog.messages == [
        f"session id {taken.hex[:12]} already exists; minting another.",
    ]


# ---- list_sessions ---------------------------------------------------------


def test_a_missing_directory_lists_nothing(tmp_path: Path) -> None:
    assert list_sessions(tmp_path / "absent") == []
    assert latest_session(tmp_path / "absent") is None


def test_sessions_list_newest_first(tmp_path: Path) -> None:
    old = _journal(tmp_path / "000000000001", UserMessage(content="a"), mtime=1_000)
    new = _journal(tmp_path / "000000000002", UserMessage(content="b"), mtime=2_000)
    assert [s.path for s in list_sessions(tmp_path)] == [new, old]
    latest = latest_session(tmp_path)
    assert latest is not None
    assert latest.path == new
    assert latest.mtime == 2_000


def test_a_listing_reads_title_model_and_record_count(tmp_path: Path) -> None:
    _journal(
        tmp_path / "abc123000000",
        ContextClear(),
        TurnContext(model="first"),
        UserMessage(content="   "),
        UserMessage(content="  fix\n  the   bug "),
        UserMessage(content="later"),
        TurnContext(model="last"),
        TurnContext(cwd="/w"),
    )
    (info,) = list_sessions(tmp_path)
    assert info.id == "abc123000000"
    assert info.title == "fix the bug"
    assert info.model == "last"
    assert info.records == 7


def test_a_session_quit_before_anything_was_said_is_not_listed(tmp_path: Path) -> None:
    """A launch makes its journal at once; quitting at the prompt leaves it empty."""
    _journal(tmp_path / "000000000001", ContextClear())
    _journal(tmp_path / "000000000002")
    _journal(
        tmp_path / "000000000003",
        ContextClear(),
        UserMessage(content="hi"),
        mtime=1,
    )
    assert [s.id for s in list_sessions(tmp_path)] == ["000000000003"]
    assert [s.id for s in find_sessions("", root=tmp_path)] == ["000000000003"]
    latest = latest_session(tmp_path)
    assert latest is not None
    assert latest.id == "000000000003"


def test_a_listing_parses_no_window_s_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snapshot holds a whole window's history; one journal held 318 of them."""
    snapshot = ContextClear(
        history=(UserMessage(content="old"), TurnContext(model="older")),
    )
    _journal(
        tmp_path / "000000000001",
        snapshot,
        TurnContext(model="m"),
        UserMessage(content="a"),
        snapshot,
        UserMessage(content="b"),
    )
    parsed: list[bytes] = []
    record = projects._record

    def counting(line: bytes, *, classes: dict[str, type]) -> SessionRecord | None:
        parsed.append(line)
        return record(line, classes=classes)

    monkeypatch.setattr(projects, "_record", counting)
    (info,) = list_sessions(tmp_path)
    assert (info.title, info.model, info.records) == ("a", "m", 5)
    assert not [line for line in parsed if b'"ContextClear"' in line[:40]]


def test_a_session_of_only_attachments_is_listed_without_a_title(
    tmp_path: Path,
) -> None:
    png = Attachment(mime_descriptor="image/png", data=b"png")
    _journal(tmp_path / "000000000001", ContextClear(), UserMessage(attachments=(png,)))
    (info,) = list_sessions(tmp_path)
    assert (info.title, info.model, info.records) == ("", "", 2)


def test_unreadable_lines_do_not_hide_the_rest(tmp_path: Path) -> None:
    path = _journal(tmp_path / "000000000001", UserMessage(content="real"))
    path.write_text('{"py/object": "UserMessage", "content": \n\n' + path.read_text())
    (info,) = list_sessions(tmp_path)
    assert info.title == "real"
    assert info.records == 2


def test_only_the_sessions_of_the_directory_are_listed(tmp_path: Path) -> None:
    session = "000000000001"
    _journal(tmp_path / session, UserMessage(content="a"))
    _journal(tmp_path / session / "000000000002", UserMessage(content="c"))
    _journal(tmp_path / "helper", UserMessage(content="c"))
    (tmp_path / "flat.jsonl").write_text("")
    (tmp_path / "memory").mkdir()
    (tmp_path / "000000000003").mkdir()
    (tmp_path / "000000000003" / "session.jsonl").write_text("{}\n")
    (tmp_path / "notes.txt").write_text("x")
    assert [s.id for s in list_sessions(tmp_path)] == [session]


def test_a_project_holding_a_log_of_its_own_still_lists_its_sessions(
    tmp_path: Path,
) -> None:
    """An old sagent once wrote a session into the project directory itself."""
    project = tmp_path / "_home_me"
    project.mkdir()
    (project / "session.jsonl").write_text("{}\n")
    path = _journal(project / "1a15f1ee9d88", UserMessage(content="a"))
    assert [s.path for s in list_sessions(project)] == [path]
    assert [s.path for s in find_sessions("1a15", root=tmp_path)] == [path]


@pytest.mark.parametrize(
    "name",
    [
        "1a15f1ee9d88",
        "1a15f1ee9d881a15f1ee9d881a15f1ee",
    ],
    ids=["twelve", "thirty-two"],
)
def test_a_session_is_a_directory_named_by_its_id(tmp_path: Path, name: str) -> None:
    assert projects.is_session(tmp_path / name / "session.jsonl")
    assert projects.is_session(tmp_path / name / "session.jsonl")


@pytest.mark.parametrize(
    "parts",
    [
        ("_home_me", "session.jsonl"),
        ("1a15f1ee9d88", "helper", "session.jsonl"),
        ("1a15f1ee9d88", "a8f9e83b-e405-40f0-a621-74ed9fb3d351", "session.jsonl"),
        ("1a15f1ee9d88", "19b33bab", "072cc7f2-6cfe-4d6e-b2c5", "session.jsonl"),
        ("1A15F1EE9D88", "session.jsonl"),
        ("1a15f1ee9d8", "session.jsonl"),
    ],
    ids=["project", "label", "old-child", "old-nested-child", "upper", "short"],
)
def test_an_agent_a_session_spawned_is_no_session(
    tmp_path: Path,
    parts: tuple[str, ...],
) -> None:
    assert not projects.is_session(tmp_path.joinpath(*parts))


def test_a_spawned_agent_named_like_an_id_is_still_no_session(tmp_path: Path) -> None:
    parent = _journal(tmp_path / "1a15f1ee9d88", UserMessage(content="a"))
    assert not projects.is_session(parent.parent / "000000000001" / "session.jsonl")


def test_the_limit_counts_only_readable_sessions(tmp_path: Path) -> None:
    for i in range(3):
        _journal(
            tmp_path / f"00000000000{i}",
            UserMessage(content=str(i)),
            mtime=1_000 + i,
        )
    (tmp_path / "00000000000a" / "session.jsonl").mkdir(parents=True)
    os.utime(tmp_path / "00000000000a" / "session.jsonl", (9_000, 9_000))
    (tmp_path / "00000000000b").mkdir()
    (tmp_path / "00000000000b" / "session.jsonl").symlink_to(tmp_path / "gone")
    assert [s.id for s in list_sessions(tmp_path, limit=2)] == [
        "000000000002",
        "000000000001",
    ]
    assert [s.id for s in list_sessions(tmp_path)] == [
        "000000000002",
        "000000000001",
        "000000000000",
    ]


def test_a_journal_that_cannot_be_read_hides_no_other(tmp_path: Path) -> None:
    (tmp_path / "gone").mkdir()
    dangling = tmp_path / "gone" / "session.jsonl"
    dangling.symlink_to(tmp_path / "nowhere")
    kept = _journal(tmp_path / "000000000001", UserMessage(content="a"))
    assert [s.path for s in projects._infos([dangling, kept], limit=None)] == [kept]


def test_a_limited_listing_reads_no_journal_past_the_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(4):
        _journal(
            tmp_path / f"00000000000{i}",
            UserMessage(content=str(i)),
            mtime=1_000 + i,
        )
    peeked: list[str] = []
    peek = projects._peek

    def counting(path: Path, *, mtime: float) -> projects.SessionInfo | None:
        peeked.append(path.parent.name)
        return peek(path, mtime=mtime)

    monkeypatch.setattr(projects, "_peek", counting)
    latest = latest_session(tmp_path)
    assert latest is not None
    assert latest.id == "000000000003"
    assert peeked == ["000000000003"]
    peeked.clear()
    list_sessions(tmp_path, limit=2)
    assert peeked == ["000000000003", "000000000002"]


# ---- find_sessions ---------------------------------------------------------


def test_find_matches_a_prefix_in_every_project_and_scope(tmp_path: Path) -> None:
    a = _journal(tmp_path / "_p1" / "ab1200000000", UserMessage(content="a"), mtime=1)
    b = _journal(
        tmp_path / "slack" / "T1" / "ab3400000000",
        UserMessage(content="b"),
        mtime=2,
    )
    _journal(tmp_path / "_p2" / "cd5600000000", UserMessage(content="c"))
    assert [s.path for s in find_sessions("ab", root=tmp_path)] == [b, a]
    assert [s.path for s in find_sessions("ab3", root=tmp_path)] == [b]
    assert len(find_sessions("", root=tmp_path)) == 3


def test_find_skips_the_journals_of_spawned_agents(tmp_path: Path) -> None:
    parent = "ab0000000000"
    _journal(tmp_path / "_p" / parent, UserMessage(content="a"))
    _journal(tmp_path / "_p" / parent / "ab_0", UserMessage(content="c"))
    _journal(tmp_path / "_p" / parent / "ab0000000001", UserMessage(content="c"))
    assert [s.id for s in find_sessions("ab", root=tmp_path)] == [parent]


def test_find_reads_a_prefix_literally(tmp_path: Path) -> None:
    _journal(tmp_path / "_p" / "ab0000000000", UserMessage(content="b"))
    assert find_sessions("[a]b", root=tmp_path) == []
    assert find_sessions("*", root=tmp_path) == []
    assert find_sessions("ab?", root=tmp_path) == []
    assert len(find_sessions("ab0", root=tmp_path)) == 1


def test_find_reads_no_journal_past_its_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every project's journals match ``""``; reading them all took 52 s."""
    for i in range(50):
        _journal(
            tmp_path / f"_p{i % 5}" / f"0000000000{i:02}",
            UserMessage(content="x"),
            mtime=1_000 + i,
        )
    peeked: list[str] = []
    peek = projects._peek

    def counting(path: Path, *, mtime: float) -> projects.SessionInfo | None:
        peeked.append(path.parent.name)
        return peek(path, mtime=mtime)

    monkeypatch.setattr(projects, "_peek", counting)
    assert [s.id for s in find_sessions("", root=tmp_path, limit=1)] == [
        "000000000049",
    ]
    assert peeked == ["000000000049"]
    peeked.clear()
    assert [s.id for s in find_sessions("0000000000", root=tmp_path, limit=3)] == [
        "000000000049",
        "000000000048",
        "000000000047",
    ]
    assert peeked == ["000000000049", "000000000048", "000000000047"]


def _renamed(path: Path) -> Path:
    """Move the session file at ``path`` to the name sessions were once written under."""
    return path.rename(path.with_name("journal.jsonl"))


def test_a_journal_named_as_before_is_renamed_when_listed(tmp_path: Path) -> None:
    path = _journal(tmp_path / "000000000001", UserMessage(content="a"))
    _renamed(path)
    child = _renamed(_journal(path.parent / "helper", UserMessage(content="c")))

    assert [s.path for s in list_sessions(tmp_path)] == [path]
    assert not path.with_name("journal.jsonl").exists()
    assert child.exists()


def test_a_journal_named_as_before_is_renamed_when_found(tmp_path: Path) -> None:
    path = _journal(tmp_path / "_p" / "ab0000000000", UserMessage(content="a"))
    _renamed(path)

    assert [s.path for s in find_sessions("ab", root=tmp_path)] == [path]
    assert not path.with_name("journal.jsonl").exists()


def test_a_journal_named_as_before_replaces_the_old_log_it_converted(
    tmp_path: Path,
) -> None:
    """The journal holds the old log converted, and whatever was said since."""
    log = tmp_path / "000000000001" / "session.jsonl"
    log.parent.mkdir()
    log.write_text('{"kind": "meta"}\n')
    journal = _renamed(_journal(tmp_path / "000000000002", UserMessage(content="a")))
    journal = journal.rename(log.with_name("journal.jsonl"))

    assert projects.session_file(log.parent) == log
    assert log.read_text().startswith('{"py/object": "UserMessage"')
    assert not journal.exists()
    assert projects.legacy_log(log.parent).read_text() == '{"kind": "meta"}\n'


def test_a_child_named_as_before_is_still_no_session(tmp_path: Path) -> None:
    parent = _renamed(_journal(tmp_path / "1a15f1ee9d88", UserMessage(content="a")))
    assert not projects.is_session(parent.parent / "000000000001" / "session.jsonl")


def test_find_defaults_to_the_per_user_root() -> None:
    root = data_dir() / "rekursiv-ai" / "sagent" / "projects"
    path = _journal(root / "_w" / "feed00000000", UserMessage(content="a"))
    assert [s.path for s in find_sessions("fe")] == [path]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
