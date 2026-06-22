"""Tests for ``sessions``: project layout, listing, and resume picker."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import json
import logging
import os
import time

import pytest

from sagent import sessions
from sagent.sessions import (
    SessionInfo,
    _peek_session,
    cwd_slug,
    existing_scope_dir,
    latest_session,
    list_all_sessions,
    list_sessions,
    new_session_dir,
    parse_jsonl,
    pick_session,
    project_dir,
    session_dir_for_scope,
)


def _write_session(
    dir_path: Path,
    *,
    session_id: str = "abc",
    model_id: str = "claude-x",
    user_text: str = "hello",
    status: str = "",
) -> Path:
    """Write a tiny v4 session.jsonl into ``dir_path``."""
    dir_path.mkdir(parents=True, exist_ok=True)
    file = dir_path / "session.jsonl"
    records = [
        {
            "kind": "meta",
            "session_id": session_id,
            "model_id": model_id,
            "status": status,
        },
        {"kind": "history", "type": "user", "text": user_text},
    ]
    file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return file


def test_cwd_slug_replaces_separators(tmp_path: Path) -> None:
    slug = cwd_slug(tmp_path)
    assert "/" not in slug
    # Slug draws from ``[A-Za-z0-9_-]``: ``_`` keeps the path-separator
    # structure injective (see ``test_cwd_slug_no_separator_collision``)
    # while still being filesystem-safe.
    assert all(c.isalnum() or c in ("-", "_") for c in slug)


def test_cwd_slug_no_separator_collision() -> None:
    """Distinct paths that differ only in ``/`` vs ``-`` must not alias.

    Pre-fix both paths slugged to ``-tmp-a-b`` because the regex
    replaced every non-alphanumeric (including ``/``) with ``-``,
    silently mixing transcripts from neighbouring directories.
    """
    assert cwd_slug("/tmp/a-b") != cwd_slug("/tmp/a/b")  # noqa: S108 -- literal paths exercise the collision invariant; no FS access.


def test_cwd_slug_is_deterministic(tmp_path: Path) -> None:
    assert cwd_slug(tmp_path) == cwd_slug(tmp_path)


def test_cwd_slug_long_path_truncates_with_stable_hash() -> None:
    """Slugs exceeding the cap are truncated + stably suffixed."""
    long_path = Path("/" + "x" * 300)
    s1 = cwd_slug(long_path)
    s2 = cwd_slug(long_path)
    assert s1 == s2
    # Suffix appended.
    assert "-" in s1
    # Truncation observed (input > 300 chars; output much shorter).
    assert len(s1) < 300


def test_project_dir_under_projects_root(tmp_path: Path) -> None:
    pdir = project_dir(tmp_path, projects_dir=tmp_path / "root")
    assert pdir.parent == tmp_path / "root"
    assert pdir.name == cwd_slug(tmp_path)


def test_new_session_dir_creates_and_returns(tmp_path: Path) -> None:
    sdir = new_session_dir(tmp_path, projects_dir=tmp_path / "root")
    assert sdir.is_dir()
    assert sdir.parent == project_dir(tmp_path, projects_dir=tmp_path / "root")


def test_session_dir_for_scope_creates_scoped_dir(tmp_path: Path) -> None:
    sdir = session_dir_for_scope("slack-T123", base=tmp_path)
    assert sdir.is_dir()
    assert sdir.parent == tmp_path / "slack-T123"


@pytest.mark.parametrize(
    "scope",
    [
        "../escape",
        "a/../b",
        "/abs",
        "",
        "..",
        "with\x00nul",
        "back\\slash",
    ],
)
def test_session_dir_for_scope_rejects_traversal(scope: str, tmp_path: Path) -> None:
    """Caller-supplied scope must not escape the projects root.

    Pre-fix, ``scope="../escape"`` would land at
    ``<projects>/../escape/<uuid>`` -- a directory outside the
    configured root that a malicious caller (slack thread id, etc.)
    could use to overwrite arbitrary files.
    """
    with pytest.raises(ValueError, match="scope"):
        session_dir_for_scope(scope, base=tmp_path)


@pytest.mark.parametrize(
    "scope",
    ["../escape", "/abs", "", "..", "with\x00nul"],
)
def test_existing_scope_dir_rejects_traversal(scope: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scope"):
        existing_scope_dir(scope, base=tmp_path)


def test_existing_scope_dir_none_when_scope_missing(tmp_path: Path) -> None:
    assert existing_scope_dir("nonexistent", base=tmp_path) is None


def test_existing_scope_dir_none_when_scope_dir_has_no_children(
    tmp_path: Path,
) -> None:
    (tmp_path / "scope-X").mkdir()
    assert existing_scope_dir("scope-X", base=tmp_path) is None


def test_existing_scope_dir_returns_most_recent(tmp_path: Path) -> None:
    """``existing_scope_dir`` selects by mtime among dirs that have a session file."""
    scope = "scope-Y"
    old = tmp_path / scope / "old"
    new = tmp_path / scope / "new"
    _write_session(old, session_id="O")
    _write_session(new, session_id="N")
    # Force ordering.
    os.utime(old, (1, 1))
    os.utime(new, (1_000_000, 1_000_000))
    selected = existing_scope_dir(scope, base=tmp_path)
    assert selected == new


def test_parse_jsonl_skips_blank_and_malformed_lines() -> None:
    text = '{"a": 1}\n\n{not json}\n{"b": 2}\n'
    records = parse_jsonl(text)
    assert records == [{"a": 1}, {"b": 2}]


def test_parse_jsonl_skips_non_dict_values() -> None:
    """Top-level JSON that isn't a dict is dropped."""
    text = '[1, 2]\n42\n{"k": "v"}\n'
    records = parse_jsonl(text)
    assert records == [{"k": "v"}]


def test_parse_jsonl_empty_string() -> None:
    assert parse_jsonl("") == []


def test_list_sessions_empty_when_project_dir_missing(tmp_path: Path) -> None:
    assert list_sessions(tmp_path, projects_dir=tmp_path / "missing") == []


def test_list_sessions_returns_newest_first(tmp_path: Path) -> None:
    """Sessions sorted by mtime descending."""
    projects = tmp_path / "projects"
    pdir = project_dir(tmp_path, projects_dir=projects)
    a = pdir / "a"
    b = pdir / "b"
    _write_session(a, session_id="A", user_text="alpha")
    _write_session(b, session_id="B", user_text="beta")
    # Force ordering: b is newer than a.
    (a / "session.jsonl").touch()
    time.sleep(0.01)
    (b / "session.jsonl").touch()
    sessions = list_sessions(tmp_path, projects_dir=projects)
    assert len(sessions) == 2
    assert sessions[0].session_id == "B"
    assert sessions[1].session_id == "A"


def test_list_sessions_skips_files_at_project_root(tmp_path: Path) -> None:
    """Stray files (not session dirs) under the project root are ignored."""
    projects = tmp_path / "projects"
    pdir = project_dir(tmp_path, projects_dir=projects)
    pdir.mkdir(parents=True)
    (pdir / "stray.txt").write_text("ignore me", encoding="utf-8")
    _write_session(pdir / "valid")
    sessions = list_sessions(tmp_path, projects_dir=projects)
    assert len(sessions) == 1


def test_list_sessions_skips_session_dir_without_jsonl(tmp_path: Path) -> None:
    """An empty session dir without ``session.jsonl`` is excluded."""
    projects = tmp_path / "projects"
    pdir = project_dir(tmp_path, projects_dir=projects)
    (pdir / "no-jsonl").mkdir(parents=True)
    _write_session(pdir / "valid")
    sessions = list_sessions(tmp_path, projects_dir=projects)
    assert len(sessions) == 1


def test_latest_session_none_when_no_sessions(tmp_path: Path) -> None:
    assert latest_session(tmp_path, projects_dir=tmp_path / "p") is None


def test_latest_session_returns_top(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    pdir = project_dir(tmp_path, projects_dir=projects)
    _write_session(pdir / "only", session_id="O", user_text="hi")
    latest = latest_session(tmp_path, projects_dir=projects)
    assert latest is not None
    assert latest.session_id == "O"


def test_list_all_sessions_returns_empty_when_root_missing(tmp_path: Path) -> None:
    assert list_all_sessions(projects_dir=tmp_path / "missing") == []


def test_list_all_sessions_aggregates_across_projects(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    _write_session(projects / "p1" / "s1", session_id="A")
    _write_session(projects / "p2" / "s2", session_id="B")
    # Stray non-dir at root + non-dir inside project: both ignored.
    (projects / "stray.txt").write_text("x", encoding="utf-8")
    (projects / "p1" / "stray.txt").write_text("x", encoding="utf-8")
    sessions = list_all_sessions(projects_dir=projects)
    assert {s.session_id for s in sessions} == {"A", "B"}


def test_session_info_extracts_meta_and_first_user_text(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    pdir = project_dir(tmp_path, projects_dir=projects)
    _write_session(
        pdir / "s",
        session_id="sid",
        model_id="claude-3-5",
        user_text="first user prompt",
        status="",
    )
    sessions = list_sessions(tmp_path, projects_dir=projects)
    assert sessions[0].session_id == "sid"
    assert sessions[0].model_id == "claude-3-5"
    assert sessions[0].status == "first user prompt"
    assert sessions[0].message_count == 1


def test_session_info_explicit_status_overrides_first_user(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    pdir = project_dir(tmp_path, projects_dir=projects)
    _write_session(
        pdir / "s",
        session_id="sid",
        status="custom status",
        user_text="some text",
    )
    sessions = list_sessions(tmp_path, projects_dir=projects)
    assert sessions[0].status == "custom status"


def test_session_info_handles_corrupt_session_jsonl(tmp_path: Path) -> None:
    """Garbled JSONL produces a SessionInfo with empty fields, not a crash."""
    projects = tmp_path / "projects"
    pdir = project_dir(tmp_path, projects_dir=projects)
    sdir = pdir / "s"
    sdir.mkdir(parents=True)
    (sdir / "session.jsonl").write_text("not json\n", encoding="utf-8")
    sessions = list_sessions(tmp_path, projects_dir=projects)
    assert len(sessions) == 1
    assert sessions[0].model_id == ""
    assert sessions[0].message_count == 0


def _info(idx: int) -> SessionInfo:
    """Build a tiny ``SessionInfo`` for picker tests."""
    return SessionInfo(
        path=Path(f"/tmp/s{idx}"),  # noqa: S108 -- in-memory placeholder
        session_id=f"sid-{idx}",
        mtime=float(idx),
        status=f"msg{idx}",
        message_count=idx,
        model_id="m",
    )


def test_pick_session_empty_returns_none() -> None:
    assert pick_session([]) is None


def test_pick_session_blank_input_picks_first() -> None:
    """Pressing Enter selects the default (most recent)."""
    sessions = [_info(1), _info(2)]
    sin = StringIO("\n")
    sout = StringIO()
    selected = pick_session(sessions, stream_in=sin, stream_out=sout)
    assert selected is sessions[0]


def test_pick_session_numeric_input_selects() -> None:
    sessions = [_info(1), _info(2), _info(3)]
    selected = pick_session(sessions, stream_in=StringIO("2\n"), stream_out=StringIO())
    assert selected is sessions[1]


def test_pick_session_out_of_range_returns_none() -> None:
    sessions = [_info(1)]
    selected = pick_session(sessions, stream_in=StringIO("99\n"), stream_out=StringIO())
    assert selected is None


def test_pick_session_non_numeric_then_eof_returns_none() -> None:
    """Non-numeric input re-prompts; EOF on the retry collapses to None."""
    sessions = [_info(1)]
    selected = pick_session(
        sessions, stream_in=StringIO("nope\n"), stream_out=StringIO()
    )
    assert selected is None


def test_pick_session_writes_a_menu(tmp_path: Path) -> None:
    """Each session renders a numbered entry."""
    del tmp_path
    sessions = [_info(1), _info(2)]
    sout = StringIO()
    _ = pick_session(sessions, stream_in=StringIO("\n"), stream_out=sout)
    out = sout.getvalue()
    assert "[ 1]" in out
    assert "[ 2]" in out


def test_pick_session_truncation_header_when_over_cap() -> None:
    """The 20-cap must surface in the prompt so users know about hidden rows."""
    sessions = [_info(i) for i in range(1, 26)]
    sout = StringIO()
    _ = pick_session(sessions, stream_in=StringIO("\n"), stream_out=sout)
    out = sout.getvalue()
    assert "20 of 25" in out


def test_pick_session_invalid_input_reprompts() -> None:
    """Non-numeric input differentiates from abort by re-prompting."""
    sessions = [_info(1), _info(2)]
    # First input is garbage; second selects 2.
    sin = StringIO("garbage\n2\n")
    sout = StringIO()
    selected = pick_session(sessions, stream_in=sin, stream_out=sout)
    assert selected is sessions[1]
    assert sout.getvalue().count("Resume which?") >= 2


def test_iter_jsonl_warns_on_malformed(caplog: pytest.LogCaptureFixture) -> None:
    """Malformed non-blank lines log a warning rather than silently dropping."""
    with caplog.at_level(logging.WARNING, logger=sessions.__name__):
        records = parse_jsonl('{"ok": 1}\n{not json}\n[1, 2]\n')
    assert records == [{"ok": 1}]
    assert sum("malformed" in r.message.lower() for r in caplog.records) >= 1
    assert sum("non-dict" in r.message.lower() for r in caplog.records) >= 1


def test_existing_scope_dir_skips_dirs_without_session_jsonl(tmp_path: Path) -> None:
    """Pre-created scope dirs missing ``session.jsonl`` must not win the max."""
    scope = "scope-Z"
    empty = tmp_path / scope / "empty"
    valid = tmp_path / scope / "valid"
    empty.mkdir(parents=True)
    _write_session(valid, session_id="V")
    # Make ``empty`` strictly newer than ``valid`` so an mtime-only max
    # would pick it; the filter is the only thing rescuing ``valid``.
    now = time.time()
    os.utime(valid, (now - 1000, now - 1000))
    os.utime(valid / "session.jsonl", (now - 1000, now - 1000))
    os.utime(empty, (now + 1000, now + 1000))
    selected = existing_scope_dir(scope, base=tmp_path)
    assert selected == valid


def test_peek_session_signals_corruption_on_mid_iteration_failure(
    tmp_path: Path,
) -> None:
    """A read failure partway through must not produce a half-counted SessionInfo.

    Either the directory is dropped (``None``) or the returned record
    carries ``corrupt=True``. The previous behaviour returned a
    SessionInfo with whatever counts had accumulated, hiding the
    corruption from callers.
    """
    projects = tmp_path / "projects"
    pdir = project_dir(tmp_path, projects_dir=projects)
    sdir = pdir / "s"
    sdir.mkdir(parents=True)
    (sdir / "session.jsonl").write_bytes(
        b'{"kind": "meta", "session_id": "S", "model_id": "m"}\n'
        b'{"kind": "history", "type": "user", "text": "ok"}\n'
        b"\xff\xfe garbage \xc3\x28\n"
    )
    info = _peek_session(sdir)
    assert info is None or info.corrupt is True


def test_latest_session_avoids_full_peek_sort(tmp_path: Path) -> None:
    """``latest_session`` peeks only the mtime winner.

    A counter wraps ``_peek_session`` and asserts at most one call --
    candidate sort is mtime-only.
    """
    projects = tmp_path / "projects"
    pdir = project_dir(tmp_path, projects_dir=projects)
    for i in range(3):
        _write_session(pdir / f"s{i}", session_id=f"S{i}")

    real_peek = sessions._peek_session
    calls = {"n": 0}

    def counting_peek(d: Path) -> SessionInfo | None:
        calls["n"] += 1
        return real_peek(d)

    with patch.object(sessions, "_peek_session", side_effect=counting_peek):
        _ = latest_session(tmp_path, projects_dir=projects)
    assert calls["n"] == 1


def _setup_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point module globals at temp claude/xdg homes; return (claude, sagent).

    ``_LEGACY_SAGENT_HOME`` is pointed at a nonexistent temp path so the
    migration takes the ``~/.claude`` squat branch (and never the host's real
    ``~/.sagent``). Tests of the real-``~/.sagent`` branch set it explicitly.
    """
    claude = tmp_path / "claude"
    sagent = tmp_path / "xdg" / "sagent"
    monkeypatch.setattr(
        sessions, "_LEGACY_SAGENT_HOME", tmp_path / "nonexistent-sagent"
    )
    monkeypatch.setattr(sessions, "_LEGACY_CLAUDE_HOME", claude)
    monkeypatch.setattr(sessions, "_SAGENT_HOME", sagent)
    monkeypatch.setattr(sessions, "_PROJECTS_DIR", sagent / "projects")
    return claude, sagent


def test_migrate_copies_sagent_sessions_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude, sagent = _setup_homes(tmp_path, monkeypatch)
    proj = claude / "projects" / "-home-u-proj"
    _write_session(proj / "deadbeef0001", session_id="S1")

    sessions.migrate_legacy_home()

    migrated = sagent / "projects" / "-home-u-proj" / "deadbeef0001" / "session.jsonl"
    assert migrated.exists()
    # Source left intact (copy, not move).
    assert (proj / "deadbeef0001" / "session.jsonl").exists()


def test_migrate_skips_claude_own_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project dir with only bare ``<uuid>.jsonl`` (Claude CLI) is not copied."""
    claude, sagent = _setup_homes(tmp_path, monkeypatch)
    proj = claude / "projects" / "-home-u-claudeonly"
    proj.mkdir(parents=True)
    (proj / "11111111-2222-3333-4444-555555555555.jsonl").write_text("{}\n")

    sessions.migrate_legacy_home()

    assert not (sagent / "projects" / "-home-u-claudeonly").exists()


def test_migrate_copies_memory_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude, sagent = _setup_homes(tmp_path, monkeypatch)
    proj = claude / "projects" / "-home-u-proj"
    _write_session(proj / "deadbeef0001", session_id="S1")
    (proj / "memory").mkdir()
    (proj / "memory" / "MEMORY.md").write_text("- note\n")

    sessions.migrate_legacy_home()

    assert (sagent / "projects" / "-home-u-proj" / "memory" / "MEMORY.md").exists()


def test_migrate_copies_papers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claude, sagent = _setup_homes(tmp_path, monkeypatch)
    (claude / "papers").mkdir(parents=True)
    (claude / "papers" / "arxiv_1.pdf").write_bytes(b"%PDF-1.4")

    sessions.migrate_legacy_home()

    assert (sagent / "papers" / "arxiv_1.pdf").read_bytes() == b"%PDF-1.4"


def test_migrate_is_idempotent_and_nondestructive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude, sagent = _setup_homes(tmp_path, monkeypatch)
    proj = claude / "projects" / "-home-u-proj"
    _write_session(proj / "deadbeef0001", session_id="S1")

    sessions.migrate_legacy_home()
    migrated = sagent / "projects" / "-home-u-proj" / "deadbeef0001" / "session.jsonl"
    sentinel = migrated.read_text()
    migrated.write_text(sentinel + "// local edit\n")

    # Second run must not clobber the already-migrated (locally edited) copy.
    sessions.migrate_legacy_home()
    assert migrated.read_text().endswith("// local edit\n")


def test_migrate_noop_without_claude_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _claude, sagent = _setup_homes(tmp_path, monkeypatch)
    sessions.migrate_legacy_home()
    assert not (sagent / "projects").exists()


def test_copy_tree_merge_does_not_follow_dir_symlink(tmp_path: Path) -> None:
    # A directory symlink (incl. a cycle) must not be followed: no recursion,
    # no fat duplication of the target. SES-001.
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("ok")
    (src / "cycle").symlink_to(src, target_is_directory=True)
    dst = tmp_path / "dst"
    sessions._copy_tree_merge(src, dst)  # must not RecursionError
    assert (dst / "a.txt").read_text() == "ok"
    # The link is preserved as a symlink, not dereferenced into a fat copy.
    assert (dst / "cycle").is_symlink()


def test_migrate_real_sagent_home_rejects_descendant_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # XDG home a child of the legacy home (e.g. XDG_DATA_HOME=~/.sagent): copying
    # legacy->child would walk the dst it just created and recurse. Must skip.
    legacy = tmp_path / "dot-sagent"
    (legacy / "projects").mkdir(parents=True)
    monkeypatch.setattr(sessions, "_LEGACY_SAGENT_HOME", legacy)
    monkeypatch.setattr(sessions, "_LEGACY_CLAUDE_HOME", tmp_path / "missing")
    monkeypatch.setattr(sessions, "_SAGENT_HOME", legacy / "xdg" / "sagent")
    monkeypatch.setattr(
        sessions, "_PROJECTS_DIR", legacy / "xdg" / "sagent" / "projects"
    )
    sessions.migrate_legacy_home()
    assert not (legacy / "xdg" / "sagent" / "xdg").exists()


def test_migrate_real_sagent_home_common_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The common pre-convention layout: a REAL ~/.sagent directory (no Claude
    # symlink). Its contents copy verbatim into the XDG home. This is the case
    # the Claude-only migration silently dropped.
    _claude, sagent = _setup_homes(tmp_path, monkeypatch)
    legacy = tmp_path / "real-dot-sagent"
    monkeypatch.setattr(sessions, "_LEGACY_SAGENT_HOME", legacy)
    _write_session(
        legacy / "projects" / "_home_u_proj" / "deadbeef0001", session_id="S1"
    )
    (legacy / "papers").mkdir(parents=True)
    (legacy / "papers" / "arxiv_1.pdf").write_bytes(b"%PDF-1.4")

    sessions.migrate_legacy_home()

    assert (
        sagent / "projects" / "_home_u_proj" / "deadbeef0001" / "session.jsonl"
    ).exists()
    assert (sagent / "papers" / "arxiv_1.pdf").read_bytes() == b"%PDF-1.4"
    # Copy, not move: legacy left intact.
    assert (legacy / "papers" / "arxiv_1.pdf").exists()


def test_migrate_merges_into_existing_projects_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The orphan bug: once a fresh session creates the XDG ``projects/`` dir, the
    # per-directory skip-if-exists made migration skip the WHOLE tree, stranding
    # every not-yet-copied project. The copy must merge into the existing dir.
    _claude, sagent = _setup_homes(tmp_path, monkeypatch)
    legacy = tmp_path / "real-dot-sagent"
    monkeypatch.setattr(sessions, "_LEGACY_SAGENT_HOME", legacy)
    _write_session(
        legacy / "projects" / "_home_u_old" / "deadbeef0001", session_id="OLD"
    )
    # Simulate a fresh session already having created the XDG projects dir.
    _write_session(
        sagent / "projects" / "_home_u_new" / "cafef00d0001", session_id="NEW"
    )

    sessions.migrate_legacy_home()

    # The pre-existing new session is untouched AND the old one is brought over.
    assert (
        sagent / "projects" / "_home_u_new" / "cafef00d0001" / "session.jsonl"
    ).exists()
    assert (
        sagent / "projects" / "_home_u_old" / "deadbeef0001" / "session.jsonl"
    ).exists()


def test_migrate_prefers_real_sagent_over_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When a real ~/.sagent exists, it is the home; the ~/.claude squat branch
    # is NOT taken (that path is only for the symlink case).
    claude, sagent = _setup_homes(tmp_path, monkeypatch)
    legacy = tmp_path / "real-dot-sagent"
    monkeypatch.setattr(sessions, "_LEGACY_SAGENT_HOME", legacy)
    _write_session(
        legacy / "projects" / "_home_u_real" / "deadbeef0001", session_id="R"
    )
    # A Claude tree also present -- must be ignored when real ~/.sagent exists.
    _write_session(
        claude / "projects" / "-home-u-claude" / "deadbeef0002", session_id="C"
    )

    sessions.migrate_legacy_home()

    assert (sagent / "projects" / "_home_u_real").exists()
    assert not (sagent / "projects" / "-home-u-claude").exists()


def test_migrate_symlinked_sagent_takes_claude_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The squat case: ~/.sagent is a SYMLINK (to ~/.claude), not a real dir, so
    # the Claude-extraction branch runs.
    claude, sagent = _setup_homes(tmp_path, monkeypatch)
    claude.mkdir(parents=True)
    link = tmp_path / "linked-dot-sagent"
    link.symlink_to(claude, target_is_directory=True)
    monkeypatch.setattr(sessions, "_LEGACY_SAGENT_HOME", link)
    _write_session(
        claude / "projects" / "-home-u-proj" / "deadbeef0001", session_id="S1"
    )

    sessions.migrate_legacy_home()

    assert (sagent / "projects" / "-home-u-proj" / "deadbeef0001").exists()


def test_bridge_skills_symlink_when_claude_has_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude, sagent = _setup_homes(tmp_path, monkeypatch)
    (claude / "skills").mkdir(parents=True)
    (claude / "skills" / "demo.md").write_text("skill\n")

    sessions.migrate_legacy_home()

    link = sagent / "skills"
    assert link.is_symlink()
    assert (link / "demo.md").read_text() == "skill\n"


def test_bridge_skills_absent_when_claude_lacks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude, sagent = _setup_homes(tmp_path, monkeypatch)
    (claude / "papers").mkdir(parents=True)  # claude exists, but no skills/
    sessions.migrate_legacy_home()
    assert not (sagent / "skills").exists()
    assert not (sagent / "skills").is_symlink()


def test_project_dir_resolves_legacy_dash_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When only a migrated ``-``-slug dir exists, project_dir returns it."""
    _claude, _sagent = _setup_homes(tmp_path, monkeypatch)
    projects = tmp_path / "projects"
    cwd = tmp_path / "work"
    cwd.mkdir()
    legacy = projects / sessions._legacy_cwd_slug(cwd)
    _write_session(legacy / "deadbeef0001", session_id="S1")

    resolved = project_dir(cwd, projects_dir=projects)
    assert resolved == legacy
    assert latest_session(cwd, projects_dir=projects) is not None


def test_new_session_writes_current_slug_even_when_legacy_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SAG-XDG-002: resume READS the legacy ``-``-slug, but a NEW session must
    # establish the current ``_``-slug -- never keep writing into the legacy dir.
    _claude, _sagent = _setup_homes(tmp_path, monkeypatch)
    projects = tmp_path / "projects"
    cwd = tmp_path / "work"
    cwd.mkdir()
    legacy = projects / sessions._legacy_cwd_slug(cwd)
    _write_session(legacy / "deadbeef0001", session_id="S1")

    created = new_session_dir(cwd, projects_dir=projects)
    assert created.parent == projects / cwd_slug(cwd)
    assert created.parent != legacy


def test_project_dir_prefers_current_slug_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _claude, _sagent = _setup_homes(tmp_path, monkeypatch)
    projects = tmp_path / "projects"
    cwd = tmp_path / "work"
    cwd.mkdir()
    current = projects / cwd_slug(cwd)
    current.mkdir(parents=True)
    assert project_dir(cwd, projects_dir=projects) == current


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
