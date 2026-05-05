"""Tests for sagent.sessions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import io
import os
import time

import pytest

from sagent import sessions


class TestSlug:
    def test_basic(self) -> None:
        # Uses the resolved path, so absolute/relative roundtrip.
        s = sessions.cwd_slug("/a/b/c")
        assert s.startswith("-")
        assert "/" not in s
        assert " " not in s

    def test_non_alnum_replaced(self) -> None:
        s = sessions.cwd_slug("/foo bar/baz-qux")
        for ch in s:
            assert ch.isalnum() or ch == "-"

    def test_stable(self) -> None:
        assert sessions.cwd_slug("/x") == sessions.cwd_slug("/x")

    def test_long_path_hashed(self) -> None:
        long_path = "/" + ("a" * 300)
        s = sessions.cwd_slug(long_path)
        assert len(s) <= sessions._MAX_SLUG_LEN + 20  # slug + hash suffix


class TestProjectDir:
    def test_under_sagent_home(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d = sessions.project_dir("/some/where", projects_dir=pdir)
        assert d.is_relative_to(pdir)


class TestSessionListing:
    def test_empty_when_no_project_dir(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        assert sessions.list_sessions("/nonexistent", projects_dir=pdir) == []
        assert sessions.latest_session("/nonexistent", projects_dir=pdir) is None

    def test_new_session_dir_creates(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d = sessions.new_session_dir("/some/cwd", projects_dir=pdir)
        assert d.exists()
        assert d.is_dir()

    def test_peek_session_parses_meta_and_messages(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d = sessions.new_session_dir("/x", projects_dir=pdir)
        (d / "session.jsonl").write_text(
            '{"kind":"meta","session_id":"abcd","model_id":"m"}\n'
            '{"kind":"message","descriptor":"text/x-user-message","content":"hello world"}\n'
            '{"kind":"message","descriptor":"multipart/x-model-message","content":[]}\n',
        )
        info = sessions._peek_session(d)
        assert info is not None
        assert info.session_id == "abcd"
        assert info.model_id == "m"
        assert info.message_count == 2
        assert info.status == "hello world"

    def test_peek_session_descriptor_user_message(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d = sessions.new_session_dir("/x", projects_dir=pdir)
        (d / "session.jsonl").write_text(
            '{"kind":"meta","session_id":"abcd","model_id":"m"}\n'
            '{"kind":"message","descriptor":"text/x-user-message","content":"hello from descriptor"}\n'
            '{"kind":"message","descriptor":"multipart/x-model-message","content":[]}\n',
        )
        info = sessions._peek_session(d)
        assert info is not None
        assert info.message_count == 2
        assert info.status == "hello from descriptor"

    def test_peek_session_prefers_meta_status(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d = sessions.new_session_dir("/x", projects_dir=pdir)
        (d / "session.jsonl").write_text(
            '{"kind":"meta","session_id":"a","status":"Refactoring tests"}\n'
            '{"kind":"message","descriptor":"text/x-user-message","content":"original prompt"}\n',
        )
        info = sessions._peek_session(d)
        assert info is not None
        assert info.status == "Refactoring tests"

    def test_list_sorted_newest_first(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d1 = sessions.new_session_dir("/x", projects_dir=pdir)
        d2 = sessions.new_session_dir("/x", projects_dir=pdir)
        f1 = d1 / "session.jsonl"
        f2 = d2 / "session.jsonl"
        f1.write_text('{"kind":"meta"}\n')
        f2.write_text('{"kind":"meta"}\n')
        # Make d2 newer.
        os.utime(f1, (1000.0, 1000.0))
        os.utime(f2, (2000.0, 2000.0))
        out = sessions.list_sessions("/x", projects_dir=pdir)
        assert len(out) == 2
        assert out[0].path == d2


class TestSessionDirForScope:
    def test_creates_dir_under_base(self, tmp_path: Path) -> None:
        d = sessions.session_dir_for_scope("my-scope", base=tmp_path)
        assert d.exists()
        assert d.parent.name == "my-scope"
        assert d.parent.parent == tmp_path

    def test_base_param_uses_given_dir(self, tmp_path: Path) -> None:
        d = sessions.session_dir_for_scope("s1", base=tmp_path)
        assert d.parent == tmp_path / "s1"


class TestExistingScopeDir:
    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert sessions.existing_scope_dir("nope", base=tmp_path) is None

    def test_returns_none_when_no_children(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        assert sessions.existing_scope_dir("empty", base=tmp_path) is None

    def test_returns_most_recent(self, tmp_path: Path) -> None:
        scope = tmp_path / "sc"
        old = scope / "old"
        new = scope / "new"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        os.utime(old, (1000.0, 1000.0))
        os.utime(new, (2000.0, 2000.0))
        assert sessions.existing_scope_dir("sc", base=tmp_path) == new


class TestIterJsonl:
    def test_skips_malformed_and_blank(self) -> None:
        lines = ["", '{"a":1}', "NOT JSON", '{"b":2}', "  "]
        result = list(sessions._iter_jsonl(lines))
        assert result == [{"a": 1}, {"b": 2}]


class TestPeekSessionEdgeCases:
    def test_missing_session_file(self, tmp_path: Path) -> None:
        d = tmp_path / "no-file"
        d.mkdir()
        assert sessions._peek_session(d) is None

    def test_stat_oserror(self, tmp_path: Path) -> None:
        d = tmp_path / "s"
        d.mkdir()
        f = d / "session.jsonl"
        f.write_text("")
        original_stat = Path.stat

        def _stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
            nonlocal calls
            if self == f:
                calls += 1
                if calls > 1:
                    raise OSError("boom")
            return original_stat(self, follow_symlinks=follow_symlinks)

        calls = 0
        with patch.object(Path, "stat", _stat):
            assert sessions._peek_session(d) is None

    def test_open_oserror(self, tmp_path: Path) -> None:
        d = tmp_path / "s"
        d.mkdir()
        f = d / "session.jsonl"
        f.write_text('{"kind":"meta"}\n')
        f.chmod(0o000)
        try:
            assert sessions._peek_session(d) is None
        finally:
            f.chmod(0o644)

    def test_meta_and_message_parsing(self, tmp_path: Path) -> None:
        d = tmp_path / "s"
        d.mkdir()
        (d / "session.jsonl").write_text(
            '{"kind":"meta","model_id":"gpt","session_id":"sid","title":"T"}\n'
            '{"kind":"message","descriptor":"multipart/x-model-message","content":[]}\n'
            '{"kind":"message","descriptor":"text/x-user-message","content":"hi"}\n'
        )
        info = sessions._peek_session(d)
        assert info is not None
        assert info.model_id == "gpt"
        assert info.session_id == "sid"
        assert info.status == "T"
        assert info.message_count == 2


class TestListSessionsNonDir:
    def test_skips_non_dir_children(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d = sessions.new_session_dir("/x", projects_dir=pdir)
        (d / "session.jsonl").write_text('{"kind":"meta"}\n')
        # Place a plain file sibling next to the session dir.
        (d.parent / "stray.txt").write_text("junk")
        out = sessions.list_sessions("/x", projects_dir=pdir)
        assert len(out) == 1


class TestFormatRelativeTime:
    def test_hours(self) -> None:
        ts = time.time() - 7200
        assert sessions._format_relative_time(ts) == "2h ago"

    def test_days(self) -> None:
        ts = time.time() - 172_800
        assert sessions._format_relative_time(ts) == "2d ago"


class TestPicker:
    def test_empty_returns_none(self) -> None:
        assert sessions.pick_session([]) is None

    def test_default_pick_first(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d = sessions.new_session_dir("/x", projects_dir=pdir)
        (d / "session.jsonl").write_text('{"kind":"meta","session_id":"a"}\n')
        all_ = sessions.list_sessions("/x", projects_dir=pdir)
        assert all_
        out = io.StringIO()
        choice = sessions.pick_session(
            all_, stream_in=io.StringIO("\n"), stream_out=out
        )
        assert choice is all_[0]
        assert "Resume which?" in out.getvalue()

    def test_index_choice(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d1 = sessions.new_session_dir("/x", projects_dir=pdir)
        d2 = sessions.new_session_dir("/x", projects_dir=pdir)
        (d1 / "session.jsonl").write_text('{"kind":"meta","session_id":"a"}\n')
        (d2 / "session.jsonl").write_text('{"kind":"meta","session_id":"b"}\n')
        all_ = sessions.list_sessions("/x", projects_dir=pdir)
        assert len(all_) == 2
        choice = sessions.pick_session(
            all_,
            stream_in=io.StringIO("2\n"),
            stream_out=io.StringIO(),
        )
        assert choice is all_[1]

    def test_out_of_range_returns_none(self) -> None:
        s = sessions.SessionInfo(
            path=Path("/x"),
            session_id="a",
            mtime=0.0,
            status="",
            message_count=0,
            model_id="",
        )
        choice = sessions.pick_session(
            [s], stream_in=io.StringIO("99\n"), stream_out=io.StringIO()
        )
        assert choice is None

    def test_non_integer_input_returns_none(self) -> None:
        s = sessions.SessionInfo(
            path=Path("/x"),
            session_id="a",
            mtime=0.0,
            status="",
            message_count=0,
            model_id="",
        )
        choice = sessions.pick_session(
            [s], stream_in=io.StringIO("abc\n"), stream_out=io.StringIO()
        )
        assert choice is None

    def test_eof_returns_none(self) -> None:
        s = sessions.SessionInfo(
            path=Path("/x"),
            session_id="a",
            mtime=0.0,
            status="",
            message_count=0,
            model_id="",
        )
        sin = io.StringIO("")  # readline returns "" on EOF
        choice = sessions.pick_session([s], stream_in=sin, stream_out=io.StringIO())
        # Empty readline → default to sessions[0]
        assert choice is s


class TestListAllSessions:
    def test_scans_all_projects(self, tmp_path: Path) -> None:
        pdir = tmp_path / "projects"
        d1 = sessions.new_session_dir("/proj-a", projects_dir=pdir)
        d2 = sessions.new_session_dir("/proj-b", projects_dir=pdir)
        f1 = d1 / "session.jsonl"
        f2 = d2 / "session.jsonl"
        f1.write_text('{"kind":"meta","session_id":"s1"}\n')
        f2.write_text('{"kind":"meta","session_id":"s2"}\n')
        os.utime(f1, (1000.0, 1000.0))
        os.utime(f2, (2000.0, 2000.0))
        out = sessions.list_all_sessions(projects_dir=pdir)
        assert len(out) == 2
        assert out[0].session_id == "s2"  # newest first

    def test_empty_when_no_projects_dir(self, tmp_path: Path) -> None:
        assert sessions.list_all_sessions(projects_dir=tmp_path / "nonexistent") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
