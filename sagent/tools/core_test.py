"""Tests for sagent.tools.core."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

import asyncio
import threading
import time

import pytest

from sagent import tools
from sagent.custom_types import (
    JsonMessage,
    MessageBase,
    MultipartMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools import core
from sagent.tools.core import (
    _MISSING_TOOL_DESCRIPTION,
    TOOL_RESULT_MAX_CHARS,
    ToolState,
    _build_schema,
    _type_to_schema,
    changed_files_context,
    get_tool_state,
    has_been_read,
    load_tool_description,
    mark_read,
    opt_str,
    read_asset,
    run_sync,
    tool,
    tool_state_context,
    truncate,
)


# -- _type_to_schema ---------------------------------------------------


class TestTypeToSchema:
    def test_str(self) -> None:
        assert _type_to_schema(str) == {"type": "string"}

    def test_int(self) -> None:
        assert _type_to_schema(int) == {"type": "integer"}

    def test_float(self) -> None:
        assert _type_to_schema(float) == {"type": "number"}

    def test_bool(self) -> None:
        assert _type_to_schema(bool) == {"type": "boolean"}

    def test_list_of_str(self) -> None:
        assert _type_to_schema(list[str]) == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_bare_list_falls_back_to_string(self) -> None:
        # ``list`` (unparameterized) has no get_origin → string fallback.
        # ``list[X]`` is the supported parameterized form.
        assert _type_to_schema(list) == {"type": "string"}

    def test_dict(self) -> None:
        assert _type_to_schema(dict[str, int]) == {"type": "object"}

    def test_unknown(self) -> None:
        # Unknown types fall back to string.
        assert _type_to_schema(complex) == {"type": "string"}


# -- _build_schema -----------------------------------------------------


class TestBuildSchema:
    def test_basic(self) -> None:
        def fn(query: str, limit: int = 10) -> str:
            del query, limit
            return ""

        s = _build_schema(fn, {"query": str, "limit": int})
        assert s == {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def test_annotated_description(self) -> None:
        def fn(q: Annotated[str, "the query"]) -> str:
            del q
            return ""

        s = cast(dict[str, Any], _build_schema(fn, {"q": Annotated[str, "the query"]}))
        assert s["properties"]["q"] == {
            "type": "string",
            "description": "the query",
        }

    def test_no_required_omits_field(self) -> None:
        def fn(x: int = 0) -> str:
            del x
            return ""

        s = cast("dict[str, JSON]", _build_schema(fn, {"x": int}))
        assert "required" not in s

    def test_self_skipped(self) -> None:
        class C:
            def m(self, x: str) -> str:
                del x
                return ""

        s = cast("dict[str, dict[str, Any]]", _build_schema(C.m, {"x": str}))
        assert "self" not in s["properties"]


# -- truncate ----------------------------------------------------------


class TestTruncate:
    def test_short_unchanged(self) -> None:
        assert truncate("hi", 100) == "hi"

    def test_long_truncated_with_notice(self) -> None:
        out = truncate("a" * 50, 10)
        assert out.startswith("a" * 10)
        assert "truncated" in out
        assert "40 chars omitted" in out

    def test_exactly_at_limit(self) -> None:
        assert truncate("a" * 10, 10) == "a" * 10


# -- Recipe assets -----------------------------------------------------


def test_default_recipe_loads_tool_descriptions() -> None:
    assert load_tool_description("agentself")


def test_all_builtin_tools_have_description_assets() -> None:
    recipe = core.recipe_dict("tool_descriptions")
    recipe_paths = {k.lower(): Path(v) for k, v in recipe.items()}
    missing_recipe: list[str] = []
    missing_file: list[str] = []

    for name in tools.__all__:
        obj = getattr(tools, name, None)
        tool_id = getattr(obj, "tool_id", None)
        if not isinstance(obj, type) or not isinstance(tool_id, str):
            continue
        if not tool_id.startswith("application/x-tool-"):
            continue
        recipe_path = recipe_paths.get(name.lower())
        if recipe_path is None:
            missing_recipe.append(name)
            continue
        if not (core._ASSETS_DIR / recipe_path).is_file():
            missing_file.append(f"{name}: {recipe_path}")

    assert missing_recipe == []
    assert missing_file == []


def test_missing_tool_description_asset_soft_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing tool description files return a generic prompt fragment."""
    monkeypatch.setattr(
        core,
        "_recipe_cache",
        {"tool_descriptions": {"MissingTool": "default/does_not_exist.md"}},
    )

    with caplog.at_level("ERROR", logger="sagent.tools.core"):
        assert load_tool_description("MissingTool") == _MISSING_TOOL_DESCRIPTION

    assert "MissingTool" in caplog.text
    assert "default/does_not_exist.md" in caplog.text


def test_missing_tool_description_include_soft_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing includes inside tool descriptions use the same soft-fail path."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tool.md").write_text("before {{include: missing.md}} after")
    monkeypatch.setattr(core, "_ASSETS_DIR", assets)
    monkeypatch.setattr(
        core,
        "_recipe_cache",
        {"tool_descriptions": {"MissingInclude": "tool.md"}},
    )

    with caplog.at_level("ERROR", logger="sagent.tools.core"):
        assert load_tool_description("MissingInclude") == _MISSING_TOOL_DESCRIPTION

    assert "MissingInclude" in caplog.text
    assert "tool.md" in caplog.text


def test_read_asset_remains_strict_for_missing_files(tmp_path: Path) -> None:
    """Direct asset reads still raise for missing files."""
    with pytest.raises(FileNotFoundError):
        read_asset(tmp_path / "missing.md")


def test_read_asset_remains_strict_for_missing_includes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Direct asset reads still raise for missing include targets."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tool.md").write_text("{{include: missing.md}}")
    monkeypatch.setattr(core, "_ASSETS_DIR", assets)

    with pytest.raises(FileNotFoundError):
        read_asset("tool.md")


# -- run_sync ---------------------------------------------------------


class TestRunSync:
    @pytest.mark.anyio
    async def test_returns_str_in_response(self) -> None:
        def fn(x: int) -> str:
            return f"got {x}"

        r = await run_sync(fn, x=42)
        assert isinstance(r, MessageBase)
        assert r.descriptor == "text/plain"
        assert r.content == "got 42"

    @pytest.mark.anyio
    async def test_truncates_large_output(self) -> None:
        def fn() -> str:
            return "x" * (TOOL_RESULT_MAX_CHARS + 100)

        r = await run_sync(fn)
        assert isinstance(r, MessageBase)
        assert "truncated" in str(r.content)
        assert len(str(r.content)) < TOOL_RESULT_MAX_CHARS + 200

    @pytest.mark.anyio
    async def test_exception_propagates(self) -> None:
        def fn() -> str:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await run_sync(fn)


# -- @tool decorator --------------------------------------------------


class TestToolDecorator:
    def test_bare_decorator(self) -> None:
        @tool
        def my_fn(x: str) -> str:
            return x.upper()

        assert my_fn.name == "my_fn"
        assert my_fn.directive_schema["required"] == ("x",)
        assert my_fn.directive_schema["additionalProperties"] is False

    def test_with_args(self) -> None:
        @tool(name="Custom", description="desc")
        def f(x: str) -> str:
            return x

        assert f.name == "Custom"
        assert f.description == "desc"

    def test_direct_call_with_args(self) -> None:
        def f(x: str) -> str:
            return x

        schema: JSON = json_freeze({"type": "object", "required": ["x"]})
        wrapped = tool(
            f,
            name="Custom",
            description="desc",
            schema=schema,
            supports_microcompaction=True,
        )

        assert wrapped.name == "Custom"
        assert wrapped.description == "desc"
        assert wrapped.directive_schema == schema
        assert wrapped.supports_microcompaction is True

    def test_default_description_from_docstring(self) -> None:
        @tool
        def f(x: str) -> str:
            """My docstring."""
            return x

        assert f.description == "My docstring."

    @pytest.mark.anyio
    async def test_async_fn_called_directly(self) -> None:
        @tool
        async def f(x: str) -> str:
            return x.upper()

        msg = MultipartMessage(
            (JsonMessage(json_freeze({"x": "hi"}), "application/x-tool-f"),),
            "multipart/x-tool-call",
        )
        r = await f.run(msg)
        assert isinstance(r, MessageBase)
        assert r.content == "HI"

    @pytest.mark.anyio
    async def test_sync_fn_threaded(self) -> None:
        @tool
        def f(x: str) -> str:
            return x.lower()

        msg = MultipartMessage(
            (JsonMessage(json_freeze({"x": "HI"}), "application/x-tool-f"),),
            "multipart/x-tool-call",
        )
        r = await f.run(msg)
        assert isinstance(r, MessageBase)
        assert r.content == "hi"

    @pytest.mark.anyio
    async def test_exception_propagates(self) -> None:
        @tool
        def f() -> str:
            raise RuntimeError("oops")

        msg = MultipartMessage(
            (JsonMessage(json_freeze({}), "application/x-tool-f"),),
            "multipart/x-tool-call",
        )
        with pytest.raises(RuntimeError, match="oops"):
            await f.run(msg)

    @pytest.mark.anyio
    async def test_truncates_at_max_result_chars(self) -> None:
        @tool(max_result_chars=20)
        def f() -> str:
            return "y" * 100

        msg = MultipartMessage(
            (JsonMessage(json_freeze({}), "application/x-tool-f"),),
            "multipart/x-tool-call",
        )
        r = await f.run(msg)
        assert isinstance(r, MessageBase)
        assert "truncated" in str(r.content)


# -- ToolState --------------------------------------------------------


class TestToolState:
    def test_mark_and_check_read(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hello")
        s = ToolState()
        assert not s.has_been_read(str(f))
        s.mark_read(str(f), offset=1, limit=100)
        assert s.has_been_read(str(f))

    def test_check_unchanged_true_when_same_mtime(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        s = ToolState()
        s.mark_read(str(f), offset=1, limit=10)
        assert s.check_unchanged(str(f), offset=1, limit=10)

    def test_check_unchanged_false_after_modify(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        s = ToolState()
        s.mark_read(str(f), offset=1, limit=10)
        time.sleep(0.01)  # ensure mtime delta
        f.write_text("bye")
        assert not s.check_unchanged(str(f), offset=1, limit=10)

    def test_check_unchanged_false_on_different_window(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        s = ToolState()
        s.mark_read(str(f), offset=1, limit=10)
        # Same file, different read window → not the same read.
        assert not s.check_unchanged(str(f), offset=2, limit=10)

    def test_check_stale_after_external_modify(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        s = ToolState()
        s.mark_read(str(f), content="hi")
        time.sleep(0.01)
        f.write_text("changed")
        assert s.check_stale(str(f))

    def test_check_stale_false_when_unread(self, tmp_path: Path) -> None:
        s = ToolState()
        # File was never read → can't be stale.
        assert not s.check_stale(str(tmp_path / "missing.txt"))

    def test_check_stale_false_when_content_unchanged_despite_mtime_bump(
        self,
        tmp_path: Path,
    ) -> None:
        """Idempotent reformatter (ruff no-op) / cloud sync touch.

        mtime bumped, content identical → content-equality fallback
        treats as not-stale so Edit doesn't spuriously fail.
        """
        f = tmp_path / "x.txt"
        f.write_text("hi")
        s = ToolState()
        s.mark_read(str(f), content="hi")
        time.sleep(0.01)
        # Rewrite with identical content - bumps mtime, same bytes.
        f.write_text("hi")
        assert not s.check_stale(str(f))

    def test_check_stale_true_when_content_cache_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """Conservatively stale path: mark_read without content (binary/PDF)."""
        f = tmp_path / "x.bin"
        f.write_text("hi")
        s = ToolState()
        # mark_read without content argument (binary/PDF/image path).
        s.mark_read(str(f))
        time.sleep(0.01)
        f.write_text("hi")  # same content, bumped mtime
        # No cached content to compare against → conservatively stale.
        assert s.check_stale(str(f))

    def test_check_stale_refreshes_mtime_on_content_match(
        self,
        tmp_path: Path,
    ) -> None:
        """After a successful content-equality match, ``check_stale``
        must refresh the cached mtime so subsequent calls hit the
        O(1) mtime-match fast path instead of re-reading the file
        forever.
        """
        f = tmp_path / "x.txt"
        f.write_text("hi")
        s = ToolState()
        s.mark_read(str(f), content="hi")
        time.sleep(0.01)
        f.write_text("hi")  # mtime bumped, content identical
        # First call does the content-equality comparison and refreshes.
        assert not s.check_stale(str(f))
        # Cached mtime now equals disk mtime; remove the content cache
        # so we can prove the fast-path check returns False on mtime
        # alone (without falling back to content equality).
        s._content_cache.pop(str(f.resolve()), None)
        assert not s.check_stale(str(f))

    def test_mark_written_clears_read_window(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("v1")
        s = ToolState()
        s.mark_read(str(f), offset=5, limit=20)
        s.mark_written(str(f))
        # check_unchanged with the original window should now fail
        # - mark_written clears offset/limit to 0,0 to force re-read.
        assert not s.check_unchanged(str(f), offset=5, limit=20)

    def test_recent_files_ordering(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("a")
        b.write_text("b")
        s = ToolState()
        s.mark_read(str(a))
        s.mark_read(str(b))
        s.mark_read(str(a))  # re-read a → moves to most recent
        assert s.recent_files == [str(b), str(a)]

    def test_enforce_read_blocks_unread(self, tmp_path: Path) -> None:
        s = ToolState()
        err = s.enforce_read(str(tmp_path / "missing.txt"))
        assert err is not None
        assert "not yet read" in err

    def test_enforce_read_passes_after_mark(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        s = ToolState()
        s.mark_read(str(f))
        assert s.enforce_read(str(f)) is None

    def test_consume_changed_files_returns_diffs(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("line1\nline2\n")
        s = ToolState()
        s.mark_read(str(f), content="line1\nline2\n")
        time.sleep(0.01)
        f.write_text("line1\nLINE2\n")
        changes = s.consume_changed_files()
        assert str(f) in changes
        assert "-line2" in changes[str(f)]
        assert "+LINE2" in changes[str(f)]

    def test_consume_changed_files_empty_when_unchanged(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        s = ToolState()
        s.mark_read(str(f), content="hi")
        assert s.consume_changed_files() == {}

    def test_consume_changed_files_preserves_read_shape(self, tmp_path: Path) -> None:
        """After change-reporting, original read window must survive.

        Guards against a past bug where ``consume_changed_files``
        reset the cache shape to ``(0, 0, 0)``, causing a subsequent
        Read with the original params to mis-classify as "different
        params, must re-fetch".
        """
        f = tmp_path / "x.txt"
        f.write_text("v1\n")
        s = ToolState()
        s.mark_read(str(f), offset=5, limit=100, last_lines=0, content="v1\n")
        time.sleep(0.01)
        f.write_text("v2\n")
        _ = s.consume_changed_files()
        # Same params after consumption should dedup against current mtime.
        assert s.check_unchanged(str(f), offset=5, limit=100, last_lines=0)

    def test_abort_event_present(self) -> None:
        s = ToolState()
        assert isinstance(s.abort_event, threading.Event)
        assert not s.abort_event.is_set()


# -- get/set_tool_state context var -----------------------------------


class TestToolStateContextVar:
    def test_default_state_returned_outside_context(self) -> None:
        s = get_tool_state()
        assert isinstance(s, ToolState)

    def test_set_state_visible_to_get(self) -> None:
        s = ToolState()
        with tool_state_context(s):
            assert get_tool_state() is s

    def test_mark_read_helper_uses_current_state(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        s = ToolState()
        with tool_state_context(s):
            mark_read(str(f))
            assert has_been_read(str(f))
            assert s.has_been_read(str(f))

    def test_context_restores_prior_state(self) -> None:
        outer = ToolState()
        inner = ToolState()
        with tool_state_context(outer):
            assert get_tool_state() is outer
            with tool_state_context(inner):
                assert get_tool_state() is inner
            assert get_tool_state() is outer


# -- changed_files_context ---------------------------------------------


class TestChangedFilesContext:
    def test_empty_when_no_changes(self) -> None:
        with tool_state_context(ToolState()):
            assert changed_files_context() == ""

    def test_emits_system_reminder_with_diff(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("v1\n")
        s = ToolState()
        s.mark_read(str(f), content="v1\n")
        time.sleep(0.01)
        f.write_text("v2\n")
        with tool_state_context(s):
            ctx = changed_files_context()
        assert "<system-reminder>" in ctx
        assert "</system-reminder>" in ctx
        assert str(f) in ctx
        assert "-v1" in ctx
        assert "+v2" in ctx

    def test_clears_after_emit(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("v1\n")
        s = ToolState()
        s.mark_read(str(f), content="v1\n")
        time.sleep(0.01)
        f.write_text("v2\n")
        with tool_state_context(s):
            first = changed_files_context()
            second = changed_files_context()
        assert first  # first call sees the change
        assert second == ""  # mtime cache updated, no longer "changed"


# -- Constants exposed as public API -----------------------------------


def test_constants_exposed() -> None:
    # These are imported by other modules; protect the public surface.
    assert TOOL_RESULT_MAX_CHARS > 0


def test_run_sync_async_compatible() -> None:
    # Spot check that asyncio.run can drive run_sync from a sync test.
    def fn() -> str:
        return "ok"

    r = asyncio.run(run_sync(fn))
    assert isinstance(r, MessageBase)
    assert r.content == "ok"


class TestOptStr:
    def test_present_value(self) -> None:
        assert opt_str({"k": "hello"}, "k") == "hello"

    def test_absent_key(self) -> None:
        assert opt_str({}, "k") is None

    def test_none_value(self) -> None:
        assert opt_str({"k": None}, "k") is None

    def test_empty_string_treated_as_absent(self) -> None:
        assert opt_str({"k": ""}, "k") is None
