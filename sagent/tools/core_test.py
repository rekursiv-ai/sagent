"""Tests for ``tools.core``: tool framework, state, sandbox."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, cast

import asyncio
import os
import time

import pytest

from sagent.agent.runtime import ToolResult
from sagent.testing import with_fake_agent
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    ReadCacheEntry,
    ToolState,
    _ToolImpl,
    changed_files_context,
    get_file_write_lock,
    get_tool_state,
    has_been_read,
    load_tool_description,
    mark_read,
    opt_int,
    opt_str,
    read_asset,
    recipe_dict,
    recipe_list,
    resolve_recipe,
    resolve_tool_path,
    run_sync,
    set_recipe,
    to_result,
    tool,
    tool_state_context,
    truncate,
)


def _schema_property(schema: object, name: str) -> Mapping[str, object]:
    """Return ``schema['properties'][name]`` as a typed Mapping."""
    schema_m: Mapping[str, object] = _as_mapping(schema)
    props = _as_mapping(schema_m["properties"])
    return _as_mapping(props[name])


def _as_mapping(value: object) -> Mapping[str, object]:
    """Narrow an object to ``Mapping[str, object]`` or fail.

    ``isinstance(v, Mapping)`` narrows to ``Mapping[Unknown, object]``
    in ty/basedpyright; cast at this single boundary so callers see
    the precise typed form.
    """
    assert isinstance(value, Mapping)
    return cast(Mapping[str, object], value)


def test_truncate_short_passthrough() -> None:
    assert truncate("abc", 10) == "abc"


def test_truncate_long_appends_notice() -> None:
    out = truncate("x" * 50, 10)
    assert out.startswith("x" * 10)
    assert "truncated" in out
    assert "40" in out


def test_to_result_wraps_string() -> None:
    r = to_result("hello")
    assert isinstance(r, ToolResult)
    assert r.content == "hello"
    assert r.call_id == ""


def test_to_result_passthrough_tool_result() -> None:
    src = ToolResult(call_id="x", content="y", is_error=True)
    assert to_result(src) is src


def test_opt_int_missing_returns_none() -> None:
    assert opt_int({}, "k") is None


def test_opt_int_present_coerces() -> None:
    assert opt_int({"k": "42"}, "k") == 42


def test_opt_str_missing_returns_none() -> None:
    assert opt_str({}, "k") is None


def test_opt_str_empty_returns_none() -> None:
    assert opt_str({"k": ""}, "k") is None


def test_opt_str_present() -> None:
    assert opt_str({"k": "v"}, "k") == "v"


def test_resolve_tool_path_empty_passthrough() -> None:
    assert resolve_tool_path("") == ""


def test_resolve_tool_path_absolute_unchanged(tmp_path: Path) -> None:
    abs_p = str(tmp_path / "f.txt")
    with with_fake_agent():
        assert resolve_tool_path(abs_p) == abs_p


def test_resolve_tool_path_relative_joins_cwd(tmp_path: Path) -> None:
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        out = resolve_tool_path("sub/f.txt")
    assert out == str(tmp_path / "sub/f.txt")


def test_resolve_tool_path_tilde_expansion() -> None:
    with with_fake_agent():
        out = resolve_tool_path("~/foo.txt")
    # ``~`` is expanded to ``$HOME``; result must be absolute.
    assert Path(out).is_absolute()


def test_tool_state_initial_state() -> None:
    s = ToolState()
    assert s.recent_files == []
    assert s.depth == 0
    assert s.bash_cwd == s.start_cwd
    assert s.additional_dirs == []
    assert s.read_cache == {}


def test_tool_state_mark_read_records(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hi")
    s = ToolState()
    s.mark_read(str(f), offset=1, limit=10, content="hi")
    assert s.has_been_read(str(f))
    assert s.recent_files == [str(f)]
    assert str(f.resolve()) in s.read_cache


def test_tool_state_mark_read_mtime_default_stats(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hi")
    s = ToolState()
    s.mark_read(str(f))
    entry = s.read_cache[str(f.resolve())]
    assert entry.mtime > 0.0


def test_tool_state_mark_read_missing_file_mtime_zero(tmp_path: Path) -> None:
    s = ToolState()
    missing = tmp_path / "missing.txt"
    s.mark_read(str(missing))
    assert s.read_cache[str(missing.resolve())].mtime == 0.0


def test_tool_state_mark_read_reorders(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a")
    b.write_text("b")
    s = ToolState()
    s.mark_read(str(a))
    s.mark_read(str(b))
    s.mark_read(str(a))
    # After re-reading ``a``, it should be at the end (most recent).
    assert s.recent_files[-1] == str(a)


def test_tool_state_mark_written_resets_window(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("v0")
    s = ToolState()
    s.mark_read(str(f), offset=5, limit=100, content="v0")
    s.mark_written(str(f))
    entry = s.read_cache[str(f.resolve())]
    assert entry.offset == 0
    assert entry.limit == 0


def test_tool_state_mark_written_missing_file_zero_mtime(tmp_path: Path) -> None:
    s = ToolState()
    missing = tmp_path / "missing.txt"
    s.mark_written(str(missing))
    assert s.read_cache[str(missing.resolve())].mtime == 0.0


def test_tool_state_check_unchanged_no_record_false(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    assert s.check_unchanged(str(f), 0, 0) is False


def test_tool_state_check_unchanged_window_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    s.mark_read(str(f), offset=1, limit=10)
    assert s.check_unchanged(str(f), 2, 10) is False


def test_tool_state_check_unchanged_match(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    s.mark_read(str(f), offset=1, limit=10)
    assert s.check_unchanged(str(f), 1, 10) is True


def test_tool_state_check_unchanged_missing_file_false(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    s.mark_read(str(f), offset=1, limit=10)
    f.unlink()
    assert s.check_unchanged(str(f), 1, 10) is False


def test_tool_state_check_stale_no_record_false(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    assert s.check_stale(str(f)) is False


def test_tool_state_check_stale_unchanged_mtime(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    s.mark_read(str(f), content="x")
    assert s.check_stale(str(f)) is False


def test_tool_state_check_stale_content_changed(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    s.mark_read(str(f), content="x", mtime=1.0)
    f.write_text("y")
    assert s.check_stale(str(f)) is True


def test_tool_state_check_stale_content_equal_refreshes_mtime(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    s.mark_read(str(f), content="x", mtime=1.0)
    new_mtime = time.time() + 100
    os.utime(f, (new_mtime, new_mtime))
    # Same content, different mtime → not stale; cache mtime updates.
    assert s.check_stale(str(f)) is False
    assert s.read_cache[str(f.resolve())].mtime == pytest.approx(new_mtime)


def test_tool_state_check_stale_missing_file_false(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    s.mark_read(str(f), content="x")
    f.unlink()
    assert s.check_stale(str(f)) is False


def test_tool_state_check_stale_binary_cache_conservative_stale(
    tmp_path: Path,
) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"\x00\x01")
    s = ToolState()
    # No ``content`` argument => no cached content → conservative stale.
    s.mark_read(str(f), mtime=1.0)
    f.write_bytes(b"\x02")
    assert s.check_stale(str(f)) is True


def test_tool_state_reset_file_tracking(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    s.mark_read(str(f), content="x")
    s.reset_file_tracking()
    assert s.read_cache == {}
    assert not s.has_been_read(str(f))
    assert s.recent_files == []


def test_tool_state_enforce_read_returns_error_when_unread(tmp_path: Path) -> None:
    s = ToolState()
    err = s.enforce_read(str(tmp_path / "missing"))
    assert err is not None
    assert "not yet read" in err


def test_tool_state_enforce_read_none_after_read(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    s.mark_read(str(f), content="x")
    assert s.enforce_read(str(f)) is None


def test_tool_state_consume_changed_files_returns_diff(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("alpha\n")
    s = ToolState()
    s.mark_read(str(f), content="alpha\n", mtime=1.0)
    f.write_text("beta\n")
    diffs = s.consume_changed_files()
    assert str(f) in diffs
    assert "beta" in diffs[str(f)]


def test_tool_state_consume_changed_files_idempotent(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("alpha\n")
    s = ToolState()
    s.mark_read(str(f), content="alpha\n", mtime=1.0)
    f.write_text("beta\n")
    _ = s.consume_changed_files()
    assert s.consume_changed_files() == {}


def test_tool_state_consume_changed_files_missing_file_skipped(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("alpha\n")
    s = ToolState()
    s.mark_read(str(f), content="alpha\n", mtime=1.0)
    f.unlink()
    assert s.consume_changed_files() == {}


def test_get_tool_state_default_outside_context() -> None:
    s = get_tool_state()
    assert isinstance(s, ToolState)


def test_tool_state_context_swaps_state() -> None:
    custom = ToolState()
    custom.bash_cwd = "/tmp"  # noqa: S108 -- test placeholder, not real fs use
    with tool_state_context(custom):
        assert get_tool_state() is custom
    assert get_tool_state() is not custom


def test_module_mark_read_and_has_been_read(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    s = ToolState()
    with tool_state_context(s):
        mark_read(str(f), content="x")
        assert has_been_read(str(f))


def test_get_file_write_lock_same_path_returns_same_lock(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("v")
    lk1 = get_file_write_lock(str(p))
    lk2 = get_file_write_lock(str(p))
    assert lk1 is lk2


def test_get_file_write_lock_diff_paths_distinct(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("v")
    b.write_text("v")
    assert get_file_write_lock(str(a)) is not get_file_write_lock(str(b))


def test_tool_decorator_bare_sync() -> None:
    @tool
    def my_fn(x: str) -> str:
        return x.upper()

    assert my_fn.name == "my_fn"
    assert my_fn.tool_id == "application/x-tool-my_fn"


def test_tool_decorator_paren_overrides() -> None:
    @tool(name="Custom", description="custom desc")
    def my_fn(x: str) -> str:
        return x

    assert my_fn.name == "Custom"
    assert my_fn.description == "custom desc"
    assert my_fn.tool_id == "application/x-tool-custom"


def test_tool_summary_default() -> None:
    @tool(name="X")
    def fn(x: str) -> str:
        return x

    assert fn.summary({"k": "v"}) == "X"


def test_tool_prompt_default_empty() -> None:
    @tool(name="X")
    def fn(x: str) -> str:
        return x

    assert fn.prompt() == ""


def test_tool_summary_result_default_none() -> None:
    @tool(name="X")
    def fn(x: str) -> str:
        return x

    assert fn.summary_result(ToolResult(call_id="", content="hi")) is None


@pytest.mark.asyncio
async def test_tool_run_sync_function_returns_result() -> None:
    @tool
    def fn(msg: str) -> str:
        return msg.upper()

    out = await fn.run({"msg": "hi"})
    assert out.content == "HI"


@pytest.mark.asyncio
async def test_tool_run_async_function_returns_result() -> None:
    @tool
    async def fn(msg: str) -> str:
        return f"async:{msg}"

    out = await fn.run({"msg": "x"})
    assert out.content == "async:x"


@pytest.mark.asyncio
async def test_tool_run_truncates_long_content() -> None:
    @tool(max_result_chars=10)
    def fn(x: str) -> str:
        del x
        return "y" * 100

    out = await fn.run({"x": ""})
    assert len(out.content) > 10  # contains the truncation notice too.
    assert "truncated" in out.content


@pytest.mark.asyncio
async def test_tool_run_tool_result_passthrough() -> None:
    @tool
    def fn(x: str) -> ToolResult:
        return ToolResult(call_id="", content=x, is_error=True)

    out = await fn.run({"x": "boom"})
    assert out.is_error
    assert out.content == "boom"


def test_tool_schema_built_from_signature() -> None:
    @tool
    def fn(name: str, count: int = 1) -> str:
        del count
        return name

    schema = fn.directive_schema
    assert isinstance(schema, Mapping)
    props = schema["properties"]
    assert isinstance(props, Mapping)
    assert "name" in props
    assert "count" in props
    assert schema["required"] == ("name",)


def test_tool_schema_annotated_description() -> None:
    @tool
    def fn(field: Annotated[str, "Field documentation."]) -> str:
        return field

    prop = _schema_property(fn.directive_schema, "field")
    assert prop["description"] == "Field documentation."


def test_tool_schema_supports_list_and_dict() -> None:
    @tool
    def fn(items: list[int], extra: dict[str, str]) -> str:
        del items, extra
        return "ok"

    assert _schema_property(fn.directive_schema, "items")["type"] == "array"
    assert _schema_property(fn.directive_schema, "extra")["type"] == "object"


def test_tool_schema_unknown_type_string_fallback() -> None:
    @tool
    def fn(x: Path) -> str:  # Path isn't in _TYPE_MAP.
        return str(x)

    assert _schema_property(fn.directive_schema, "x")["type"] == "string"


def test_tool_decorator_explicit_schema_override() -> None:
    @tool(schema=None)
    def fn(x: str) -> str:
        return x

    # ``schema=None`` falls back to auto-generation.
    assert isinstance(fn, _ToolImpl)


@pytest.mark.asyncio
async def test_run_sync_wraps_string() -> None:
    def fn(*, x: str) -> str:
        return f"got:{x}"

    out = await run_sync(fn, x="hello")
    assert out.content == "got:hello"


@pytest.mark.asyncio
async def test_run_sync_passes_tool_result() -> None:
    def fn() -> ToolResult:
        return ToolResult(call_id="", content="hi", is_error=True)

    out = await run_sync(fn)
    assert out.is_error


@pytest.mark.asyncio
async def test_run_sync_truncates_long() -> None:
    def fn() -> str:
        return "x" * (TOOL_RESULT_MAX_CHARS + 100)

    out = await run_sync(fn)
    assert "truncated" in out.content


def test_resolve_recipe_bare_name() -> None:
    p = resolve_recipe("sagent")
    assert p.name == "sagent.yaml"


def test_resolve_recipe_explicit_path(tmp_path: Path) -> None:
    target = tmp_path / "x.yaml"
    target.write_text("k: v\n")
    p = resolve_recipe(str(target))
    assert p == target.resolve()


def test_set_recipe_switches_and_clears_cache(tmp_path: Path) -> None:
    target = tmp_path / "test.yaml"
    target.write_text("tool_descriptions:\n  Demo: nope\n")
    set_recipe(str(target))
    try:
        # recipe_dict reflects the override.
        out = recipe_dict("tool_descriptions")
        assert out.get("Demo") == "nope"
    finally:
        set_recipe("sagent")


def test_recipe_dict_missing_section_empty() -> None:
    assert recipe_dict("__nonexistent__") == {}


def test_recipe_list_missing_section_empty() -> None:
    assert recipe_list("__nonexistent__", "k") == []


def test_recipe_dict_non_dict_value_returns_empty(tmp_path: Path) -> None:
    target = tmp_path / "test.yaml"
    target.write_text("scalar: 42\n")
    set_recipe(str(target))
    try:
        # ``scalar`` is an int, not a dict → empty mapping.
        assert recipe_dict("scalar") == {}
    finally:
        set_recipe("sagent")


def test_recipe_list_present(tmp_path: Path) -> None:
    target = tmp_path / "test.yaml"
    target.write_text("kit:\n  tools: [a, b, c]\n")
    set_recipe(str(target))
    try:
        assert recipe_list("kit", "tools") == ["a", "b", "c"]
    finally:
        set_recipe("sagent")


def test_recipe_list_non_list_value(tmp_path: Path) -> None:
    target = tmp_path / "test.yaml"
    target.write_text("kit:\n  tools: not-a-list\n")
    set_recipe(str(target))
    try:
        assert recipe_list("kit", "tools") == []
    finally:
        set_recipe("sagent")


def test_recipe_list_section_not_dict(tmp_path: Path) -> None:
    target = tmp_path / "test.yaml"
    target.write_text("kit: [a, b]\n")
    set_recipe(str(target))
    try:
        assert recipe_list("kit", "tools") == []
    finally:
        set_recipe("sagent")


def test_recipe_yaml_non_dict_root_returns_empty(tmp_path: Path) -> None:
    target = tmp_path / "test.yaml"
    target.write_text("just a string\n")
    set_recipe(str(target))
    try:
        # Non-dict root → empty config.
        assert recipe_dict("anything") == {}
    finally:
        set_recipe("sagent")


def test_read_asset_includes_directive(tmp_path: Path) -> None:
    inner = tmp_path / "inner.txt"
    inner.write_text("INSIDE")
    outer = tmp_path / "outer.txt"
    outer.write_text(f"a\n{{{{include: {inner}}}}}\nb")
    text = read_asset(outer)
    assert "INSIDE" in text
    assert "a\n" in text
    assert "\nb" in text


def test_load_tool_description_missing_returns_empty() -> None:
    # A name not present in the sagent recipe falls back to "".
    set_recipe("sagent")
    desc = load_tool_description("__totally_made_up_tool__")
    assert desc == ""


def test_load_tool_description_known_returns_nonempty() -> None:
    set_recipe("sagent")
    desc = load_tool_description("Read")
    assert desc != ""


def test_changed_files_context_empty_when_no_changes() -> None:
    with with_fake_agent():
        assert changed_files_context() == ""


def test_changed_files_context_returns_reminder(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("v0\n")
    with with_fake_agent() as agent:
        agent.tool_state.mark_read(str(f), content="v0\n", mtime=1.0)
        f.write_text("v1\n")
        out = changed_files_context()
    assert "<system-reminder>" in out
    assert str(f) in out


def test_changed_files_context_truncates_large_diff(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("\n".join(f"old{i}" for i in range(50)) + "\n")
    with with_fake_agent() as agent:
        agent.tool_state.mark_read(str(f), content=f.read_text(), mtime=1.0)
        f.write_text("\n".join(f"new{i}" for i in range(50)) + "\n")
        out = changed_files_context(max_diff_lines=2)
    assert "(truncated)" in out


def test_read_cache_entry_is_namedtuple() -> None:
    e = ReadCacheEntry(0, 1, 2, 3.0)
    assert e.offset == 0
    assert e.limit == 1
    assert e.last_lines == 2
    assert e.mtime == 3.0


@pytest.mark.asyncio
async def test_get_file_write_lock_serializes(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("v")
    lock = get_file_write_lock(str(p))
    order: list[str] = []

    async def t(label: str) -> None:
        async with lock:
            order.append(f"in:{label}")
            await asyncio.sleep(0)
            order.append(f"out:{label}")

    await asyncio.gather(t("a"), t("b"))
    assert order in (
        ["in:a", "out:a", "in:b", "out:b"],
        ["in:b", "out:b", "in:a", "out:a"],
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
