"""Tests for ``tools.wiki``: structural primitives for an LLM wiki."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.agent.runtime import ToolResult
from sagent.testing import with_fake_agent
from sagent.tools import wiki as wm
from sagent.tools.wiki import Wiki


def _make_wiki(root: Path, pages: dict[str, str] | None = None) -> Path:
    """Create a minimal wiki tree at ``root`` with SCHEMA.md + pages."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "SCHEMA.md").write_text("# schema\n", encoding="utf-8")
    pages_dir = root / "pages"
    pages_dir.mkdir(exist_ok=True)
    for slug, content in (pages or {}).items():
        (pages_dir / f"{slug}.md").write_text(content, encoding="utf-8")
    return root


def _resolved_str(p: Path) -> str:
    """Sync helper to call ``.resolve()`` outside async contexts."""
    return str(p.resolve())


_GOOD_FRONTMATTER = (
    "---\n"
    "title: A page\n"
    "tags: [t1]\n"
    "sources: [s1]\n"
    "updated: 2026-01-01\n"
    "---\n\n"
    "Body text.\n"
)


def test_valid_slug_accepts_simple() -> None:
    assert wm.valid_slug("abc")
    assert wm.valid_slug("a-b-c-1")


def test_valid_slug_rejects_uppercase_and_punct() -> None:
    assert not wm.valid_slug("AbC")
    assert not wm.valid_slug("a_b")
    assert not wm.valid_slug("")
    assert not wm.valid_slug("-abc")


def test_find_root_at_top(tmp_path: Path) -> None:
    _make_wiki(tmp_path)
    assert wm.find_root(tmp_path) == tmp_path.resolve()


def test_find_root_walks_up(tmp_path: Path) -> None:
    wiki = _make_wiki(tmp_path / "wiki")
    deep = tmp_path / "wiki" / "pages"
    assert wm.find_root(deep) == wiki.resolve()


def test_find_root_via_wiki_subdir(tmp_path: Path) -> None:
    wiki = _make_wiki(tmp_path / "wiki")
    assert wm.find_root(tmp_path) == wiki.resolve()


def test_find_root_returns_none(tmp_path: Path) -> None:
    assert wm.find_root(tmp_path) is None


def test_list_pages_sorted(tmp_path: Path) -> None:
    root = _make_wiki(tmp_path, {"b": "b", "a": "a", "c-d": "c"})
    assert wm.list_pages(root) == ["a", "b", "c-d"]


def test_list_pages_missing_dir(tmp_path: Path) -> None:
    (tmp_path / "SCHEMA.md").write_text("", encoding="utf-8")
    assert wm.list_pages(tmp_path) == []


def test_read_page_returns_text(tmp_path: Path) -> None:
    root = _make_wiki(tmp_path, {"hello": "hi"})
    assert wm.read_page(root, "hello") == "hi"


def test_read_page_invalid_slug_returns_none(tmp_path: Path) -> None:
    root = _make_wiki(tmp_path)
    assert wm.read_page(root, "BAD!") is None


def test_read_page_missing_returns_none(tmp_path: Path) -> None:
    root = _make_wiki(tmp_path)
    assert wm.read_page(root, "missing") is None


def test_lint_clean(tmp_path: Path) -> None:
    root = _make_wiki(tmp_path, {"a": _GOOD_FRONTMATTER + "Refs [[a]]\n"})
    out = wm.lint(root)
    assert out == {"broken_links": [], "missing_frontmatter": []}


def test_lint_detects_missing_frontmatter(tmp_path: Path) -> None:
    root = _make_wiki(tmp_path, {"a": "Just a body, no frontmatter.\n"})
    out = wm.lint(root)
    assert out["missing_frontmatter"]
    assert "missing" in out["missing_frontmatter"][0]


def test_lint_detects_broken_links(tmp_path: Path) -> None:
    root = _make_wiki(tmp_path, {"a": _GOOD_FRONTMATTER + "See [[absent]]\n"})
    out = wm.lint(root)
    assert any("[[absent]]" in s for s in out["broken_links"])


def test_metadata_basics() -> None:
    t = Wiki()
    assert t.name == "Wiki"
    assert t.tool_id == "application/x-tool-wiki"
    assert t.supports_microcompaction is False
    assert t.prompt() == ""


def test_summary_includes_slug() -> None:
    t = Wiki()
    assert t.summary({"operation": "read_page", "slug": "intro"}) == (
        "Wiki read_page:intro"
    )
    assert t.summary({"operation": "list"}) == "Wiki list"
    assert t.summary_result(ToolResult(call_id="", content="x")) is None


@pytest.mark.asyncio
async def test_run_unknown_operation(tmp_path: Path) -> None:
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "bogus"})
    assert result.is_error
    assert "Unknown operation" in result.content


@pytest.mark.asyncio
async def test_run_no_wiki_found(tmp_path: Path) -> None:
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "locate"})
    assert result.is_error
    assert "No wiki found" in result.content


@pytest.mark.asyncio
async def test_run_locate(tmp_path: Path) -> None:
    _make_wiki(tmp_path)
    expected = _resolved_str(tmp_path)
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "locate"})
    assert not result.is_error
    assert result.content == expected


@pytest.mark.asyncio
async def test_run_list_pages(tmp_path: Path) -> None:
    _make_wiki(tmp_path, {"a": "x", "b": "y"})
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "list"})
    assert not result.is_error
    assert result.content == "a\nb"


@pytest.mark.asyncio
async def test_run_list_empty(tmp_path: Path) -> None:
    _make_wiki(tmp_path)
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "list"})
    assert result.content == "(no pages)"


@pytest.mark.asyncio
async def test_run_read_page_missing_slug(tmp_path: Path) -> None:
    _make_wiki(tmp_path)
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "read_page"})
    assert result.is_error
    assert "requires" in result.content


@pytest.mark.asyncio
async def test_run_read_page_invalid_slug(tmp_path: Path) -> None:
    _make_wiki(tmp_path)
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "read_page", "slug": "BAD!"})
    assert result.is_error
    assert "Invalid page slug" in result.content


@pytest.mark.asyncio
async def test_run_read_page_no_such(tmp_path: Path) -> None:
    _make_wiki(tmp_path)
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "read_page", "slug": "absent"})
    assert result.is_error
    assert "No such page" in result.content


@pytest.mark.asyncio
async def test_run_read_page_success(tmp_path: Path) -> None:
    _make_wiki(tmp_path, {"hello": "<safe>body</safe>"})
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "read_page", "slug": "hello"})
    assert not result.is_error
    # html.escape() converts < and >.
    assert "&lt;safe&gt;" in result.content


@pytest.mark.asyncio
async def test_run_read_index_missing(tmp_path: Path) -> None:
    _make_wiki(tmp_path)
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "read_index"})
    assert not result.is_error
    assert result.content == "(no index.md)"


@pytest.mark.asyncio
async def test_run_read_index_present(tmp_path: Path) -> None:
    _make_wiki(tmp_path)
    (tmp_path / "index.md").write_text("Hello <world>", encoding="utf-8")
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "read_index"})
    assert not result.is_error
    assert "&lt;world&gt;" in result.content


@pytest.mark.asyncio
async def test_run_lint_clean(tmp_path: Path) -> None:
    _make_wiki(tmp_path, {"a": _GOOD_FRONTMATTER})
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "lint"})
    assert result.content == "Lint clean."


@pytest.mark.asyncio
async def test_run_lint_reports_both_kinds(tmp_path: Path) -> None:
    _make_wiki(
        tmp_path,
        {
            "a": _GOOD_FRONTMATTER + "See [[absent]]\n",
            "b": "no frontmatter\n",
        },
    )
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "lint"})
    assert "Broken wikilinks" in result.content
    assert "Missing frontmatter" in result.content


@pytest.mark.asyncio
async def test_run_uses_cwd_override(tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    _make_wiki(other)
    expected = _resolved_str(other)
    t = Wiki()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"operation": "locate", "cwd": str(other)})
    assert not result.is_error
    assert result.content == expected


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
