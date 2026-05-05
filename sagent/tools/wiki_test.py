# pytest fixtures look unused to pyright; pytest wires them by name
"""Tests for tools.wiki."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools import wiki as wiki_mod
from sagent.tools.core import get_tool_state


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-x"),),
        "multipart/x-tool-call",
    )


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Build a minimal wiki under tmp_path and return its root."""
    root = tmp_path / "wiki"
    (root / "pages").mkdir(parents=True)
    (root / "SCHEMA.md").write_text("# Wiki schema\n")
    (root / "index.md").write_text("# Index\n- [[one]]\n")
    (root / "pages" / "one.md").write_text(
        "---\ntitle: One\ntags: []\nsources: []\nupdated: 2026-01-01\n---\n\nSee [[two]].\n",
    )
    (root / "pages" / "two.md").write_text(
        "---\ntitle: Two\ntags: []\nsources: []\nupdated: 2026-01-01\n---\n\nSee [[missing]].\n",
    )
    (root / "pages" / "bad.md").write_text("no frontmatter\n\n[[one]]\n")
    return root


class TestFindRoot:
    def test_from_wiki_dir(self, wiki_root: Path) -> None:
        assert wiki_mod.find_root(wiki_root) == wiki_root

    def test_finds_nested_wiki_subdir(self, tmp_path: Path) -> None:
        root = tmp_path
        (root / "wiki").mkdir()
        (root / "wiki" / "SCHEMA.md").write_text("x")
        assert wiki_mod.find_root(root) == root / "wiki"

    def test_walks_up(self, wiki_root: Path) -> None:
        # Start from a subdirectory and walk up to root.
        sub = wiki_root / "pages"
        assert wiki_mod.find_root(sub) == wiki_root

    def test_none_when_no_schema(self, tmp_path: Path) -> None:
        assert wiki_mod.find_root(tmp_path) is None


class TestListAndRead:
    def test_list_pages(self, wiki_root: Path) -> None:
        pages = wiki_mod.list_pages(wiki_root)
        assert "one" in pages
        assert "two" in pages
        assert "bad" in pages

    def test_read_page(self, wiki_root: Path) -> None:
        content = wiki_mod.read_page(wiki_root, "one")
        assert content is not None
        assert "See [[two]]" in content

    def test_read_missing(self, wiki_root: Path) -> None:
        assert wiki_mod.read_page(wiki_root, "nope") is None


class TestLint:
    def test_broken_link(self, wiki_root: Path) -> None:
        result = wiki_mod.lint(wiki_root)
        assert any("missing" in err for err in result["broken_links"])

    def test_missing_frontmatter(self, wiki_root: Path) -> None:
        result = wiki_mod.lint(wiki_root)
        assert any(err.startswith("bad.md") for err in result["missing_frontmatter"])


@pytest.fixture
def wiki_cwd(wiki_root: Path) -> Path:
    """Point the tool-state cwd at a populated wiki root."""
    get_tool_state().bash_cwd = str(wiki_root)
    return wiki_root


def _text(r: Message) -> str:
    if isinstance(r, TextMessage):
        return r.content
    if isinstance(r, MultipartMessage):
        for p in r.content:
            if isinstance(p, TextMessage) and p.descriptor == "text/plain":
                return p.content
    return ""


class TestWikiTool:
    def test_description_discourages_discovery_use(self) -> None:
        desc = wiki_mod.Wiki.description
        assert "Do not use for ordinary repo docs" in desc
        assert "If locate fails once, do not retry" in desc

    @pytest.mark.anyio
    async def test_locate(self, wiki_cwd: Path) -> None:
        tool = wiki_mod.Wiki()
        resp = await tool.run(_msg(json_freeze({"operation": "locate"})))
        assert str(wiki_cwd) in _text(resp)

    @pytest.mark.anyio
    async def test_list(self, wiki_cwd: Path) -> None:
        del wiki_cwd
        tool = wiki_mod.Wiki()
        resp = await tool.run(_msg(json_freeze({"operation": "list"})))
        assert "one" in _text(resp)
        assert "two" in _text(resp)

    @pytest.mark.anyio
    async def test_read_page(self, wiki_cwd: Path) -> None:
        del wiki_cwd
        tool = wiki_mod.Wiki()
        resp = await tool.run(
            _msg(json_freeze({"operation": "read_page", "slug": "one"}))
        )
        assert "See [[two]]" in _text(resp)

    @pytest.mark.anyio
    async def test_rejects_invalid_slug(self, wiki_cwd: Path) -> None:
        del wiki_cwd
        tool = wiki_mod.Wiki()
        resp = await tool.run(
            _msg(json_freeze({"operation": "read_page", "slug": "../index"}))
        )
        assert resp.descriptor == "text/x-error"
        assert "Invalid page slug" in str(resp.content)

    @pytest.mark.anyio
    async def test_read_index(self, wiki_cwd: Path) -> None:
        del wiki_cwd
        tool = wiki_mod.Wiki()
        resp = await tool.run(_msg(json_freeze({"operation": "read_index"})))
        assert "Index" in _text(resp)

    @pytest.mark.anyio
    async def test_lint_reports(self, wiki_cwd: Path) -> None:
        del wiki_cwd
        tool = wiki_mod.Wiki()
        resp = await tool.run(_msg(json_freeze({"operation": "lint"})))
        assert "missing" in _text(resp)

    @pytest.mark.anyio
    async def test_unknown_op(self, wiki_cwd: Path) -> None:
        del wiki_cwd
        tool = wiki_mod.Wiki()
        resp = await tool.run(_msg(json_freeze({"operation": "no_such"})))
        assert resp.descriptor == "text/x-error"
        assert "Unknown operation" in str(resp.content)

    @pytest.mark.anyio
    async def test_no_wiki(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        get_tool_state().bash_cwd = str(empty)
        tool = wiki_mod.Wiki()
        resp = await tool.run(_msg(json_freeze({"operation": "locate"})))
        assert resp.descriptor == "text/x-error"
        assert "No wiki found" in str(resp.content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
