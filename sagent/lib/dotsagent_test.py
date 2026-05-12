"""Tests for ``lib.dotsagent``: ``.sagent/`` discovery + frontmatter parser."""

from __future__ import annotations

from pathlib import Path

from sagent.lib.dotsagent import parse_frontmatter, walk_up


def test_walk_up_orders_root_first(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    parts = walk_up(deep)
    # Last element is the input; earlier elements are ancestors.
    assert parts[-1] == deep.resolve()
    assert deep.resolve().parent in parts
    assert deep.resolve().parent.parent in parts


def test_walk_up_root_only() -> None:
    parts = walk_up(Path("/"))
    assert parts == [Path("/").resolve()]


def test_parse_frontmatter_present() -> None:
    raw = "---\nkey: value\nn: 1\n---\nbody text\n"
    meta, body = parse_frontmatter(raw)
    assert meta == {"key": "value", "n": 1}
    assert body == "body text\n"


def test_parse_frontmatter_missing() -> None:
    raw = "no frontmatter here\n"
    meta, body = parse_frontmatter(raw)
    assert meta == {}
    assert body == raw


def test_parse_frontmatter_crlf_line_endings() -> None:
    raw = "---\r\nkey: value\r\n---\r\nbody\r\n"
    meta, body = parse_frontmatter(raw)
    assert meta == {"key": "value"}
    assert body == "body\r\n"


def test_parse_frontmatter_invalid_yaml_returns_empty() -> None:
    # Unmatched bracket triggers ``yaml.YAMLError`` -> ({}, raw).
    raw = "---\nkey: [unclosed\n---\nbody\n"
    meta, body = parse_frontmatter(raw)
    assert meta == {}
    assert body == raw


def test_parse_frontmatter_non_dict_yaml_returns_empty() -> None:
    # YAML list, not mapping.
    raw = "---\n- a\n- b\n---\nbody\n"
    meta, body = parse_frontmatter(raw)
    assert meta == {}
    assert body == "body\n"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
