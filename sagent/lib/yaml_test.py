"""YAML documents parse, or refuse what a typo would silently change."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from sagent.lib.yaml import parse, split_frontmatter


if TYPE_CHECKING:
    from collections.abc import Iterator


def test_a_document_parses_to_plain_data() -> None:
    assert parse("a: [1, two]\nb: {c: d}\n") == {"a": [1, "two"], "b": {"c": "d"}}


def test_an_empty_document_is_none() -> None:
    assert parse("") is None


def test_a_repeated_key_is_refused_at_its_line() -> None:
    with pytest.raises(ValueError, match=r"^line 2: the key 'R' is given twice\.$"):
        parse("R: '#000000'\nR: '#ffffff'\n")


def test_a_refusal_names_its_source() -> None:
    with pytest.raises(ValueError, match=r"^palette\.yaml, line 3: the key 'R' is"):
        parse("a: 1\nR: x\nR: y\n", source="palette.yaml")


def test_two_spellings_of_one_key_are_one_key_given_twice() -> None:
    # ``on`` and ``yes`` both read as ``True``; the second would replace the first.
    with pytest.raises(ValueError, match=r"^line 2: the key 'yes' is given twice\.$"):
        parse("on: a\nyes: b\n", string_keys=False)


def test_an_int_and_a_str_key_of_one_spelling_are_two_keys() -> None:
    assert parse("1: a\n'1': b\n", string_keys=False) == {1: "a", "1": "b"}


def test_a_merge_key_is_an_alias_a_document_may_allow() -> None:
    text = "base: &b {x: 1}\nchild:\n  <<: *b\n  y: 2\n"
    assert parse(text, aliases=True) == {"base": {"x": 1}, "child": {"x": 1, "y": 2}}
    with pytest.raises(ValueError, match=r"^line 3: YAML aliases are not allowed\.$"):
        parse(text)


def test_a_repeated_key_deep_in_a_list_is_refused() -> None:
    with pytest.raises(ValueError, match="the key 'a' is given twice"):
        parse("- {a: 1, a: 2}\n")


@pytest.mark.parametrize(
    ("text", "line"),
    [("a: &x [1]\nb: *x\n", 2), ("a: [&x 1, *x]\n", 1)],
)
def test_an_alias_is_refused_at_its_use_unless_allowed(text: str, line: int) -> None:
    with pytest.raises(ValueError, match=rf"^line {line}: YAML aliases are not"):
        parse(text)
    assert parse(text, aliases=True) == yaml.safe_load(text)


def test_a_refusal_is_a_yaml_error_too() -> None:
    # Callers already catch ``yaml.YAMLError`` for a malformed document.
    with pytest.raises(yaml.YAMLError):
        parse("a: 1\na: 2\n")


@pytest.mark.parametrize("text", ["1: one\n", "on: push\n", "? [a]\n: b\n"])
def test_a_key_that_is_no_string_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match=r"^line 1: YAML mapping keys must be strings"):
        parse(text)


def test_keys_may_be_any_scalar_when_asked() -> None:
    # A GitHub workflow's ``on:`` reads as ``True``.
    assert parse("on: push\n2: two\n", string_keys=False) == {True: "push", 2: "two"}


@pytest.mark.parametrize(
    ("text", "meta", "body"),
    [
        ("---\nname: a\n---\nbody\n", {"name": "a"}, "body\n"),
        ("---\r\nname: a\r\n---\r\nbody\r\n", {"name": "a"}, "body\r\n"),
        ("\ufeff---\nname: a\n---\nbody\n", {"name": "a"}, "body\n"),
        ("---\n---\nbody\n", None, "body\n"),
        ("---  \nname: a\n---\n", {"name": "a"}, ""),
        ("---\nf: |\n  ---\n---\nbody\n", {"f": "---\n"}, "body\n"),
        ("---\nname: a\n---\none\ntwo\n", {"name": "a"}, "one\ntwo\n"),
    ],
    ids=["lf", "crlf", "bom", "empty", "trailing-space", "indented-fence", "lines"],
)
def test_frontmatter_is_split_from_the_body(
    text: str,
    meta: object,
    body: str,
) -> None:
    assert split_frontmatter(text) == (meta, body)


@pytest.mark.parametrize(
    "text",
    ["body only\n", "---\nname: a\nno closing line\n", "", "text\n---\nname: a\n---\n"],
)
def test_text_that_opens_with_no_block_has_no_frontmatter(text: str) -> None:
    assert split_frontmatter(text) is None


def test_a_frontmatter_error_names_the_file_and_its_line() -> None:
    with pytest.raises(ValueError, match=r"^SKILL\.md, line 3: the key 'a' is given"):
        split_frontmatter("---\na: 1\na: 2\n---\nbody\n", source="SKILL.md")


def test_a_frontmatter_error_without_a_source_names_only_the_line() -> None:
    with pytest.raises(ValueError, match=r"^line 3: the key 'a' is given twice\.$"):
        split_frontmatter("---\na: 1\na: 2\n---\nbody\n")


@pytest.mark.parametrize(("text", "passes"), [("a: [1, 2]\n", 0), ("a: '*'\n", 1)])
def test_only_a_document_holding_a_star_is_read_twice(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    passes: int,
) -> None:
    # An alias is spelled with ``*``; without one, the event pass finds none.
    events = yaml.parse
    calls: list[str] = []

    def counted(
        stream: str,
        Loader: type[yaml.SafeLoader | yaml.CSafeLoader],  # noqa: N803 -- PyYAML's name.
    ) -> Iterator[yaml.Event]:
        calls.append(stream)
        return events(stream, Loader=Loader)

    monkeypatch.setattr(yaml, "parse", counted)
    parse(text)
    assert len(calls) == passes


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
