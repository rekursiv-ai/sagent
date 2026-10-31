"""YAML documents, parsed so a typo fails instead of silently changing a value.

Typical usage example:

  palette = parse(path.read_text(), source=str(path))
  policy = parse(text, aliases=True)  # A document that shares nodes on purpose.
  meta, body = split_frontmatter(path.read_text(), source=str(path)) or (None, text)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import yaml


class StrictYAMLError(yaml.YAMLError, ValueError):
    """A document ``parse`` refuses: a key given twice, an odd key, an alias."""


def parse(
    text: str,
    *,
    aliases: bool = False,
    string_keys: bool = True,
    source: str = "",
) -> object:
    """Return the YAML document ``text`` as plain data.

    PyYAML keeps the last of a key given twice; this refuses it, since in a
    hand-written document the first was a typo or a lost edit. The C parser is
    used when PyYAML has it: about nine times the pure-Python one on sagent's
    assets, and the checks add about a third to one parse.

    Args:
      text: One YAML document.
      aliases: Accept ``&anchor`` / ``*alias`` node sharing, merge keys included.
      string_keys: Refuse a mapping key YAML reads as other than a string, as
        ``1:`` or ``on:``. A GitHub workflow's ``on:`` needs ``False``.
      source: Where ``text`` came from, such as its path, named in errors.

    Returns:
      document: Its plain data, or ``None`` when ``text`` holds none.

    Raises:
      StrictYAMLError: A key repeats within a mapping, a key is no string
        under ``string_keys``, or the document uses an alias without
        ``aliases``; the message names the line.

    """
    where = f"{source}, " if source else ""
    loader_class = yaml.CSafeLoader if yaml.__with_libyaml__ else yaml.SafeLoader
    # An alias leaves no trace in the composed nodes but the node it shares, so
    # only the events say where the alias itself was written. It is spelled
    # ``*``: a document without one skips that second pass.
    if not aliases and "*" in text:
        for event in yaml.parse(text, Loader=loader_class):
            if isinstance(event, yaml.AliasEvent):
                # The C parser's mark is its own class, no ``yaml.Mark``.
                assert isinstance(event.start_mark, _Position)
                line = event.start_mark.line + 1
                raise StrictYAMLError(
                    f"{where}line {line}: YAML aliases are not allowed.",
                )
    loader = loader_class(text)
    try:
        node = loader.get_single_node()
        _validate(node, loader=loader, where=where, string_keys=string_keys, seen=set())
        return None if node is None else loader.construct_document(node)
    finally:
        loader.dispose()


def split_frontmatter(text: str, *, source: str = "") -> tuple[object, str] | None:
    """Return a Markdown file's YAML frontmatter, parsed, and the body after it.

    The frontmatter is the lines between a first line of ``---`` and the next
    line of ``---``; a leading byte-order mark is dropped.

    Args:
      text: The file's text.
      source: Where ``text`` came from, such as its path, named in errors.

    Returns:
      split: The frontmatter's plain data (``None`` when empty) and the text
        after its closing line; ``None`` when ``text`` opens with no block.

    Raises:
      YAMLError: The frontmatter will not parse; the message names the file's
        line, and ``source``.

    """
    lines = text.removeprefix("\ufeff").splitlines(keepends=True)
    # Only trailing space is dropped: an indented ``---`` is a block scalar's text.
    if not lines or lines[0].rstrip() != "---":
        return None
    closing = next(
        (i for i, line in enumerate(lines) if i and line.rstrip() == "---"),
        0,
    )
    if not closing:
        return None
    # The leading newline stands for the opening line, so errors count the
    # file's lines, not the block's.
    meta = parse("\n" + "".join(lines[1:closing]), source=source)
    return meta, "".join(lines[closing + 1 :])


def _validate(
    node: yaml.Node | None,
    *,
    loader: yaml.SafeLoader | yaml.CSafeLoader,
    where: str,
    string_keys: bool,
    seen: set[int],
) -> None:
    """Reject keys given twice, and keys that are no string under ``string_keys``."""
    # An allowed alias reuses a node, and a recursive one would loop forever.
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    rules = {"loader": loader, "where": where, "string_keys": string_keys, "seen": seen}
    if isinstance(node, yaml.MappingNode):
        keys: set[object] = set()
        for key, value in node.value:
            line = key.start_mark.line + 1
            merge = key.tag == "tag:yaml.org,2002:merge"
            if string_keys and not merge and key.tag != "tag:yaml.org,2002:str":
                raise StrictYAMLError(
                    f"{where}line {line}: YAML mapping keys must be strings.",
                )
            # Compared as the values they read as: ``on`` and ``yes`` are both True.
            if isinstance(key, yaml.ScalarNode) and not merge:
                read = loader.construct_object(key)
                if read in keys:
                    raise StrictYAMLError(
                        f"{where}line {line}: the key {key.value!r} is given twice.",
                    )
                keys.add(read)
            _validate(key, **rules)
            _validate(value, **rules)
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            _validate(value, **rules)


@runtime_checkable
class _Position(Protocol):
    """Where an event starts, as both the Python and the C parser mark it."""

    line: int
    column: int
