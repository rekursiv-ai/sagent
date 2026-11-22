"""Write what a vendor states back into its catalog document.

Each edit is placed by the document's own YAML structure (``yaml.compose``),
so a row is a row however it is spelled (``- model_id:``, ``- &anchor``,
``- <<: *anchor``), and the comments between two rows are the second's. An
edit is kept only if every section of the edited document still loads with
``ModelTable.from_yaml``; one that would not is left for a person.

A value the model's own row states is rewritten in place, so the document's
comments, order, and spelling stay. A value the row shares (a node another
place reaches too) or inherits (a profile, the reference, a merge key) is left
for a person: writing one row's copy would split a shared fact.

A model the vendor lists and the catalog lacks is added as a row in its
family's newest-first order, or first in ``models`` for a new family. It names
the profile whose levels it has, and the anchored window it shares when the
anchor comes before it, rather than restating either.

A section's ``omits`` follow what its login is served: an omitted model now
served loses its entry, and a served one no longer listed gains one.

Typical usage example:

  left = write(files(google) / "google.yaml", mismatches)  # What it could not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sys

from treekle import from_plain

import yaml

from sagent.lib.yaml import parse, position
from sagent.model.capability import ThinkingBudget, ThinkingEffort, literal_members
from sagent.providers.table import CatalogDocument, ModelTable


if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Mapping, Sequence
    from pathlib import Path

    from sagent.scripts.check_catalog.sources import Finding, NewRow


__all__ = [
    "write",
]


def write(path: Path, findings: Sequence[Finding]) -> list[Finding]:
    """Write each finding's source into the document, where it can be written.

    Args:
      path: The vendor's catalog document.
      findings: Facts whose source differs from the catalog: values, models it
        lacks (``catalogued``), and what a section serves (``served``).

    Returns:
      left: The findings it could not write, in order.

    """
    text = original = path.read_text()
    left: list[Finding] = []
    for finding in findings:
        edited = _edited(text, finding)
        if edited is None or not _loads(edited):
            left.append(finding)
        else:
            text = edited
    if text != original:
        path.write_text(text)
    return left


def _edited(text: str, finding: Finding) -> str | None:
    """Return ``text`` with ``finding`` written in; ``None`` where it cannot be."""
    lines = text.splitlines(keepends=True)
    root = yaml.compose(text)
    assert isinstance(root, yaml.MappingNode)
    if finding.field == "served":
        served = _served(
            lines,
            root,
            section=finding.provider,
            model_id=finding.model_id,
            serve=finding.source == "yes",
            why=f"not listed by {finding.url}",
        )
        return None if served is None else "".join(served)
    if finding.field == "catalogued":
        return (
            None if finding.row is None else "".join(_added(lines, root, finding.row))
        )
    node = _value_node(root, lines, finding)
    if node is None or finding.source is None:
        return None
    start, end = position(node), position(node, end=True)
    line = lines[start.line]
    lines[start.line] = (
        f"{line[: start.column]}{_value(finding.field, finding.source)}"
        f"{line[end.column :]}"
    )
    return "".join(lines)


def _loads(text: str) -> bool:
    """Whether every section of the document ``text`` loads, as ``ModelTable`` reads it."""
    try:
        document = from_plain(parse(text, aliases=True), CatalogDocument)
        for section in document.providers:
            ModelTable.from_text(text, section=section)
    except (ValueError, yaml.YAMLError):
        return False
    return True


def _value_node(
    root: yaml.MappingNode,
    lines: list[str],
    finding: Finding,
) -> yaml.ScalarNode | None:
    """Return the scalar holding ``finding``'s value in its own row, if it has one."""
    row = next((r for r in _rows(root, lines) if r.model_id == finding.model_id), None)
    if row is None:
        return None
    field = finding.field
    if field == "knowledge_cutoff":
        found = _get(row.node, field)
    elif _is_limit(field):
        found = _get(_get(row.node, "limits"), field)
    elif field.startswith("price."):
        found = _price(_get(row.node, "prices"), field.removeprefix("price."))
    else:
        return None
    # A node reached from two places is shared; one row's copy would split it.
    if not isinstance(found, yaml.ScalarNode) or id(found) in _shared(root):
        return None
    return found


def _get(node: yaml.Node | None, key: str) -> yaml.Node | None:
    """Return the value ``node`` maps ``key`` to itself, never through a merge."""
    if not isinstance(node, yaml.MappingNode):
        return None
    return next((v for k, v in node.value if _spells(k, key)), None)


def _key(node: yaml.MappingNode, key: str) -> yaml.Node | None:
    """Return the node that spells ``key`` in ``node``, where the key's line is."""
    return next((k for k, _ in node.value if _spells(k, key)), None)


def _spells(node: yaml.Node, key: str) -> bool:
    """Whether the mapping key ``node`` is the scalar ``key``."""
    return isinstance(node, yaml.ScalarNode) and node.value == key


def _shared(root: yaml.Node) -> set[int]:
    """Return the id of each scalar more than one place reaches, itself or from above."""
    seen: set[int] = set()
    shared: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            shared |= {id(leaf) for leaf in _leaves(node)}
            continue
        seen.add(id(node))
        if isinstance(node, yaml.MappingNode):
            pending += [child for pair in node.value for child in pair]
        elif isinstance(node, yaml.SequenceNode):
            pending += node.value
    return shared


def _is_limit(field: str) -> bool:
    """Whether ``field`` is one of a row's token limits."""
    return field in ("max_request_tokens", "max_response_tokens", "max_context_tokens")


# ``auto.request``, ``auto@2026-12-01.request``, ``priority@2027-03-01.long.request``.
def _price(prices: yaml.Node | None, label: str) -> yaml.Node | None:
    """Return the meter's node in the one price card ``label`` names."""
    parts = label.split(".")
    # ``priority`` alone says a tier is priced or not: no meter to rewrite.
    if not isinstance(prices, yaml.SequenceNode) or len(parts) < 2:
        return None
    tier_day, *qualifiers, meter = parts
    tier, _, day = tier_day.partition("@")
    if qualifiers not in ([], ["long"]):
        return None
    for card in prices.value:
        stated = _get(card, "effective_date")
        when = stated.value if isinstance(stated, yaml.ScalarNode) else ""
        long = _get(card, "effective_prompt_tokens") is not None
        tier_node = _get(card, "service_tier")
        named = tier_node.value if isinstance(tier_node, yaml.ScalarNode) else "auto"
        if (named, when, long) == (tier, day, bool(qualifiers)):
            return _get(card, meter)
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class _Row:
    """One row of ``models``: its node, and the lines that are its own."""

    model_id: str
    node: yaml.MappingNode
    head: int
    """The first line that is the row's: its first comment, else its ``- `` line."""

    end: int
    """The row's last line holding any of its own content."""


def _rows(root: yaml.MappingNode, lines: list[str]) -> list[_Row]:
    """Return each row of ``models`` in order, its comments its own."""
    models = _get(root, "models")
    key = _key(root, "models")
    # Only a document ``CatalogDocument`` read is written to: it has a list of rows.
    assert isinstance(models, yaml.SequenceNode)
    assert isinstance(key, yaml.ScalarNode)
    rows: list[_Row] = []
    before = position(key).line
    for item in models.value:
        assert isinstance(item, yaml.MappingNode)
        start = position(item).line
        # Content is what the row's own scalars span; an alias reaches back above.
        # Its first key is on its first line, so there is always one.
        end = max(
            position(leaf, end=True).line
            for leaf in _leaves(item)
            if position(leaf).line >= start
        )
        head = next(
            (i for i in range(before + 1, start) if lines[i].lstrip().startswith("#")),
            start,
        )
        model_id = _get(item, "model_id")
        # ``CatalogDocument`` read every row's own id, never one merged in.
        assert isinstance(model_id, yaml.ScalarNode)
        rows.append(_Row(model_id=model_id.value, node=item, head=head, end=end))
        before = end
    return rows


def _leaves(node: yaml.Node) -> Iterator[yaml.ScalarNode]:
    """Yield every scalar under ``node``, keys included."""
    if isinstance(node, yaml.ScalarNode):
        yield node
    elif isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            yield from _leaves(key)
            yield from _leaves(value)
    else:
        assert isinstance(node, yaml.SequenceNode)
        for child in node.value:
            yield from _leaves(child)


def _added(lines: list[str], root: yaml.MappingNode, row: NewRow) -> list[str]:
    """Return ``lines`` with ``row`` written in, among its family's by version."""
    document = from_plain(parse("".join(lines), aliases=True), CatalogDocument)
    unprofiled: dict[str, object] = {}
    # The reference or a profile with the row's levels; first one whose window it
    # shares, so no limit is restated.
    fitting = [
        (name, document.reference | values)
        for name, values in (("", unprofiled), *document.profiles.items())
        if _fits(document.reference | values, row)
    ]
    profile, layer = next(
        (
            fit
            for fit in fitting
            if {**_window(fit[1]), **row.limits} == _window(fit[1])
        ),
        fitting[0] if fitting else ("", document.reference),
    )
    out = [
        f"- model_id: {row.model_id}\n",
        f"  wire_model_id: {row.wire_model_id}\n",
    ]
    if profile:
        out.append(f"  profile: {profile}\n")
    if not fitting:
        budgets = row.thinking_budget or _axis(document.reference, "budget")
        outputs = _axis(document.reference, "output")
        out += [
            "  thinking:\n",
            f"    effort: {_flow(row.thinking_effort)}\n",
            f"    budget: {_flow(budgets)}\n",
            f"    output: {_flow(outputs)}\n",
        ]
    out += _defaults(row, layer=layer)
    out.append(f"  knowledge_cutoff: '{row.knowledge_cutoff}'\n")
    rows = _rows(root, lines)
    place = _place(rows, model_id=row.model_id)
    below = _window(layer)
    limits: dict[str, object] = {**below, **row.limits}
    if limits != below:
        out += _limits(lines, rows, document, limits=limits, place=place)
    out += _prices(row)
    return _inserted(lines, root, out, place=place)


def _defaults(row: NewRow, *, layer: Mapping[str, object]) -> list[str]:
    """Return the lines stating each default of ``row`` that ``layer`` does not."""
    out: list[str] = []
    if row.thinking_off_max_effort not in (None, layer.get("thinking_off_max_effort")):
        out.append(f"  thinking_off_max_effort: {row.thinking_off_max_effort}\n")
    efforts: tuple[ThinkingEffort, ...] = literal_members(ThinkingEffort)
    if row.thinking_effort_default != _starts_at(
        layer,
        "thinking_effort_default",
        offered=row.thinking_effort,
        ladder=efforts,
    ):
        out.append(f"  thinking_effort_default: {row.thinking_effort_default}\n")
    modes: tuple[ThinkingBudget, ...] = literal_members(ThinkingBudget)
    if row.thinking_budget_default not in (
        None,
        _starts_at(
            layer,
            "thinking_budget_default",
            offered=row.thinking_budget or (),
            ladder=modes,
        ),
    ):
        out.append(f"  thinking_budget_default: {row.thinking_budget_default}\n")
    return out


# Above the first older row of the family, or below its last; a new family goes
# last, below every anchor a window of its may alias.
def _place(rows: Sequence[_Row], *, model_id: str) -> tuple[_Row | None, bool]:
    """Return the row a new one goes next to, and whether it goes above that row."""
    family, version = _versioned(model_id)
    kin = [r for r in rows if _versioned(r.model_id)[0] == family]
    older = next((r for r in kin if _versioned(r.model_id)[1] < version), None)
    if older is not None:
        return older, True
    if kin:
        return kin[-1], False
    return (rows[-1] if rows else None), False


def _limits(
    lines: list[str],
    rows: Sequence[_Row],
    document: CatalogDocument,
    *,
    limits: Mapping[str, object],
    place: tuple[_Row | None, bool],
) -> list[str]:
    """Return a new row's ``limits``: an alias of an equal anchored window before it."""
    anchors = {
        position(token).line: token.value
        for token in yaml.scan("".join(lines))
        if isinstance(token, yaml.AnchorToken)
    }
    windows = {
        str(model.get("model_id")): model.get("limits") for model in document.models
    }
    near, above = place
    before = (
        rows[: rows.index(near) + (0 if above else 1)] if near is not None else rows
    )
    for owner in before:
        key = _key(owner.node, "limits")
        if (
            key is not None
            and position(key).line in anchors
            and windows.get(owner.model_id) == limits
        ):
            return [f"  limits: *{anchors[position(key).line]}\n"]
    return ["  limits:\n", *(f"    {k}: {_scalar(v)}\n" for k, v in limits.items())]


def _prices(row: NewRow) -> list[str]:
    """Return a new row's ``prices``, each card's rates as YAML spells floats."""
    out: list[str] = ["  prices:\n"]
    for price in row.prices:
        out.append(f"  - service_tier: {price.service_tier}\n")
        if price.effective_date:
            out.append(f"    effective_date: '{price.effective_date}'\n")
        if price.effective_prompt_tokens:
            out.append(
                f"    effective_prompt_tokens: {price.effective_prompt_tokens:_}\n",
            )
        rates = {
            "request": price.request,
            "response": price.response,
            "cache_write": price.cache_write,
            "cache_write_1h": price.cache_write_1h,
            "cache_read": price.cache_read,
        }
        out += [f"    {meter}: {_spelled(rate)}\n" for meter, rate in rates.items()]
    return out


def _inserted(
    lines: list[str],
    root: yaml.MappingNode,
    out: list[str],
    *,
    place: tuple[_Row | None, bool],
) -> list[str]:
    """Return ``lines`` with the row ``out`` next to the row ``place`` names."""
    near, above = place
    if near is None:
        # ``models: []`` has no block to join: the key opens one instead.
        models = _get(root, "models")
        assert isinstance(models, yaml.SequenceNode)
        line = position(models).line
        return [*lines[:line], "models:\n", *out, "\n", *lines[line + 1 :]]
    if above:
        return [*lines[: near.head], *out, "\n", *lines[near.head :]]
    return [*lines[: near.end + 1], "\n", *out, *lines[near.end + 1 :]]


def _fits(layer: Mapping[str, object], row: NewRow) -> bool:
    """Whether a row under ``layer`` takes ``row``'s levels, and its budget modes."""
    if set(_axis(layer, "effort")) != set(row.thinking_effort):
        return False
    return row.thinking_budget is None or set(_axis(layer, "budget")) == set(
        row.thinking_budget,
    )


def _axis(layer: Mapping[str, object], axis: str) -> list[str]:
    """Return one axis of a layer's ``thinking``; ``[none]`` where it names none."""
    thinking = from_plain(layer.get("thinking", {}), dict[str, object])
    return from_plain(thinking.get(axis, ["none"]), list[str])


def _window(layer: Mapping[str, object]) -> dict[str, object]:
    """Return a layer's ``limits``, empty where it states none."""
    return from_plain(layer.get("limits", {}), dict[str, object])


def _served(
    lines: list[str],
    root: yaml.MappingNode,
    *,
    section: str,
    model_id: str,
    serve: bool,
    why: str,
) -> list[str] | None:
    """Return ``lines`` with ``section`` serving ``model_id``, or omitting it for ``why``."""
    found = _get(_get(root, "providers"), section)
    if not isinstance(found, yaml.MappingNode) or not found.value:
        return None
    omits = _get(found, "omits")
    entries = omits.value if isinstance(omits, yaml.MappingNode) else []
    if serve:
        removed = [(k, v) for k, v in entries if _spells(k, model_id)]
        # Every line of an omit goes: a reason may run over several.
        gone = {
            line
            for k, v in removed
            for line in range(position(k).line, position(v, end=True).line + 1)
        }
        opened = _key(found, "omits")
        if removed and len(removed) == len(entries) and opened is not None:
            gone.add(position(opened).line)
        return [line for i, line in enumerate(lines) if i not in gone]
    key_column = position(found.value[0][0]).column
    if entries:
        last = position(entries[-1][1], end=True).line
        column = position(entries[0][0]).column
        return [*lines[: last + 1], _entry(model_id, why, column), *lines[last + 1 :]]
    models = _key(found, "models")
    at = (
        position(models).line
        if models is not None
        else max(position(leaf, end=True).line for leaf in _leaves(found)) + 1
    )
    opened = f"{' ' * key_column}omits:\n"
    return [*lines[:at], opened, _entry(model_id, why, key_column + 2), *lines[at:]]


def _entry(model_id: str, why: str, column: int) -> str:
    """Return one ``omits`` line at ``column``."""
    return f"{' ' * column}{model_id}: {_spelled(why)}\n"


def _starts_at(
    layer: Mapping[str, object],
    key: str,
    *,
    offered: Collection[str],
    ladder: Sequence[str],
) -> object:
    """Return what a row of ``layer`` holds at ``key``: stated, else the lowest offered."""
    stated = layer.get(key)
    if stated is not None:
        return stated
    return next((value for value in ladder if value in offered), None)


def _versioned(model_id: str) -> tuple[str, tuple[int, ...]]:
    """Split ``sonnet-4.6`` into ``("sonnet", (4, 6))``."""
    family, _, version = model_id.rpartition("-")
    return family, tuple(int(part) for part in version.split(".") if part.isdecimal())


def _flow(values: Sequence[str]) -> str:
    """Return a list as YAML's flow form: ``[low, high]``."""
    return f"[{', '.join(values)}]"


def _scalar(value: object) -> str:
    """Return a limit as the catalogs spell it: ``1_048_576``, ``true``."""
    return str(value).lower() if isinstance(value, bool) else f"{value:_}"


def _spelled(value: float | str) -> str:
    """Return ``value`` as YAML's own representer spells it: ``1.0e-05``, a quoted ``a: b``."""
    # Unbounded width: a long reason is one line, never folded onto the next.
    return yaml.safe_dump(value, width=sys.maxsize).split("\n")[0]


def _value(field: str, source: str) -> str:
    """Return ``source`` spelled as the document spells the field."""
    if field == "knowledge_cutoff":
        return f"'{source}'"
    if _is_limit(field):
        return f"{int(source):_}"
    return _spelled(float(source))
