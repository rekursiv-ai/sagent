"""Tests for the Memory tool: memory files, their ``MEMORY.md`` index, and its caps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import stat

import pytest
import yaml

from sagent.tools.memory import Memory
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree

    from sagent.session.types import UncategorizedToolResult


def test_summary_names_the_action_and_the_memory(tmp_path: Path) -> None:
    memory = Memory(directory=tmp_path)
    assert memory.summary({"action": "save", "name": "role"}) == "Memory save role"
    assert memory.summary({"action": "forget"}) == "Memory forget"
    assert memory.summary({}) == "Memory"


def test_every_call_queues_behind_the_index_it_rewrites(tmp_path: Path) -> None:
    memory = Memory(directory=tmp_path / "m")
    key = memory.serialize_key(ToolContext(cwd=tmp_path), {"action": "save"})
    assert key == str(tmp_path / "m" / "MEMORY.md")


def test_the_description_teaches_saving_and_never_changes(tmp_path: Path) -> None:
    """The description is in the tools' cached prefix, so it says no index."""
    (tmp_path / "MEMORY.md").write_text("- [role](user_role.md) -- Leads the team\n")
    description = Memory(directory=tmp_path).description
    assert "## Memory types" in description
    assert "What NOT to save" in description
    assert "Leads the team" not in description
    assert description == Memory(directory=tmp_path / "other").description


def test_the_briefing_says_where_the_memories_are_and_carries_the_index(
    tmp_path: Path,
) -> None:
    (tmp_path / "MEMORY.md").write_text("- [role](user_role.md) -- Leads the team\n")
    assert Memory(directory=tmp_path).briefing(ToolContext(cwd=tmp_path)) == (
        f"## MEMORY.md\n\nThe memories live at `{tmp_path}/`; Read one when it"
        " seems relevant.\n\n- [role](user_role.md) -- Leads the team"
    )


def test_a_project_without_memories_says_so(tmp_path: Path) -> None:
    briefing = Memory(directory=tmp_path / "none").briefing(ToolContext(cwd=tmp_path))
    assert briefing.endswith("(no memories yet)")


@pytest.mark.asyncio
async def test_a_save_writes_the_memory_and_heads_the_index_with_it(
    tmp_path: Path,
) -> None:
    memory = Memory(directory=tmp_path / "m")
    first = await _save(memory, "older", kind="user", description="the first")
    second = await _save(memory, "newer", description="the second", content="  body  ")
    assert (first.text(), second.text()) == (
        "Saved user_older.md.",
        "Saved feedback_newer.md.",
    )
    assert (tmp_path / "m" / "feedback_newer.md").read_text() == (
        "---\nname: newer\ndescription: the second\ntype: feedback\n---\n\nbody\n"
    )
    assert (tmp_path / "m" / "MEMORY.md").read_text() == (
        "- [newer](feedback_newer.md) -- the second\n"
        "- [older](user_older.md) -- the first\n"
    )
    assert stat.S_IMODE((tmp_path / "m").stat().st_mode) == 0o700


@pytest.mark.asyncio
async def test_the_briefing_reads_the_index_as_it_is_when_asked(tmp_path: Path) -> None:
    """The agent asks once a window; a save shows from the next one on."""
    memory = Memory(directory=tmp_path)
    context = ToolContext(cwd=tmp_path)
    assert memory.briefing(context).endswith("(no memories yet)")
    await _save(memory, "role")
    assert memory.briefing(context).endswith("- [role](feedback_role.md) -- role")


@pytest.mark.asyncio
async def test_saving_a_name_again_replaces_it_and_may_change_its_type(
    tmp_path: Path,
) -> None:
    memory = Memory(directory=tmp_path)
    await _save(memory, "role", kind="user", description="old")
    await _save(memory, "other")
    await _save(memory, "role", kind="project", description="new")
    assert not (tmp_path / "user_role.md").exists()
    assert (tmp_path / "project_role.md").exists()
    assert (tmp_path / "MEMORY.md").read_text() == (
        "- [role](project_role.md) -- new\n- [other](feedback_other.md) -- other\n"
    )


@pytest.mark.asyncio
async def test_a_memory_is_one_index_line_whatever_its_whitespace(
    tmp_path: Path,
) -> None:
    memory = Memory(directory=tmp_path)
    await _save(memory, " role\n lead ", kind="user", description="Leads\n  the team")
    assert (tmp_path / "MEMORY.md").read_text() == (
        "- [role lead](user_role_lead.md) -- Leads the team\n"
    )
    assert "name: role lead\ndescription: Leads the team\n" in (
        (tmp_path / "user_role_lead.md").read_text()
    )


@pytest.mark.asyncio
async def test_a_memorys_frontmatter_reads_back_as_written(tmp_path: Path) -> None:
    """A colon or a leading ``#`` in a plain YAML value breaks the whole block."""
    memory = Memory(directory=tmp_path)
    await _save(
        memory,
        "Deploy: staging",
        kind="project",
        description="#1 rule: never force-push",
    )
    raw = (tmp_path / "project_deploy_staging.md").read_text()
    _, block, body = raw.split("---\n", 2)
    assert yaml.safe_load(block) == {
        "name": "Deploy: staging",
        "description": "#1 rule: never force-push",
        "type": "project",
    }
    assert body == "\nx\n"


@pytest.mark.asyncio
async def test_saving_and_forgetting_leave_a_memory_whose_name_ends_the_same(
    tmp_path: Path,
) -> None:
    """A glob of ``*_testing.md`` would match ``feedback_unit_testing.md`` too."""
    memory = Memory(directory=tmp_path)
    await _save(memory, "unit testing")
    await _save(memory, "testing")
    assert (tmp_path / "feedback_unit_testing.md").exists()
    forgot = await memory.call(
        ToolContext(cwd=tmp_path),
        {"action": "forget", "name": "testing"},
    )
    assert forgot.text() == "Forgot feedback_testing.md."
    assert forgot.is_error is None
    assert (tmp_path / "feedback_unit_testing.md").exists()
    assert (tmp_path / "MEMORY.md").read_text() == (
        "- [unit testing](feedback_unit_testing.md) -- unit testing\n"
    )


@pytest.mark.asyncio
async def test_forget_removes_every_type_of_a_name_and_keeps_other_lines(
    tmp_path: Path,
) -> None:
    memory = Memory(directory=tmp_path)
    await _save(memory, "keep")
    await _save(memory, "role", kind="user")
    (tmp_path / "reference_role.md").write_text("---\nname: role\n---\n")
    with (tmp_path / "MEMORY.md").open("a") as index:
        index.write("\n  \nhand-written line\n")
    result = await memory.call(
        ToolContext(cwd=tmp_path),
        {"action": "forget", "name": "role"},
    )
    assert result.text() == "Forgot user_role.md, reference_role.md."
    assert _names(tmp_path) == [
        "MEMORY.md",
        "feedback_keep.md",
    ]
    assert (tmp_path / "MEMORY.md").read_text() == (
        "- [keep](feedback_keep.md) -- keep\nhand-written line\n"
    )


@pytest.mark.asyncio
async def test_a_name_that_files_like_another_is_refused_not_overwritten(
    tmp_path: Path,
) -> None:
    """``foo-bar`` and ``Foo Bar`` share a file, so the second would erase the first."""
    memory = Memory(directory=tmp_path)
    await _save(memory, "Foo Bar")
    saved = await _save(memory, "foo-bar", content="y")
    forgot = await memory.call(
        ToolContext(cwd=tmp_path),
        {"action": "forget", "name": "foo-bar"},
    )
    assert saved.is_error is True
    assert saved.text() == (
        "'foo-bar' files as the memory 'Foo Bar'; save it under that name, or another."
    )
    assert forgot.is_error is True
    assert forgot.text() == (
        "'foo-bar' files as the memory 'Foo Bar'; forget it under that name."
    )
    assert (tmp_path / "feedback_foo_bar.md").read_text().endswith("\nx\n")
    assert (tmp_path / "MEMORY.md").read_text().startswith("- [Foo Bar]")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "held",
    [
        b"written by hand\n",
        b"---\nname: Deploy: staging\n---\n\nx\n",
        b"---\n- a list\n---\n\nx\n",
        b"---\nname: [1]\n---\n\nx\n",
        b"\xff\xfe",
    ],
    ids=["no-frontmatter", "bad-yaml", "not-a-mapping", "no-string-name", "not-utf8"],
)
async def test_a_file_that_names_no_memory_is_replaced_by_its_name(
    tmp_path: Path,
    held: bytes,
) -> None:
    (tmp_path / "user_role.md").write_bytes(held)
    result = await _save(Memory(directory=tmp_path), "role", kind="user")
    assert result.is_error is None, result.text()
    assert "name: role\n" in (tmp_path / "user_role.md").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"action": "save", "name": "role [lead]"}, "name must not contain [ or ]."),
        ({"action": "save", "name": "!!"}, "name must contain a letter or digit."),
        ({"action": "forget", "name": "  "}, "name must contain a letter or digit."),
        ({"action": "forget", "name": "absent"}, "No memory named 'absent'."),
        (
            {"action": "save", "name": "x", "description": "d", "content": "c"},
            "A save needs a type, a description, and content.",
        ),
        (
            {
                "action": "save",
                "name": "x",
                "type": "user",
                "description": " ",
                "content": "c",
            },
            "A save needs a type, a description, and content.",
        ),
        (
            {
                "action": "save",
                "name": "x",
                "type": "user",
                "description": "d",
                "content": "\n",
            },
            "A save needs a type, a description, and content.",
        ),
    ],
)
async def test_a_call_that_names_no_memory_or_lacks_one_is_an_error(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    message: str,
) -> None:
    result = await Memory(directory=tmp_path).call(ToolContext(cwd=tmp_path), arguments)
    assert result.is_error is True
    assert result.call_id == ""
    assert result.text() == message
    assert not (tmp_path / "MEMORY.md").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "remember", "name": "x"},
        {"action": "save"},
        {"action": "save", "name": "x", "type": "fact"},
    ],
)
async def test_arguments_that_do_not_fit_are_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    with pytest.raises(ValueError, match=r"argument|must be"):
        await Memory(directory=tmp_path).call(ToolContext(cwd=tmp_path), arguments)


@pytest.mark.parametrize(
    ("index", "lines", "size", "carried"),
    [
        ("a\nb\nc\n", 3, 100, "a\nb\nc"),
        ("a\nb\nc\nd\n", 3, 100, "a\nb\nc\n\n[MEMORY.md truncated: over 3 lines]"),
        ("aaa\nbbb\nccc\n", 9, 9, "aaa\nbbb\n\n[MEMORY.md truncated: over 9 bytes]"),
        ("aaa\nbbb\nccc\n", 9, 12, "aaa\nbbb\nccc"),
        ("aaaaaaaaaaaa\n", 9, 9, "[MEMORY.md truncated: over 9 bytes]"),
        (
            "aaa\nbbb\nccc\nddd\n",
            3,
            6,
            "aaa\n\n[MEMORY.md truncated: over 3 lines; over 6 bytes]",
        ),
    ],
)
def test_the_index_carried_is_cut_at_its_caps_and_says_so(
    tmp_path: Path,
    index: str,
    lines: int,
    size: int,
    carried: str,
) -> None:
    (tmp_path / "MEMORY.md").write_text(index)
    memory = Memory(directory=tmp_path, max_index_lines=lines, max_index_bytes=size)
    assert memory.briefing(ToolContext(cwd=tmp_path)).endswith(
        f"relevant.\n\n{carried}",
    )


@pytest.mark.parametrize("raw", [b"\xff\xfe not utf-8", None])
def test_an_index_that_cannot_be_read_is_carried_as_none(
    tmp_path: Path,
    raw: bytes | None,
) -> None:
    if raw is None:
        (tmp_path / "MEMORY.md").mkdir()
    else:
        (tmp_path / "MEMORY.md").write_bytes(raw)
    briefing = Memory(directory=tmp_path).briefing(ToolContext(cwd=tmp_path))
    assert briefing.endswith("(no memories yet)")


async def _save(
    memory: Memory,
    name: str,
    *,
    kind: str = "feedback",
    description: str = "",
    content: str = "x",
) -> UncategorizedToolResult:
    """Save ``name``; its description defaults to its name."""
    return await memory.call(
        ToolContext(cwd=memory.directory),
        {
            "action": "save",
            "name": name,
            "type": kind,
            "description": description or name,
            "content": content,
        },
    )


def _names(directory: Path) -> list[str]:
    """Return the names in ``directory``, sorted."""
    return sorted(p.name for p in directory.iterdir())


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
