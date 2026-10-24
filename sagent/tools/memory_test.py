"""Tests for ``tools.memory``: the Memory tool saves and forgets memories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.agent.state import ToolState, tool_state_context
from sagent.lib.dotsagent import parse_frontmatter
from sagent.memory import memory_dir
from sagent.tools.memory import Memory


if TYPE_CHECKING:
    from pathlib import Path


def _state(tmp_path: Path) -> ToolState:
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    return state


@pytest.mark.asyncio
async def test_save_writes_the_memory_and_its_index_line(tmp_path: Path) -> None:
    with tool_state_context(_state(tmp_path)):
        result = await Memory().run(
            {
                "action": "save",
                "type": "feedback",
                "name": "Testing approach",
                "description": "Prefer real files over mocks",
                "content": "Use tmp_path.\n\n**Why:** mocks hid a bug.",
            },
        )
    assert not result.is_error
    directory = memory_dir(str(tmp_path))
    saved = (directory / "feedback_testing_approach.md").read_text()
    assert saved.startswith("---\nname: Testing approach\n")
    assert "type: feedback\n" in saved
    assert saved.endswith("Use tmp_path.\n\n**Why:** mocks hid a bug.\n")
    assert (directory / "MEMORY.md").read_text() == (
        "- [Testing approach](feedback_testing_approach.md) -- Prefer real files"
        " over mocks\n"
    )


@pytest.mark.asyncio
async def test_saving_a_name_again_replaces_its_index_line(tmp_path: Path) -> None:
    memory = Memory()
    with tool_state_context(_state(tmp_path)):
        for description in ("first", "second"):
            _ = await memory.run(
                {
                    "action": "save",
                    "type": "user",
                    "name": "role",
                    "description": description,
                    "content": "x",
                },
            )
    index = (memory_dir(str(tmp_path)) / "MEMORY.md").read_text()
    assert index == "- [role](user_role.md) -- second\n"


@pytest.mark.asyncio
async def test_forget_removes_the_memory_and_its_index_line(tmp_path: Path) -> None:
    memory = Memory()
    with tool_state_context(_state(tmp_path)):
        for name in ("role", "team"):
            _ = await memory.run(
                {
                    "action": "save",
                    "type": "user",
                    "name": name,
                    "description": name,
                    "content": "x",
                },
            )
        result = await memory.run({"action": "forget", "name": "role"})
    directory = memory_dir(str(tmp_path))
    assert not result.is_error
    assert not (directory / "user_role.md").exists()
    assert (directory / "MEMORY.md").read_text() == "- [team](user_team.md) -- team\n"


@pytest.mark.asyncio
async def test_forgetting_an_unknown_memory_is_an_error(tmp_path: Path) -> None:
    with tool_state_context(_state(tmp_path)):
        result = await Memory().run({"action": "forget", "name": "nothing"})
    assert result.is_error
    assert "nothing" in result.content


@pytest.mark.asyncio
async def test_a_save_without_a_known_type_is_an_error(tmp_path: Path) -> None:
    with tool_state_context(_state(tmp_path)):
        result = await Memory().run(
            {
                "action": "save",
                "type": "misc",
                "name": "x",
                "description": "x",
                "content": "x",
            },
        )
    assert result.is_error
    assert not memory_dir(str(tmp_path)).joinpath("MEMORY.md").exists()


async def _save(memory: Memory, name: str, *, kind: str = "feedback") -> None:
    """Save ``name`` with placeholder text."""
    result = await memory.run(
        {
            "action": "save",
            "type": kind,
            "name": name,
            "description": name,
            "content": "x",
        },
    )
    assert not result.is_error, result.content


@pytest.mark.asyncio
async def test_a_save_leaves_a_memory_whose_name_ends_the_same_way(
    tmp_path: Path,
) -> None:
    """``*_testing.md`` also matched ``feedback_unit_testing.md``, and deleted it."""
    memory = Memory()
    with tool_state_context(_state(tmp_path)):
        await _save(memory, "unit testing")
        await _save(memory, "testing")
    directory = memory_dir(str(tmp_path))
    assert (directory / "feedback_unit_testing.md").exists()
    assert (directory / "feedback_testing.md").exists()
    assert (directory / "MEMORY.md").read_text() == (
        "- [testing](feedback_testing.md) -- testing\n"
        "- [unit testing](feedback_unit_testing.md) -- unit testing\n"
    )


@pytest.mark.asyncio
async def test_forget_leaves_a_memory_whose_name_ends_the_same_way(
    tmp_path: Path,
) -> None:
    memory = Memory()
    with tool_state_context(_state(tmp_path)):
        await _save(memory, "unit testing")
        await _save(memory, "testing")
        result = await memory.run({"action": "forget", "name": "testing"})
    directory = memory_dir(str(tmp_path))
    assert result.content == "Forgot feedback_testing.md."
    assert (directory / "feedback_unit_testing.md").exists()
    assert (directory / "MEMORY.md").read_text() == (
        "- [unit testing](feedback_unit_testing.md) -- unit testing\n"
    )


@pytest.mark.asyncio
async def test_saving_a_name_as_another_type_moves_it(tmp_path: Path) -> None:
    memory = Memory()
    with tool_state_context(_state(tmp_path)):
        await _save(memory, "role", kind="user")
        await _save(memory, "role", kind="project")
    directory = memory_dir(str(tmp_path))
    assert not (directory / "user_role.md").exists()
    assert (directory / "MEMORY.md").read_text() == (
        "- [role](project_role.md) -- role\n"
    )


@pytest.mark.asyncio
async def test_a_saved_memory_is_one_index_line(tmp_path: Path) -> None:
    """A newline in a description split its index line in two."""
    with tool_state_context(_state(tmp_path)):
        result = await Memory().run(
            {
                "action": "save",
                "type": "user",
                "name": "role\nlead",
                "description": "Leads\n  the team",
                "content": "x",
            },
        )
    assert not result.is_error
    directory = memory_dir(str(tmp_path))
    assert (directory / "MEMORY.md").read_text() == (
        "- [role lead](user_role_lead.md) -- Leads the team\n"
    )
    assert (
        "name: role lead\ndescription: Leads the team\n"
        in (directory / "user_role_lead.md").read_text()
    )


@pytest.mark.asyncio
async def test_a_name_with_a_bracket_is_an_error(tmp_path: Path) -> None:
    """``]`` closes the index link early, so the line no longer names its file."""
    with tool_state_context(_state(tmp_path)):
        result = await Memory().run(
            {
                "action": "save",
                "type": "user",
                "name": "role [lead]",
                "description": "d",
                "content": "x",
            },
        )
    assert result.is_error
    assert result.content == "name must not contain [ or ]."


@pytest.mark.asyncio
async def test_the_newest_memory_heads_the_index(tmp_path: Path) -> None:
    """The index is cut at its cap from the end, so the newest must come first."""
    memory = Memory()
    with tool_state_context(_state(tmp_path)):
        await _save(memory, "older")
        await _save(memory, "newer")
    assert (
        (memory_dir(str(tmp_path)) / "MEMORY.md")
        .read_text()
        .startswith(
            "- [newer]",
        )
    )


@pytest.mark.asyncio
async def test_a_save_goes_where_the_agent_was_shown_its_memories(
    tmp_path: Path,
) -> None:
    """After a ``cd`` the index the model reads is still the project's it started in."""
    state = _state(tmp_path / "project")
    state.project_cwd = state.bash_cwd
    state.bash_cwd = str(tmp_path / "elsewhere")
    with tool_state_context(state):
        await _save(Memory(), "role")
    assert (memory_dir(str(tmp_path / "project")) / "feedback_role.md").exists()
    assert not memory_dir(str(tmp_path / "elsewhere")).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("args", "error"),
    [
        ({"action": "save", "name": "!!"}, "name must contain a letter or digit."),
        (
            {"action": "save", "name": "x", "type": "user", "content": "c"},
            "a save needs a description and content.",
        ),
    ],
)
async def test_a_save_missing_what_it_needs_is_an_error(
    tmp_path: Path,
    args: dict[str, object],
    error: str,
) -> None:
    with tool_state_context(_state(tmp_path)):
        result = await Memory().run(args)
    assert result.is_error
    assert result.content == error


def test_the_description_teaches_when_to_save_and_names_no_directory() -> None:
    """Fixed text: the directory is per project, so it lives in the project block."""
    memory = Memory()
    assert "## Memory types" in memory.description
    assert "What NOT to save" in memory.description
    assert "/memory" not in memory.description
    assert memory.prompt() == ""


@pytest.mark.asyncio
async def test_a_saved_memorys_frontmatter_reads_back_as_written(
    tmp_path: Path,
) -> None:
    """A colon or a leading ``#`` in a plain YAML value broke the whole block."""
    args = {
        "action": "save",
        "type": "project",
        "name": "Deploy: staging",
        "description": "#1 rule: never force-push",
        "content": "x",
    }
    with tool_state_context(_state(tmp_path)):
        result = await Memory().run(args)
    assert not result.is_error, result.content
    raw = (memory_dir(str(tmp_path)) / "project_deploy_staging.md").read_text()
    meta, body = parse_frontmatter(raw)
    assert meta == {
        "name": "Deploy: staging",
        "description": "#1 rule: never force-push",
        "type": "project",
    }
    assert body == "\nx\n"


@pytest.mark.asyncio
async def test_a_name_that_files_like_another_is_refused_not_overwritten(
    tmp_path: Path,
) -> None:
    """``foo-bar`` and ``Foo Bar`` share a file; the second save erased the first."""
    memory = Memory()
    with tool_state_context(_state(tmp_path)):
        await _save(memory, "Foo Bar")
        saved = await memory.run(
            {
                "action": "save",
                "type": "user",
                "name": "foo-bar",
                "description": "d",
                "content": "y",
            },
        )
        forgot = await memory.run({"action": "forget", "name": "foo-bar"})
    assert saved.is_error
    assert "Foo Bar" in saved.content
    assert forgot.is_error
    directory = memory_dir(str(tmp_path))
    assert [p.name for p in directory.glob("*_foo_bar.md")] == ["feedback_foo_bar.md"]
    assert (directory / "MEMORY.md").read_text().startswith("- [Foo Bar]")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
