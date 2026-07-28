"""Tests for ``tools.skill``: discovery + invocation of user-authored skills."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.testing import with_fake_agent
from sagent.tools import skill as sk
from sagent.tools.core import ToolState
from sagent.tools.skill import (
    Skill,
    SkillInfo,
    discover,
    format_listing,
)
from sagent.types.runtime import (
    ModelContextEvent,
    ToolResult,
    UserMessage,
)


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "Use when foo.",
    body: str = "Do the foo.\n",
    metadata_name: str | None = None,
) -> Path:
    """Write a SKILL.md under ``root/.sagent/skills/<name>/``."""
    skill_dir = root / ".sagent" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    meta_name = metadata_name if metadata_name is not None else name
    text = f"---\nname: {meta_name}\ndescription: {description}\n---\n\n{body}"
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    return skill_dir / "SKILL.md"


@pytest.fixture(autouse=True)
def _isolate_user_skills(  # pyright: ignore[reportUnusedFunction] -- autouse fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point user-skill discovery at an empty tmp tree.

    Many devs have ``~/.sagent/skills`` populated; tests must not see
    those, or assertions about "exactly N skills" will flake.
    """
    monkeypatch.setattr(sk, "_USER_SKILL_ROOTS", ())


def test_discover_finds_project_skill(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", description="Trigger A")
    out = discover(tmp_path)
    assert len(out) == 1
    s = out[0]
    assert s.name == "alpha"
    assert s.description == "Trigger A"
    assert s.source == "project"


def test_discover_dedupes_by_name(tmp_path: Path) -> None:
    project = tmp_path / "child"
    project.mkdir()
    _write_skill(tmp_path, "shared", description="root")
    _write_skill(project, "shared", description="child")
    out = discover(project)
    # ``discover`` iterates ``reversed(walk_up(cwd))`` -- closest dir
    # first. Dedup keeps the first occurrence, so ``child`` wins.
    assert len(out) == 1
    assert out[0].description == "child"


def test_discover_skips_invalid_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_skill(tmp_path, "ok-name", metadata_name="Bad!Name")
    with caplog.at_level("WARNING"):
        out = discover(tmp_path)
    assert out == []


def test_discover_ignores_missing_skill_md(tmp_path: Path) -> None:
    # Skill dir exists but no SKILL.md inside.
    (tmp_path / ".sagent" / "skills" / "empty").mkdir(parents=True)
    out = discover(tmp_path)
    assert out == []


def test_discover_ignores_files_inside_root(tmp_path: Path) -> None:
    root = tmp_path / ".sagent" / "skills"
    root.mkdir(parents=True)
    (root / "stray.txt").write_text("ignored", encoding="utf-8")
    out = discover(tmp_path)
    assert out == []


def test_discover_finds_nested_child_skill(tmp_path: Path) -> None:
    # Parent skill ``trax`` with a nested child doc ``trax/paper/SKILL.md``
    # whose frontmatter name is ``trax-paper``. Both must register.
    parent = _write_skill(tmp_path, "trax", description="parent").parent
    child = parent / "paper"
    child.mkdir()
    (child / "SKILL.md").write_text(
        "---\nname: trax-paper\ndescription: author a Paper\n---\n\nBody.\n",
        encoding="utf-8",
    )
    out = discover(tmp_path)
    names = {s.name for s in out}
    assert "trax" in names
    assert "trax-paper" in names


def test_discover_nested_survives_symlink_cycle(tmp_path: Path) -> None:
    # ``.claude`` -> ``.sagent`` style self-link must not loop forever.
    parent = _write_skill(tmp_path, "trax", description="parent").parent
    loop_link = parent / "self"
    try:
        loop_link.symlink_to(tmp_path / ".sagent" / "skills")
    except OSError:
        pytest.skip("symlinks unsupported on this platform")
    out = discover(tmp_path)
    assert "trax" in {s.name for s in out}


def test_discover_falls_back_to_first_nonempty_line(tmp_path: Path) -> None:
    """When frontmatter lacks ``description``, the first non-header line wins."""
    skill_dir = tmp_path / ".sagent" / "skills" / "alpha"
    skill_dir.mkdir(parents=True)
    text = "---\nname: alpha\n---\n\n# heading\n\nReal description line.\n"
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    out = discover(tmp_path)
    assert len(out) == 1
    assert out[0].description == "Real description line."


def test_format_listing_empty_returns_blank() -> None:
    assert format_listing([]) == ""


def test_format_listing_truncates_long_description() -> None:
    info = SkillInfo(
        name="x",
        description="a" * 300,
        body="body",
        source="project",
        path=Path("/x/SKILL.md"),
    )
    out = format_listing([info])
    assert "x" in out
    assert "..." in out


def test_format_listing_uses_placeholder_for_missing_description() -> None:
    info = SkillInfo(
        name="x",
        description="",
        body="body",
        source="project",
        path=Path("/x/SKILL.md"),
    )
    out = format_listing([info])
    assert "(no description)" in out


def test_metadata_basics() -> None:
    t = Skill()
    assert t.name == "Skill"
    assert t.tool_id == "application/x-tool-skill"
    assert t.clearable_results is False
    assert t.summary({"skill": "alpha"}) == "Skill alpha"
    assert t.summary({}) == "Skill"
    assert t.summary_result(ToolResult(call_id="", content="")) is None


def test_prompt_lists_discovered_skills(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", description="Use when alpha")
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    t = Skill()
    with with_fake_agent(tool_state=state):
        out = t.prompt()
    assert "alpha" in out
    assert "Use when alpha" in out


def test_prompt_empty_without_skills(tmp_path: Path) -> None:
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    t = Skill()
    with with_fake_agent(tool_state=state):
        out = t.prompt()
    assert out == ""


@pytest.mark.asyncio
async def test_run_unknown_skill(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", description="A")
    t = Skill()
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    with with_fake_agent(tool_state=state):
        result = await t.run({"skill": "bogus"})
    assert result.is_error
    assert "Unknown skill" in result.content
    assert "alpha" in result.content


@pytest.mark.asyncio
async def test_run_loads_body_and_marks_invoked(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", body="hello <world>\n")
    t = Skill()
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    with with_fake_agent(tool_state=state):
        result = await t.run({"skill": "alpha"})
    assert not result.is_error
    # Body is html-escaped before insertion.
    assert "hello &lt;world&gt;" in result.content
    assert "<skill name='alpha'" in result.content
    assert "alpha" in state.invoked_skills


@pytest.mark.asyncio
async def test_run_short_circuits_on_second_invocation(tmp_path: Path) -> None:
    """Second call returns a stub; the body is not re-emitted."""
    _write_skill(tmp_path, "alpha", body="full body content\n")
    t = Skill()
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    with with_fake_agent(tool_state=state):
        first = await t.run({"skill": "alpha"})
        second = await t.run({"skill": "alpha"})
    assert "full body content" in first.content
    assert "full body content" not in second.content
    assert "already loaded earlier" in second.content
    assert state.invoked_skills == {"alpha"}


@pytest.mark.asyncio
async def test_run_reloads_body_after_reset_tool_recall(tmp_path: Path) -> None:
    """``reset_tool_recall`` re-arms the short-circuit so body re-emits."""
    _write_skill(tmp_path, "alpha", body="full body content\n")
    t = Skill()
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    with with_fake_agent(tool_state=state):
        first = await t.run({"skill": "alpha"})
        state.reset_tool_recall()
        second = await t.run({"skill": "alpha"})
    assert "full body content" in first.content
    assert "full body content" in second.content
    assert "already loaded earlier" not in second.content


@pytest.mark.asyncio
async def test_run_appends_args(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", body="body\n")
    t = Skill()
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    with with_fake_agent(tool_state=state):
        result = await t.run({"skill": "alpha", "args": "<go>"})
    assert not result.is_error
    assert "Arguments: &lt;go&gt;" in result.content


@pytest.mark.asyncio
async def test_post_compact_restore_noop_without_invoked(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", body="b")
    t = Skill()
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    await t.post_compact_restore(history, state)
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert entry.text == "hi"


@pytest.mark.asyncio
async def test_post_compact_restore_default_disabled(tmp_path: Path) -> None:
    """Default ``restore_after_compact=False`` drops bodies on compact."""
    _write_skill(tmp_path, "alpha", body="instr-body")
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.invoked_skills.add("alpha")
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    await Skill().post_compact_restore(history, state)
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert entry.text == "hi"


@pytest.mark.asyncio
async def test_post_compact_restore_reattaches_into_first_user(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", body="instr-body")
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.invoked_skills.add("alpha")
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    await Skill(restore_after_compact=True).post_compact_restore(history, state)
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert "instr-body" in entry.text
    assert entry.text.endswith("hi")


@pytest.mark.asyncio
async def test_post_compact_restore_skips_when_cwd_unset(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", body="b")
    state = ToolState()
    state.bash_cwd = ""
    state.invoked_skills.add("alpha")
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    await Skill(restore_after_compact=True).post_compact_restore(history, state)
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert entry.text == "hi"


@pytest.mark.asyncio
async def test_post_compact_restore_truncates_huge_body(tmp_path: Path) -> None:
    huge = "x" * (Skill._MAX_CHARS_PER_SKILL + 100)
    _write_skill(tmp_path, "alpha", body=huge)
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.invoked_skills.add("alpha")
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    await Skill(restore_after_compact=True).post_compact_restore(history, state)
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert "(truncated)" in entry.text


@pytest.mark.asyncio
async def test_post_compact_restore_budget_caps_total(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", body="A" * 500)
    _write_skill(tmp_path, "beta", body="B" * 500)
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.invoked_skills.update({"alpha", "beta"})
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    # Budget caps total at ~one body; the second body is skipped.
    await Skill(restore_after_compact=True).post_compact_restore(
        history,
        state,
        budget_chars=700,
    )
    entry = history[0]
    assert isinstance(entry, UserMessage)
    # Exactly one of the two skill bodies survives.
    has_alpha = "A" * 500 in entry.text
    has_beta = "B" * 500 in entry.text
    assert has_alpha != has_beta


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
