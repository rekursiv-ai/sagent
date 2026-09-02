"""Tests for ``tools.skill``: discovery + invocation of user-authored skills."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.agent.state import ToolState
from sagent.lib.userdirs import data_dir
from sagent.testing import with_fake_agent
from sagent.tools import skill as sk
from sagent.tools.skill import (
    Skill,
    SkillInfo,
    discover,
    format_listing,
)
from sagent.types.runtime import (
    ModelContextEvent,
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
def _isolate_user_skills() -> None:  # pyright: ignore[reportUnusedFunction] -- autouse fixture
    """Assert the autouse XDG isolation actually reaches skill discovery.

    Many devs have a populated user skills dir; tests must not see it, or
    assertions about "exactly N skills" flake. ``isolate_user_dirs``
    already repoints ``XDG_DATA_HOME`` at a tmp root, and
    :func:`user_skill_roots` resolves per call -- so no patching is
    needed. This guard fails loudly if that resolution regresses to an
    import-time constant, which would silently reintroduce the flake.
    """
    assert not sk.user_skill_roots()[0].exists(), (
        "user skill discovery escaped XDG isolation; tests would see real skills"
    )


def test_user_skill_root_follows_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """User-skill discovery must resolve XDG at call time, not import time.

    The autouse ``isolate_user_dirs`` fixture redirects the XDG vars per
    test, but a module-level constant froze its value when the module was
    first imported -- so discovery still reads the operator's real skills
    dir. That is why ``_isolate_user_skills`` has to patch a private
    constant instead of just setting the environment.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    expected = data_dir() / "rekursiv-ai" / "sagent" / "skills"
    assert sk.user_skill_roots() == (expected,), (
        "user skill roots must follow a late XDG_DATA_HOME"
    )


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


def test_format_listing_keeps_long_description() -> None:
    info = SkillInfo(
        name="x",
        description="a" * 300,
        body="body",
        source="project",
        path=Path("/x/SKILL.md"),
    )
    out = format_listing([info])
    assert "x" in out
    # The trigger description is author-authored prompt text, not model
    # output: clipping it hid the very condition it exists to state.
    assert "a" * 300 in out


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


def _estimate(text: str) -> int:
    """Stand-in for the model's tokenizer; four chars to the token."""
    return len(text) // 4


@pytest.mark.asyncio
async def test_post_compact_restore_noop_without_invoked(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", body="b")
    t = Skill()
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    await t.post_compact_restore(
        history, state, budget_tokens=0, estimate_tokens=_estimate
    )
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
    await Skill().post_compact_restore(
        history, state, budget_tokens=0, estimate_tokens=_estimate
    )
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
    await Skill(restore_after_compact=True).post_compact_restore(
        history, state, budget_tokens=0, estimate_tokens=_estimate
    )
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
    await Skill(restore_after_compact=True).post_compact_restore(
        history, state, budget_tokens=0, estimate_tokens=_estimate
    )
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert entry.text == "hi"


@pytest.mark.asyncio
async def test_post_compact_restore_keeps_huge_body_whole(tmp_path: Path) -> None:
    """A skill body is a contract: restore it whole or not at all."""
    huge = "x" * 60_000
    _write_skill(tmp_path, "alpha", body=huge)
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.invoked_skills.add("alpha")
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    await Skill(restore_after_compact=True).post_compact_restore(
        history, state, budget_tokens=0, estimate_tokens=_estimate
    )
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert huge in entry.text
    assert "(truncated)" not in entry.text


@pytest.mark.asyncio
async def test_post_compact_restore_keeps_every_skill(tmp_path: Path) -> None:
    """With no budget (``0``), every invoked skill is restored whole."""
    _write_skill(tmp_path, "alpha", body="A" * 500)
    _write_skill(tmp_path, "beta", body="B" * 500)
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.invoked_skills.update({"alpha", "beta"})
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    await Skill(restore_after_compact=True).post_compact_restore(
        history, state, budget_tokens=0, estimate_tokens=_estimate
    )
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert "A" * 500 in entry.text
    assert "B" * 500 in entry.text
    assert "Not restored" not in entry.text


@pytest.mark.asyncio
async def test_post_compact_restore_honors_the_budget(tmp_path: Path) -> None:
    """The compactor passes real remaining capacity; the hook must respect it.

    ``post_compact_enrich`` computes ``hook_budget`` from the space left
    after compaction. Ignoring it can push the just-compacted request
    straight back over the window -- the condition compaction just ran
    to relieve.
    """
    _write_skill(tmp_path, "alpha", body="A" * 500)
    _write_skill(tmp_path, "beta", body="B" * 500)
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.invoked_skills.update({"alpha", "beta"})
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    await Skill(restore_after_compact=True).post_compact_restore(
        history, state, budget_tokens=150, estimate_tokens=_estimate
    )
    entry = history[0]
    assert isinstance(entry, UserMessage)
    # The overrun allowance covers the wrapper prose, which is outside the
    # per-skill accounting the budget governs.
    injected = _estimate(entry.text)
    assert injected <= 150 + 100, (
        f"hook injected {injected} tokens against a 150-token budget"
    )
    assert "A" * 500 in entry.text or "B" * 500 in entry.text
    assert "Not restored" in entry.text, (
        "skills were dropped for budget with no mention of the omission"
    )
    assert "beta" in entry.text, "the omitted skill was not named"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
