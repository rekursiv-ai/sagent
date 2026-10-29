"""Tests for the Skill tool: finding skills, listing them, and loading one."""

from __future__ import annotations

from importlib import util
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import logging
import os
import re
import sys

import pytest

from sagent.lib.userdirs import data_dir
from sagent.session.types import UncategorizedToolResult
from sagent.tools import skill
from sagent.tools.skill import Skill, SkillInfo
from sagent.tools.types import ToolContext, schema_of


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from treekle import PlainTree

    from sagent.tools.skill import Source


def _write(
    root: Path,
    rel: str,
    *,
    name: str | None = None,
    description: str | None = "Use when foo.",
    body: str = "Do the foo.\n",
) -> Path:
    """Write ``root/.sagent/skills/<rel>/SKILL.md`` with the frontmatter given."""
    directory = root / ".sagent" / "skills" / rel
    directory.mkdir(parents=True, exist_ok=True)
    meta = "".join(
        f"{key}: {value}\n"
        for key, value in (("name", name), ("description", description))
        if value is not None
    )
    path = directory / "SKILL.md"
    path.write_text(f"---\n{meta}---\n\n{body}" if meta else body, encoding="utf-8")
    return path


def _backdate(path: Path, *, sec: int) -> None:
    """Set ``path``'s times ``sec`` past the epoch, so a change cannot share a tick."""
    os.utime(path, ns=(sec * 1_000_000_000, sec * 1_000_000_000))


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Return the directories discovery reads a ``SKILL.md`` from, as it reads them."""
    seen: list[Path] = []
    real = skill._load

    def counting(directory: Path, source: Source) -> SkillInfo | None:
        seen.append(directory)
        return real(directory, source)

    monkeypatch.setattr(skill, "_load", counting)
    return seen


# Executed under its own name, so ``sagent.tools.skill`` and the classes the
# other tests imported from it stay the ones they compare against.
def _fresh(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> Mapping[str, object]:
    """Return ``module``'s namespace, run again from its file as a separate module."""
    name = f"{module.__name__}_fresh"
    spec = util.spec_from_file_location(name, module.__file__ or "")
    assert spec is not None
    assert spec.loader is not None
    fresh = util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, fresh)
    spec.loader.exec_module(fresh)
    return vars(fresh)


def _names(skills: list[SkillInfo]) -> list[str]:
    return [s.name for s in skills]


@pytest.fixture(autouse=True)
def no_builtins(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Find no skills sagent ships, unless a test asks for them."""
    empty = tmp_path_factory.mktemp("package")

    def nowhere(package: str) -> Path:
        del package
        return empty

    monkeypatch.setattr(skill, "files", nowhere)


def test_the_skills_sagent_ships_are_found_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    shipped = Skill(user_dir=tmp_path / "u")
    assert shipped.builtin_dir == _CWD.parent / "skills"
    (watch,) = [s for s in shipped.discover(tmp_path) if s.name == "watch"]
    assert watch.source == "builtin"
    assert "Judge by evidence" in watch.body

    _write(tmp_path, "watch", name="watch", description="Mine.")
    (mine,) = [s for s in shipped.discover(tmp_path) if s.name == "watch"]
    assert (mine.source, mine.description) == ("project", "Mine.")


def test_the_watch_skill_sends_findings_to_the_agent_not_the_user() -> None:
    body = (_CWD.parent / "skills" / "watch" / "SKILL.md").read_text(encoding="utf-8")
    act = " ".join(body.split("## Act", 1)[1].split())
    assert 'Every finding goes to the agent with `AgentSend(to="<label>", ...)`' in act
    # The user is messaged from one rule alone, never as where findings go.
    assert act.count('to="user"') == 1
    assert 'Message the user, with `AgentSend(to="user", ...)`, only to reply' in act


def test_discover_reads_a_project_skill(tmp_path: Path) -> None:
    path = _write(tmp_path, "alpha", name="alpha", description="Trigger A")
    assert Skill(user_dir=tmp_path / "user").discover(tmp_path) == [
        SkillInfo(
            name="alpha",
            description="Trigger A",
            body="Do the foo.",
            source="project",
            path=path.resolve(),
        ),
    ]


def test_default_user_dir_is_sagent_skills_under_data_dir() -> None:
    assert Skill().user_dir == data_dir() / "rekursiv-ai" / "sagent" / "skills"


def test_the_nearest_skill_of_a_name_wins(tmp_path: Path) -> None:
    child = tmp_path / "child"
    user = tmp_path / "user"
    _write(tmp_path, "shared", description="root")
    _write(tmp_path, "rootonly", description="root")
    _write(child, "shared", description="child")
    _write(user, "shared", description="user")
    _write(user, "mine", description="user")
    skills = Skill(user_dir=user / ".sagent" / "skills").discover(child)
    assert [(s.name, s.description, s.source) for s in skills] == [
        ("shared", "child", "project"),
        ("rootonly", "root", "project"),
        ("mine", "user", "user"),
    ]


def test_a_skill_named_by_its_directory_when_frontmatter_omits_it(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "from-dir", name=None)
    assert _names(Skill(user_dir=tmp_path / "u").discover(tmp_path)) == ["from-dir"]


@pytest.mark.parametrize(
    "name",
    ["Bad!Name", "-leading-dash", "UPPER", "a" * 65],
)
def test_an_invalid_name_is_skipped_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    name: str,
) -> None:
    _write(tmp_path, "ok", name=f"'{name}'")
    _write(tmp_path, "good", name="good")
    with caplog.at_level(logging.WARNING):
        skills = Skill(user_dir=tmp_path / "u").discover(tmp_path)
    assert _names(skills) == ["good"]
    assert f"invalid name {name!r}" in caplog.text


def test_a_sixty_four_character_name_is_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "long", name="a" * 64)
    assert _names(Skill(user_dir=tmp_path / "u").discover(tmp_path)) == ["a" * 64]


@pytest.mark.parametrize(
    "meta",
    ["name: 123\n", "name: [a, b]\n", "description: {a: 1}\n"],
)
def test_a_non_text_name_or_description_is_skipped_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    meta: str,
) -> None:
    directory = tmp_path / ".sagent" / "skills" / "odd"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\n{meta}---\nbody\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert Skill(user_dir=tmp_path / "u").discover(tmp_path) == []
    assert "non-text name or description" in caplog.text


@pytest.mark.parametrize(
    "frontmatter",
    ["---\nname: [unclosed\n---\n", "---\n- a list\n---\n", "\ufeff---\nx: 1\n---\n"],
)
def test_unusable_frontmatter_is_dropped_and_the_body_kept(
    tmp_path: Path,
    frontmatter: str,
) -> None:
    directory = tmp_path / ".sagent" / "skills" / "plain"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"{frontmatter}The body.\n", encoding="utf-8")
    (info,) = Skill(user_dir=tmp_path / "u").discover(tmp_path)
    assert (info.name, info.description, info.body) == (
        "plain",
        "The body.",
        "The body.",
    )


def test_a_file_without_frontmatter_is_all_body(tmp_path: Path) -> None:
    _write(tmp_path, "bare", name=None, description=None, body="# Title\n\nFirst.\n")
    (info,) = Skill(user_dir=tmp_path / "u").discover(tmp_path)
    assert (info.name, info.description, info.body) == (
        "bare",
        "First.",
        "# Title\n\nFirst.",
    )


def test_the_description_falls_back_to_the_first_line_past_headings(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "a",
        description=None,
        body="\n# heading\n\n  Real line.  \nNext.\n",
    )
    (info,) = Skill(user_dir=tmp_path / "u").discover(tmp_path)
    assert info.description == "Real line."


def test_a_skill_of_only_headings_has_no_description(tmp_path: Path) -> None:
    _write(tmp_path, "a", description=None, body="# one\n## two\n")
    (info,) = Skill(user_dir=tmp_path / "u").discover(tmp_path)
    assert info.description == ""


def test_a_frontmatter_only_skill_keeps_the_raw_file_as_its_body(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "empty", name="empty", body="  \n")
    (info,) = Skill(user_dir=tmp_path / "u").discover(tmp_path)
    assert info.body == path.read_text(encoding="utf-8").strip()


def test_an_unreadable_skill_is_skipped_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    directory = tmp_path / ".sagent" / "skills" / "binary"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_bytes(b"\xff\xfe\x00bad")
    with caplog.at_level(logging.WARNING):
        assert Skill(user_dir=tmp_path / "u").discover(tmp_path) == []
    assert "unreadable skill" in caplog.text


def test_directories_without_skill_md_and_stray_files_are_ignored(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / ".sagent" / "skills"
    (root / "empty").mkdir(parents=True)
    (root / "stray.md").write_text("ignored", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert Skill(user_dir=tmp_path / "u").discover(tmp_path) == []
    assert caplog.text == ""


def test_skills_nest_and_list_depth_first_in_name_order(tmp_path: Path) -> None:
    _write(tmp_path, "trax", name="trax")
    _write(tmp_path, "trax/paper", name="trax-paper")
    _write(tmp_path, "group/inner", name="inner")
    _write(tmp_path, "alpha", name="alpha")
    assert _names(Skill(user_dir=tmp_path / "u").discover(tmp_path)) == [
        "alpha",
        "inner",
        "trax",
        "trax-paper",
    ]


def test_a_link_back_up_the_tree_is_entered_once(tmp_path: Path) -> None:
    _write(tmp_path, "trax", name="trax")
    root = tmp_path / ".sagent" / "skills"
    (root / "trax" / "loop").symlink_to(root)
    (root / "again").symlink_to(root / "trax")
    assert _names(Skill(user_dir=tmp_path / "u").discover(tmp_path)) == ["trax"]


def test_a_user_dir_linked_to_a_project_dir_lists_its_skills_once(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "alpha", name="alpha")
    link = tmp_path / "user-skills"
    link.symlink_to(tmp_path / ".sagent" / "skills")
    skills = Skill(user_dir=link).discover(tmp_path)
    assert [(s.name, s.source) for s in skills] == [("alpha", "project")]


def test_reminder_lists_each_skill_with_its_trigger(tmp_path: Path) -> None:
    _write(tmp_path, "alpha", name="alpha", description="|\n  Use when\n  alpha.")
    _write(tmp_path, "beta", name="beta", description=None, body="# only\n")
    user = tmp_path / "user"
    _write(user, "gamma", name="gamma", description="Use when gamma.")
    context = ToolContext(cwd=tmp_path)
    text = Skill(user_dir=user / ".sagent" / "skills").reminder(context)
    assert text == (
        "Skills you can load with the Skill tool:\n"
        "- alpha (project): Use when alpha.\n"
        "- beta (project): (no description)\n"
        "- gamma (user): Use when gamma."
    )


def test_the_skills_are_where_the_agent_started_wherever_it_went(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    away = tmp_path / "away"
    _write(root, "alpha", name="alpha")
    _write(away, "beta", name="beta")
    context = ToolContext(cwd=away, root=root)

    text = Skill(user_dir=tmp_path / "u").reminder(context)

    assert "- alpha (project)" in text
    assert "beta" not in text


def test_reminder_is_empty_without_skills(tmp_path: Path) -> None:
    assert Skill(user_dir=tmp_path / "u").reminder(ToolContext(cwd=tmp_path)) == ""


def test_a_second_reminder_without_changes_reads_no_skill_file(
    tmp_path: Path,
    loads: list[Path],
) -> None:
    _write(tmp_path, "alpha", name="alpha")
    _write(tmp_path, "alpha/inner", name="inner")
    _write(tmp_path / "user", "mine", name="mine")
    tool = Skill(user_dir=tmp_path / "user" / ".sagent" / "skills")
    context = ToolContext(cwd=tmp_path)
    first = tool.reminder(context)
    read = len(loads)
    assert tool.reminder(context) == first
    assert (read, len(loads)) == (3, 3)


def test_each_working_directory_keeps_its_own_scan(
    tmp_path: Path,
    loads: list[Path],
) -> None:
    _write(tmp_path / "a", "alpha", name="alpha")
    _write(tmp_path / "b", "beta", name="beta")
    tool = Skill(user_dir=tmp_path / "u")
    for cwd in ("a", "b", "a", "b"):
        tool.discover(tmp_path / cwd)
    assert len(loads) == 2


def test_an_edited_skill_is_read_again(tmp_path: Path) -> None:
    path = _write(tmp_path, "alpha", name="alpha", description="old")
    tool = Skill(user_dir=tmp_path / "u")
    assert [s.description for s in tool.discover(tmp_path)] == ["old"]
    _write(tmp_path, "alpha", name="alpha", description="new")
    _backdate(path, sec=1)
    assert [s.description for s in tool.discover(tmp_path)] == ["new"]


@pytest.mark.parametrize("change", ["size", "inode", "links"])
def test_a_change_within_one_clock_tick_is_still_seen(
    tmp_path: Path,
    change: str,
) -> None:
    path = _write(tmp_path, "alpha", name="alpha", description="old")
    root = path.parent.parent
    tool = Skill(user_dir=tmp_path / "u")
    tool.discover(tmp_path)
    # The rename also touches the directory, which then must look untouched too.
    changed = [root] if change == "links" else [path, path.parent]
    mtimes = [p.stat().st_mtime_ns for p in changed]
    if change == "size":
        _write(tmp_path, "alpha", name="alpha", description="newer")
    elif change == "inode":
        fresh = tmp_path / "fresh.md"
        fresh.write_text(path.read_text().replace("old", "new"), encoding="utf-8")
        fresh.replace(path)
    else:
        _write(tmp_path, "beta", name="beta")
    for p, mtime in zip(changed, mtimes, strict=True):
        os.utime(p, ns=(mtime, mtime))
    assert [(s.name, s.description) for s in tool.discover(tmp_path)] == {
        "size": [("alpha", "newer")],
        "inode": [("alpha", "new")],
        "links": [("alpha", "old"), ("beta", "Use when foo.")],
    }[change]


def test_an_added_skill_appears_and_a_removed_one_goes(tmp_path: Path) -> None:
    _write(tmp_path, "alpha", name="alpha")
    root = tmp_path / ".sagent" / "skills"
    tool = Skill(user_dir=tmp_path / "u")
    assert _names(tool.discover(tmp_path)) == ["alpha"]
    _write(tmp_path, "beta", name="beta")
    _backdate(root, sec=1)
    assert _names(tool.discover(tmp_path)) == ["alpha", "beta"]
    (root / "alpha" / "SKILL.md").unlink()
    (root / "alpha").rmdir()
    _backdate(root, sec=2)
    assert _names(tool.discover(tmp_path)) == ["beta"]


def test_a_skill_added_below_another_appears(tmp_path: Path) -> None:
    _write(tmp_path, "trax", name="trax")
    tool = Skill(user_dir=tmp_path / "u")
    assert _names(tool.discover(tmp_path)) == ["trax"]
    _write(tmp_path, "trax/paper", name="trax-paper")
    _backdate(tmp_path / ".sagent" / "skills" / "trax", sec=1)
    assert _names(tool.discover(tmp_path)) == ["trax", "trax-paper"]


def test_a_skills_directory_made_later_is_found(tmp_path: Path) -> None:
    user = tmp_path / "user"
    tool = Skill(user_dir=user / ".sagent" / "skills")
    assert tool.discover(tmp_path) == []
    _write(tmp_path, "alpha", name="alpha")
    assert _names(tool.discover(tmp_path)) == ["alpha"]
    _write(user, "mine", name="mine")
    assert _names(tool.discover(tmp_path)) == ["alpha", "mine"]


@pytest.mark.asyncio
async def test_call_returns_the_escaped_body_in_a_skill_block(tmp_path: Path) -> None:
    _write(tmp_path, "alpha", name="alpha", body="hello <world> & </skill>\n")
    result = await Skill(user_dir=tmp_path / "u").call(
        ToolContext(cwd=tmp_path),
        {"skill": "alpha"},
    )
    assert result == UncategorizedToolResult(
        call_id="",
        content="<skill name='alpha' source='project'>\n"
        "hello &lt;world&gt; &amp; &lt;/skill&gt;\n</skill>",
    )


@pytest.mark.asyncio
async def test_call_appends_the_escaped_args(tmp_path: Path) -> None:
    _write(tmp_path, "alpha", name="alpha", body="body\n")
    result = await Skill(user_dir=tmp_path / "u").call(
        ToolContext(cwd=tmp_path),
        {"skill": "alpha", "args": "<go>"},
    )
    assert result.content == (
        "<skill name='alpha' source='project'>\nbody\n\nArguments: &lt;go&gt;\n</skill>"
    )


@pytest.mark.asyncio
async def test_call_finds_skills_from_the_callers_cwd(tmp_path: Path) -> None:
    user = tmp_path / "user"
    _write(user, "mine", name="mine", body="user body\n")
    result = await Skill(user_dir=user / ".sagent" / "skills").call(
        ToolContext(cwd=tmp_path / "elsewhere"),
        {"skill": "mine"},
    )
    assert result.content == "<skill name='mine' source='user'>\nuser body\n</skill>"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("present", "available"),
    [(True, "alpha, beta"), (False, "(none)")],
)
async def test_an_unknown_skill_raises_naming_those_available(
    tmp_path: Path,
    *,
    present: bool,
    available: str,
) -> None:
    if present:
        _write(tmp_path, "beta", name="beta")
        _write(tmp_path, "alpha", name="alpha")
    with pytest.raises(
        ValueError,
        match=rf"^Unknown skill 'bet'\. Available: {re.escape(available)}\.$",
    ):
        # A prefix of a real name, so only an exact match may load.
        await Skill(user_dir=tmp_path / "u").call(
            ToolContext(cwd=tmp_path),
            {"skill": "bet"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "Missing argument 'skill'"),
        ({"skill": 3}, "'skill' must be string"),
        ({"skill": "a", "extra": "x"}, "Unknown arguments: extra"),
    ],
)
async def test_bad_arguments_raise(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await Skill(user_dir=tmp_path / "u").call(ToolContext(cwd=tmp_path), arguments)


def test_the_description_is_read_as_utf8_whatever_the_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The asset holds an arrow, which a Latin-1 locale would read as three letters."""
    real = Path.read_text

    def latin1_default(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        return real(self, encoding=encoding or "latin-1", errors=errors)

    monkeypatch.setattr(Path, "read_text", latin1_default)
    tool = _fresh(monkeypatch, skill)["Skill"]
    assert isinstance(tool, _Described)
    assert "→" in tool.description
    assert "â" not in tool.description


def test_the_model_sees_the_name_schema_and_asset() -> None:
    tool = Skill()
    # The compactor elides results of a tool by this exact name.
    assert tool.name == "Skill"
    assert tool.schema == schema_of(Skill.Arguments)
    assert tool.schema["required"] == ["skill"]
    assert tool.description.startswith("Load a user-authored skill")


@pytest.mark.parametrize(
    ("arguments", "label"),
    [({"skill": "alpha"}, "Skill alpha"), ({}, "Skill")],
)
def test_summary_names_the_skill(arguments: dict[str, str], label: str) -> None:
    assert Skill().summary(arguments) == label


def test_calls_run_beside_anything(tmp_path: Path) -> None:
    assert Skill().serialize_key(ToolContext(cwd=tmp_path), {"skill": "a"}) is None


@runtime_checkable
class _Described(Protocol):
    """A tool class from a module run again under another name."""

    description: str


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
