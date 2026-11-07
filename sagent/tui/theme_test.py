"""Terminal defaults, validated overrides, and renderer parity for themes."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import patch

from prompt_toolkit.styles import (
    Style as PromptStyle,
    merge_styles,
)
from prompt_toolkit.styles.defaults import default_ui_style
from pygments import format as format_tokens
from pygments.formatters import Terminal256Formatter
from pygments.style import Style as PygmentsStyle
from pygments.token import Token
from rich.color import ColorType
from rich.console import Console
from rich.default_styles import DEFAULT_STYLES
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

import pytest

from sagent.lib.files.owner_only import write_private
from sagent.lib.userdirs import config_dir, state_dir
from sagent.tui import theme as theme_module
from sagent.tui.theme import Theme, ThemeState, load_theme, theme_choices


if TYPE_CHECKING:
    from collections.abc import Mapping

    from pygments.token import _TokenType


def _custom(text: str, *, name: str = "custom") -> Path:
    directory = config_dir() / "rekursiv-ai" / "sagent" / "themes"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(text)
    return path


def test_default_restores_the_previous_message_and_diff_styles() -> None:
    theme = load_theme()
    assert theme.name == "default"
    assert load_theme("default") is theme
    assert ThemeState().current is theme
    for role, expected in {
        "text": "",
        "muted": "dim",
        "thinking": "dim italic",
        "status": "bright_black not reverse",
        "user": "white on rgb(55,55,55)",
        "to_agent": "white on rgb(25,45,70)",
        "to_user": "black on rgb(150,115,30)",
        "failure": "dim red",
        "added": "on rgb(34,92,43)",
        "removed": "on rgb(122,41,54)",
        "added_word": "on rgb(56,166,96)",
        "removed_word": "on rgb(179,89,107)",
        "gutter": "rgb(160,160,160)",
    }.items():
        assert theme.style(role) == Style.parse(expected), role


def test_terminal_preserves_terminal_palette_and_is_cached() -> None:
    theme = load_theme("terminal")
    assert theme.name == "terminal"
    assert load_theme("terminal") is theme
    for role in (
        "text",
        "muted",
        "status",
        "thinking",
        "accent",
        "success",
        "warning",
        "error",
        "heading",
        "section",
        "user",
        "to_agent",
        "to_user",
        "selected",
        "code",
        "comment",
        "keyword",
        "string",
        "number",
        "function",
        "added",
        "removed",
        "added_word",
        "removed_word",
    ):
        style = theme.style(role)
        assert style.bgcolor is None
        assert style.color is None or style.color.type in (
            ColorType.DEFAULT,
            ColorType.STANDARD,
        )
    assert theme.style("muted").dim is True
    assert theme.style("selected").reverse is True
    assert theme.style("code") == theme.style("text")
    assert theme.prompt_style() is theme.prompt_style()
    with pytest.raises(KeyError):
        theme.style("typo")


def test_terminal_status_keeps_its_previous_gray_without_reverse_or_dimming() -> None:
    style = merge_styles([default_ui_style(), load_theme("terminal").prompt_style()])
    status = style.get_attrs_for_style_str("class:bottom-toolbar")
    assert status.color == "ansibrightblack"
    assert status.bgcolor == "default"
    assert status.reverse is False
    assert status.dim is False
    for name in ("queued_input_pane", "tree.help", "auto-suggestion"):
        assert style.get_attrs_for_style_str(f"class:{name}").color == "ansibrightblack"
    prompt = style.get_attrs_for_style_str("class:input_pane")
    assert prompt.bold is True
    assert prompt.color == "default"


def test_default_prompt_matches_the_previous_merged_prompt_styles() -> None:
    previous = merge_styles(
        [
            default_ui_style(),
            PromptStyle.from_dict(
                {
                    "bottom-toolbar": "fg:ansibrightblack noreverse bg:default",
                    "queued_input_pane": "fg:ansibrightblack",
                    "input_pane": "bold",
                },
            ),
        ],
    )
    current = merge_styles([default_ui_style(), load_theme().prompt_style()])
    for name in (
        "bottom-toolbar",
        "queued_input_pane",
        "input_pane",
        "auto-suggestion",
        "selected",
    ):
        actual = current.get_attrs_for_style_str(f"class:{name}")
        expected = previous.get_attrs_for_style_str(f"class:{name}")
        assert actual._replace(
            color=actual.color or "default",
            bgcolor=actual.bgcolor or "default",
        ) == expected._replace(
            color=expected.color or "default",
            bgcolor=expected.bgcolor or "default",
        )


@pytest.mark.parametrize(
    ("token", "color"),
    [
        ((), "f8f8f2"),
        (("Keyword",), "f92672"),
        (("Keyword", "Type"), "66d9ef"),
        (("Name", "Builtin"), "a6e22e"),
        (("Name", "Class"), "a6e22e"),
        (("Name", "Function", "Magic"), "a6e22e"),
        (("Name", "Attribute"), "a6e22e"),
        (("Name", "Decorator"), "a6e22e"),
        (("Name", "Variable"), "ffffff"),
        (("Literal", "Number", "Integer"), "ae84ff"),
        (("Literal", "String"), "e6db74"),
        (("Literal", "String", "Escape"), "ae84ff"),
        (("Comment", "Single"), "75715e"),
        (("Operator",), "f92672"),
        (("Punctuation",), "f8f8f2"),
        (("Name", "Namespace"), "f92672"),
        (("Name", "Tag"), "f92672"),
    ],
)
def test_default_command_and_diff_tokens_match_the_previous_256_color_formatter(
    token: tuple[str, ...],
    color: str,
) -> None:
    class _Color(PygmentsStyle):
        styles: ClassVar[Mapping[_TokenType, str]] = {Token: f"#{color}"}

    ansi = format_tokens([(Token.Text, "x")], Terminal256Formatter[str](style=_Color))
    previous = Text.from_ansi(ansi).get_style_at_offset(Console(), 0)
    assert load_theme().source_style(token) == previous


def test_default_markdown_matches_the_previous_rich_styles() -> None:
    styles = load_theme().rich_theme().styles
    for name, expected in DEFAULT_STYLES.items():
        if name.startswith("markdown."):
            assert styles[name] == expected, name


@pytest.mark.parametrize(
    "token",
    [
        (),
        ("Keyword",),
        ("Keyword", "Type"),
        ("Punctuation",),
        ("Name", "Builtin"),
        ("Name", "Variable"),
        ("Comment", "Single"),
        ("Literal", "String", "Escape"),
        ("Generic", "Heading"),
        ("Error",),
    ],
)
def test_default_fenced_syntax_matches_the_previous_monokai(
    token: tuple[str, ...],
) -> None:
    previous = Syntax.get_theme("monokai")
    theme = load_theme()
    assert theme.get_style_for_token(token) == previous.get_style_for_token(token)
    assert theme.get_background_style() == previous.get_background_style()


def test_original_appearance_can_be_partially_overridden_from_yaml() -> None:
    _custom(
        "styles: {user: 'bold cyan on #123456', keyword: italic}\n"
        "source_tokens: {Keyword.Type: blue}\n",
    )
    theme = load_theme("custom")
    previous = load_theme()
    assert theme.syntax == "monokai"
    assert theme.style("user") == Style.parse("bold cyan on #123456")
    assert theme.style("to_agent") == previous.style("to_agent")
    assert theme.source_style(("Keyword", "Type")) == Style.parse("italic blue")
    assert theme.source_style(("Literal", "String")) == previous.source_style(
        ("Literal", "String"),
    )
    assert theme.get_style_for_token(("Keyword",)) == (
        Syntax.get_theme("monokai").get_style_for_token(("Keyword",))
        + Style(italic=True)
    )
    path = _custom(
        "syntax: ''\nsource_tokens: {Token: '', Name: ''}\nstyles: {code: cyan}",
    )
    theme = load_theme("custom")
    assert theme.get_background_style() == Style()
    assert theme.get_style_for_token(("Name",)) == Style.parse("cyan")
    assert theme.source_style(("Name", "Variable")) == Style.parse("cyan")
    selected = ThemeState(current=theme)
    path.write_text("syntax: no-such-pygments-theme")
    with pytest.raises(
        ValueError,
        match="unknown syntax theme: 'no-such-pygments-theme'",
    ):
        selected.select("custom")
    assert selected.current is theme


def test_custom_status_style_is_independent_of_muted_scrollback() -> None:
    _custom("styles: {status: 'bold cyan not reverse', muted: italic}")
    theme = load_theme("custom")
    style = merge_styles([default_ui_style(), theme.prompt_style()])
    status = style.get_attrs_for_style_str("class:bottom-toolbar")
    assert status.color == "ansicyan"
    assert status.bold is True
    assert status.reverse is False
    assert status.italic is False
    assert theme.style("muted").italic is True


def test_markdown_maps_every_role_without_losing_the_base_or_emphasis() -> None:
    roles = {
        role: Style(color=f"color({index})")
        for index, role in enumerate(load_theme().styles)
    }
    roles["text"] += Style(bgcolor="#abcdef")
    theme = Theme("distinct", roles)
    text = roles["text"]
    expected = {
        "paragraph": text,
        "text": text,
        "em": text + Style(italic=True),
        "emph": text + Style(italic=True),
        "strong": text + Style(bold=True),
        "s": text + Style(strike=True),
        "h1": text + roles["heading"] + Style(underline=True),
        "h2": text + roles["section"] + Style(underline=True),
        "h3": text + roles["section"] + Style(bold=True),
        "h4": text + roles["section"] + Style(italic=True),
        "h5": text + Style(italic=True),
        "h6": text + roles["muted"],
        "h7": text + roles["muted"] + Style(italic=True),
        "code": text + roles["code"] + roles["inline_code"],
        "code_block": text + roles["code"] + roles["block_code"],
        "block_quote": text + roles["quote"],
        "list": text + roles["list"],
        "item": text,
        "item.bullet": text + roles["bullet"],
        "item.number": text + roles["list_number"],
        "hr": text + roles["muted"],
        "h1.border": text,
        "link": text + roles["link"],
        "link_url": text + roles["link_url"] + Style(underline=True),
        "table.border": text + roles["table_border"],
        "table.header": text + roles["table_header"],
        "kbd": text + roles["kbd"] + Style(bold=True),
    }
    assert {
        name: style
        for name, style in theme.rich_theme().styles.items()
        if name.startswith("markdown.")
    } == {f"markdown.{name}": style for name, style in expected.items()}
    prompt = theme.prompt_style().get_attrs_for_style_str("class:input_pane")
    assert prompt.bold is True
    assert prompt.bgcolor == "abcdef"


def test_markdown_keeps_heading_levels_distinct_instead_of_bolding_all() -> None:
    styles = load_theme().rich_theme().styles
    assert styles["markdown.h1"].bold is True
    assert styles["markdown.h1"].underline is True
    assert styles["markdown.h2"].bold is not True
    assert styles["markdown.h2"].underline is True
    assert styles["markdown.h3"].bold is True
    assert styles["markdown.h4"].bold is not True
    assert styles["markdown.h4"].italic is True
    assert styles["markdown.h5"].bold is not True
    assert styles["markdown.h5"].italic is True
    assert styles["markdown.h6"].bold is not True
    assert styles["markdown.h6"].dim is True


def test_terminal_preserves_message_emphasis_and_syntax_color_families() -> None:
    theme = load_theme("terminal")
    assert theme.style("text").bold is not True
    assert theme.style("user").bold is not True
    assert theme.style("thinking") == Style.parse("dim italic")
    for role, color in (
        ("accent", "blue"),
        ("to_agent", "blue"),
        ("to_user", "yellow"),
        ("keyword", "red"),
        ("function", "green"),
        ("string", "yellow"),
        ("number", "magenta"),
    ):
        assert theme.style(role).color == Style.parse(color).color


@pytest.mark.parametrize("name", ["light", "dark"])
def test_explicit_presets_have_text_foreground_and_background(name: str) -> None:
    theme = load_theme(name)
    assert theme.name == name
    assert theme.style("text").color is not None
    assert theme.style("text").bgcolor is not None
    assert theme.style("code") == theme.style("text")
    assert theme.get_background_style() == theme.style("code").background_style


def test_custom_partial_roles_inherit_text_and_reload_atomically() -> None:
    path = _custom(
        "preset: light\nstyles:\n  text: '#102030 on #f0e0d0'\n  muted: dim\n  accent: bold magenta\n",
    )
    state = ThemeState()
    state.select("custom")
    selected = state.current
    assert selected.name == "custom"
    assert selected.style("muted") == Style.parse("dim #102030 on #f0e0d0")
    assert selected.style("accent") == Style.parse("bold magenta on #f0e0d0")
    assert selected.style("code") == selected.style("text")
    path.write_text("styles: {accent: bad-color}")
    with pytest.raises(ValueError, match="custom"):
        state.select("custom")
    assert state.current is selected
    path.write_text("styles: {accent: green}")
    state.select("custom")
    assert state.current.style("accent") == Style.parse("green")
    state.select("default")
    assert state.current.name == "default"
    assert state.prompt_style() is state.current.prompt_style()


def test_bundled_theme_requires_all_semantic_roles() -> None:
    with (
        patch.object(Path, "read_text", return_value="styles: {accent: cyan}"),
        pytest.raises(ValueError, match="missing semantic roles"),
    ):
        theme_module._bundled.__wrapped__("terminal")


def test_a_bundled_preset_is_read_once_and_takes_its_missing_roles_from_terminal() -> (
    None
):
    # Cached per process, so another file may have read every preset first.
    theme_module._bundled.cache_clear()

    dark = load_theme("dark")
    terminal = load_theme("terminal")

    assert set(dark.styles) == set(terminal.styles)
    assert dark.styles["muted"] == terminal.styles["muted"]
    assert dark.styles["accent"] != terminal.styles["accent"]
    assert load_theme("dark") is dark


def test_unknown_preset_reports_the_requested_base() -> None:
    path = _custom("preset: missing\nstyles: {}")
    with pytest.raises(ValueError, match="custom") as error:
        load_theme("custom")
    assert str(error.value) == (
        f"theme 'custom' ({path}): unknown theme preset: 'missing'"
    )


def test_permission_failure_preserves_current_theme() -> None:
    path = _custom("styles: {accent: red}")
    state = ThemeState()
    selected = state.current
    with (
        patch.object(Path, "read_text", side_effect=PermissionError("access denied")),
        pytest.raises(ValueError, match="custom") as error,
    ):
        state.select("custom")
    assert str(path) in str(error.value)
    assert isinstance(error.value.__cause__, PermissionError)
    assert state.current is selected


@pytest.mark.parametrize(
    "document",
    [
        "",
        "null",
        "[]",
        "{}",
        "styles: []",
        "styles: {unknown: red}",
        "styles: {accent: 12}",
        "styles: {accent: bad-color}",
        "styles: {accent: 'rgb(300,0,0)'}",
        "styles: {accent: '#123456junk'}",
        "styles: {accent: red}\nextra: true",
        "preset: custom\nstyles: {}",
        "styles: {accent: 'link https://example.org'}",
        'styles: {accent: "red\\u001b[0m"}',
        "styles: {accent: red, accent: blue}",
        "syntax: no-such-pygments-theme",
        "syntax: 12",
        "source_tokens: {Keyword.Type: bad-color}",
        "source_tokens: {keyword: red}",
        "source_tokens: {Name: 'link https://example.org'}",
        "source_tokens: {Name: 12}",
    ],
)
def test_invalid_custom_documents_are_rejected(document: str) -> None:
    _custom(document)
    with pytest.raises(ValueError, match="custom"):
        load_theme("custom")


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../dark", "/outside/theme", "a/b", "a\\b", "a\n", "x.yaml"],
)
def test_theme_names_cannot_escape_the_theme_directory(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        load_theme(name)


def test_choices_reserve_builtins_and_filter_unsafe_names() -> None:
    _custom("styles: {accent: red}", name="zebra")
    _custom("styles: {accent: blue}", name="alpha")
    _custom("invalid", name="terminal")
    _custom("invalid", name="default")
    _custom("invalid", name="bad name")
    assert theme_choices() == ("default", "terminal", "light", "dark", "alpha", "zebra")
    assert load_theme("terminal").name == "terminal"
    with pytest.raises(ValueError, match="missing"):
        load_theme("missing")


def test_custom_theme_cannot_follow_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "outside.yaml"
    target.write_text("styles: {accent: red}")
    path = _custom("styles: {}")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="custom") as error:
        load_theme("custom")
    assert str(error.value) == (
        f"theme 'custom' ({path}): theme files must not be symbolic links"
    )
    assert "custom" not in theme_choices()


@pytest.mark.parametrize(
    ("token", "role"),
    [
        (("Comment", "Single"), "comment"),
        (("Keyword", "Namespace"), "keyword"),
        (("Operator",), "keyword"),
        (("Punctuation",), "keyword"),
        (("Name", "Namespace"), "keyword"),
        (("Name", "Tag"), "keyword"),
        (("Literal", "String", "Double"), "string"),
        (("Literal", "Number", "Integer"), "number"),
        (("Name", "Function"), "function"),
        (("Name", "Class"), "function"),
        (("Name", "Builtin"), "function"),
        (("Name", "Decorator"), "function"),
        (("Name", "Attribute"), "function"),
        (("Generic", "Inserted"), "syntax_added"),
        (("Generic", "Deleted"), "syntax_removed"),
        (("Error",), "syntax_error"),
        (("Name", "Variable"), "code"),
        (("Name", "Function", "Magic"), "function"),
        (("Name", "Tag", "Custom"), "keyword"),
        (("Generic", "Inserted", "Custom"), "syntax_added"),
        (("Generic", "Deleted", "Custom"), "syntax_removed"),
        (("Error", "Custom"), "syntax_error"),
        ((), "code"),
    ],
)
def test_syntax_tokens_follow_semantic_roles(token: tuple[str, ...], role: str) -> None:
    theme = load_theme("terminal")
    assert theme.get_style_for_token(token) == theme.style("code") + theme.style(role)
    assert theme.source_style(token) == theme.get_style_for_token(token)


def test_rich_markdown_and_prompt_styles_follow_custom_palette() -> None:
    _custom(
        "preset: terminal\nstyles:\n  text: '#123456 on #abcdef'\n  accent: cyan\n  selected: bold reverse\n  code: italic\n",
    )
    theme = load_theme("custom")
    rich = theme.rich_theme()
    assert rich.styles["markdown.code"] == theme.style("code")
    assert rich.styles["markdown.h1"] == theme.style("heading") + Style(underline=True)
    assert rich.styles["markdown.h2"] == theme.style("section") + Style(underline=True)
    prompt = theme.prompt_style()
    assert prompt.get_attrs_for_style_str("").color == "123456"
    for name in (
        "",
        "text",
        "background",
        "input_pane",
        "queued_input_pane",
        "bottom-toolbar",
        "auto-suggestion",
        "tree.help",
    ):
        assert (
            prompt.get_attrs_for_style_str(f"class:{name}" if name else "").bgcolor
            == "abcdef"
        )
    for name in ("selected-text", "tree.selected"):
        assert prompt.get_attrs_for_style_str(f"class:{name}").reverse is True
    output = StringIO()
    console = Console(file=output, force_terminal=True, color_system="truecolor")
    console.print(Syntax("def f():\n    return 42", "python", theme=theme))
    assert "38;2;" in output.getvalue()


@pytest.mark.parametrize("base", ["default", "terminal", "light", "dark"])
def test_select_creates_an_exact_copy_and_remembers_it(base: str) -> None:
    state = ThemeState(current=load_theme(base))
    before = state.current
    state.select("paper")
    copied = load_theme("paper")
    assert copied.name == "paper"
    assert copied.styles == before.styles
    assert copied.syntax == before.syntax
    assert copied.source_tokens == before.source_tokens
    assert ThemeState().current.name == "paper"
    path = config_dir() / "rekursiv-ai" / "sagent" / "themes" / "paper.yaml"
    assert path.stat().st_mode & 0o777 == 0o600
    state.select("dark")
    assert ThemeState().current.name == "dark"
    state.select("paper")
    assert state.current.styles == before.styles


def test_edits_replace_one_raw_role_and_reload_other_sessions_changes() -> None:
    _custom("preset: light\nstyles: {thinking: bold}\n", name="paper")
    state = ThemeState()
    state.select("paper")
    other = ThemeState()
    other.edit("paper", "accent", "italic magenta")
    state.edit("paper", "thinking", "dim italic")
    assert state.current.name == "paper"
    assert state.current.styles["thinking"] == Style.parse("dim italic")
    assert state.current.styles["accent"] == Style.parse("italic magenta")
    assert ThemeState().current.styles == state.current.styles
    state.edit("paper", "thinking", "")
    assert load_theme("paper").styles["thinking"] == Style()


@pytest.mark.parametrize("base", ["default", "terminal", "light", "dark"])
def test_builtin_edits_fork_without_mutating_presets(base: str) -> None:
    _custom("styles: {thinking: bold}", name="user1")
    before = load_theme(base)
    state = ThemeState()
    state.edit(base, "thinking", "italic cyan")
    assert state.current.name == "user2"
    assert state.current.styles["thinking"] == Style.parse("italic cyan")
    assert state.current.syntax == before.syntax
    assert state.current.source_tokens == before.source_tokens
    assert load_theme(base) is before
    assert load_theme(base).styles["thinking"] != Style.parse("italic cyan")
    assert load_theme("user1").styles["thinking"] == Style.parse("bold")
    assert ThemeState().current.name == "user2"
    state.edit("user2", "thinking", "dim")
    assert state.current.name == "user2"
    assert "user3" not in theme_choices()


@pytest.mark.parametrize(
    ("name", "role", "value"),
    [
        ("default", "thinking", "bad-color"),
        ("../bad", "thinking", "red"),
        ("default", "typo", "red"),
        ("missing", "thinking", "red"),
    ],
)
def test_invalid_edits_leave_current_files_and_selection_unchanged(
    name: str,
    role: str,
    value: str,
) -> None:
    state = ThemeState()
    state.select("light")
    before = state.current
    with pytest.raises(ValueError, match="theme"):
        state.edit(name, role, value)
    assert state.current is before
    assert ThemeState().current.name == "light"
    assert theme_choices() == ("default", "terminal", "light", "dark")


def test_unusable_remembered_theme_falls_back_without_creating_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = ThemeState()
    state.select("paper")
    path = config_dir() / "rekursiv-ai" / "sagent" / "themes" / "paper.yaml"
    path.write_text("invalid")
    assert ThemeState().current.name == "default"
    assert "paper" in caplog.text
    assert path.read_text() == "invalid"
    path.unlink()
    assert ThemeState().current.name == "default"
    assert not path.exists()


def test_write_failure_keeps_current_and_remembered_selection() -> None:
    state = ThemeState()
    state.select("light")
    before = state.current
    with (
        patch.object(
            theme_module,
            "write_private",
            side_effect=PermissionError("denied"),
        ),
        pytest.raises(ValueError, match="denied"),
    ):
        state.select("dark")
    assert state.current is before
    assert ThemeState().current.name == "light"


@pytest.mark.parametrize("name", ["default", "paper"])
def test_edit_save_failures_leave_the_old_theme_and_selection(name: str) -> None:
    state = ThemeState()
    state.select("paper")
    before = state.current
    path = config_dir() / "rekursiv-ai" / "sagent" / "themes" / "paper.yaml"
    original = path.read_bytes()
    with (
        patch.object(
            theme_module,
            "write_private",
            side_effect=PermissionError("denied"),
        ),
        pytest.raises(ValueError, match="cannot save theme: denied"),
    ):
        state.edit(name, "thinking", "red")
    assert state.current is before
    assert path.read_bytes() == original
    assert "user1" not in theme_choices()
    assert ThemeState().current.name == "paper"


@pytest.mark.parametrize("name", ["default", "paper"])
def test_selection_write_failure_rolls_back_the_yaml(name: str) -> None:
    state = ThemeState()
    state.select("paper")
    before = state.current
    path = config_dir() / "rekursiv-ai" / "sagent" / "themes" / "paper.yaml"
    original = path.read_bytes()
    with (
        patch.object(theme_module, "write_private", side_effect=_PreferenceFailure()),
        pytest.raises(ValueError, match="denied"),
    ):
        state.edit(name, "thinking", "red")
    assert state.current is before
    assert path.read_bytes() == original
    assert "user1" not in theme_choices()
    assert ThemeState().current.name == "paper"


class _PreferenceFailure:
    def __call__(self, path: Path, data: str | bytes) -> None:
        if path == state_dir() / "rekursiv-ai" / "sagent" / "theme":
            raise PermissionError("denied")
        write_private(path, data)


def test_first_preset_edit_forks_as_user1_and_reserves_dangling_links(
    tmp_path: Path,
) -> None:
    state = ThemeState()
    state.edit("default", "thinking", "bold")
    assert state.current.name == "user1"
    assert ThemeState().current.name == "user1"
    directory = config_dir() / "rekursiv-ai" / "sagent" / "themes"
    (directory / "user2.yaml").symlink_to(tmp_path / "missing")
    state.edit("default", "thinking", "italic")
    assert state.current.name == "user3"
    assert (directory / "user2.yaml").is_symlink()
    with pytest.raises(ValueError, match="symbolic links"):
        state.select("user2")
    with pytest.raises(ValueError, match="symbolic links"):
        state.edit("user2", "thinking", "red")


@pytest.mark.parametrize("data", [b"../invalid", b"\xff"])
def test_invalid_remembered_selection_falls_back_without_writing(data: bytes) -> None:
    path = state_dir() / "rekursiv-ai" / "sagent" / "theme"
    write_private(path, data)
    assert ThemeState().current.name == "default"
    assert path.read_bytes() == data


def test_unreadable_preference_does_not_prevent_startup() -> None:
    expected = load_theme()
    with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
        assert ThemeState().current is expected


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
