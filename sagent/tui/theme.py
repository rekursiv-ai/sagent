"""Semantic YAML themes shared by Rich and prompt-toolkit renderers.

Custom themes live under the user's sagent configuration directory, in
``themes/NAME.yaml``. A ``preset`` selects default, terminal, light, or dark;
``styles`` partially overrides its semantic Rich styles. The default preserves
sagent's original appearance; terminal opts into the terminal's palette.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

import fcntl
import logging
import os
import re

from pygments.util import ClassNotFound
from treekle import from_plain
from yaml import YAMLError, safe_dump

from sagent.lib.files.owner_only import open_private, write_private
from sagent.lib.userdirs import config_dir, state_dir
from sagent.lib.yaml import parse


_CWD: Final[Path] = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from prompt_toolkit.styles import Style as PromptStyle
    from pygments.styles import get_style_by_name
    from rich import errors
    from rich.color import Color
    from rich.style import Style
    from rich.syntax import Syntax, SyntaxTheme
    from rich.theme import Theme as RichTheme
else:
    from wrapt import lazy_import

    PromptStyle = lazy_import("prompt_toolkit.styles", "Style")
    get_style_by_name = lazy_import("pygments.styles", "get_style_by_name")
    errors = lazy_import("rich.errors")
    Style = lazy_import("rich.style", "Style")
    Syntax = lazy_import("rich.syntax", "Syntax")
    SyntaxTheme = lazy_import("rich.syntax", "SyntaxTheme")
    RichTheme = lazy_import("rich.theme", "Theme")


_ROLES: Final = (
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
    "failure",
    "gutter",
    "inline_code",
    "block_code",
    "quote",
    "list",
    "bullet",
    "list_number",
    "link",
    "link_url",
    "table_border",
    "table_header",
    "kbd",
    "syntax_added",
    "syntax_removed",
    "syntax_error",
    "suggestion",
)
_PRESETS: Final = ("default", "terminal", "light", "dark")
_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_ANSI: Final = (
    "ansiblack",
    "ansired",
    "ansigreen",
    "ansiyellow",
    "ansiblue",
    "ansimagenta",
    "ansicyan",
    "ansigray",
    "ansibrightblack",
    "ansibrightred",
    "ansibrightgreen",
    "ansibrightyellow",
    "ansibrightblue",
    "ansibrightmagenta",
    "ansibrightcyan",
    "ansiwhite",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Document:
    preset: str = "default"
    styles: dict[str, str] = field(default_factory=dict[str, str])
    syntax: str | None = None
    source_tokens: dict[str, str] = field(default_factory=dict[str, str])


class Theme(SyntaxTheme):
    """A named palette with stable Rich and prompt-toolkit adapters."""

    def __init__(
        self,
        name: str,
        styles: Mapping[str, Style],
        *,
        syntax: str = "",
        source_tokens: Mapping[str, Style] | None = None,
    ) -> None:
        self.name = name
        if syntax:
            try:
                get_style_by_name(syntax)
            except ClassNotFound as error:
                raise ValueError(f"unknown syntax theme: {syntax!r}") from error
        self.syntax = syntax
        self.source_tokens: Mapping[str, Style] = MappingProxyType(
            dict(source_tokens or {}),
        )
        self._syntax = Syntax.get_theme(syntax) if syntax else None
        self._styles = MappingProxyType(dict(styles))
        self._rich = RichTheme(self._markdown_styles())
        self._prompt = PromptStyle.from_dict(
            {
                "": _prompt(self.style("text")),
                "text": _prompt(self.style("text")),
                "background": _prompt(self.style("text")),
                "bottom-toolbar": _prompt(self.style("status")),
                "queued_input_pane": _prompt(self.style("status")),
                "input_pane": _prompt(self.style("text") + Style(bold=True)),
                "selected-text": _prompt(self.style("selected")),
                "auto-suggestion": _prompt(self.style("suggestion")),
                "tree.selected": _prompt(self.style("selected")),
                "tree.help": _prompt(self.style("status")),
            },
        )

    @property
    def styles(self) -> Mapping[str, Style]:
        """The read-only role overrides, before the base text is composed."""
        return self._styles

    def style(self, role: str) -> Style:
        """Return the semantic style over the palette's base text."""
        return self._styles["text"] + self._styles[role]

    @override
    def get_style_for_token(self, token_type: tuple[str, ...]) -> Style:
        base = self._syntax.get_style_for_token(token_type) if self._syntax else Style()
        return base + self.style("code") + self._styles[self._token_role(token_type)]

    def source_style(self, token_type: tuple[str, ...]) -> Style:
        """Return command and diff syntax without the fenced-code background.

        Args:
          token_type: Hierarchical Pygments token name.

        Returns:
          style: Source-token style composed with semantic overrides.

        """
        token = token_type
        while token:
            style = self.source_tokens.get(".".join(token))
            if style is not None:
                break
            token = token[:-1]
        else:
            style = self.source_tokens.get("Token", Style())
        return style + self.style("code") + self._styles[self._token_role(token_type)]

    def _token_role(self, token_type: tuple[str, ...]) -> str:
        if token_type[:1] == ("Comment",):
            role = "comment"
        elif token_type[:1] in (("Keyword",), ("Operator",), ("Punctuation",)):
            role = "keyword"
        elif token_type[:2] == ("Literal", "String"):
            role = "string"
        elif token_type[:2] == ("Literal", "Number"):
            role = "number"
        elif token_type[:2] in (
            ("Name", "Function"),
            ("Name", "Class"),
            ("Name", "Builtin"),
            ("Name", "Decorator"),
            ("Name", "Attribute"),
        ):
            role = "function"
        elif token_type[:2] in (("Name", "Namespace"), ("Name", "Tag")):
            role = "keyword"
        elif token_type[:2] == ("Generic", "Inserted"):
            role = "syntax_added"
        elif token_type[:2] == ("Generic", "Deleted"):
            role = "syntax_removed"
        elif token_type[:1] == ("Error",):
            role = "syntax_error"
        else:
            role = "code"
        return role

    @override
    def get_background_style(self) -> Style:
        base = self._syntax.get_background_style() if self._syntax else Style()
        return base + self.style("code").background_style

    def rich_theme(self) -> RichTheme:
        """Return every Markdown color through semantic roles."""
        return self._rich

    def prompt_style(self) -> PromptStyle:
        """Return the stable live-pane palette."""
        return self._prompt

    def _markdown_styles(self) -> dict[str, Style]:
        styles = {
            "markdown.paragraph": self.style("text"),
            "markdown.text": self.style("text"),
            "markdown.em": self.style("text") + Style(italic=True),
            "markdown.emph": self.style("text") + Style(italic=True),
            "markdown.strong": self.style("text") + Style(bold=True),
            "markdown.s": self.style("text") + Style(strike=True),
            "markdown.h1": self.style("heading") + Style(underline=True),
            "markdown.h2": self.style("section") + Style(underline=True),
            "markdown.h3": self.style("section") + Style(bold=True),
            "markdown.h4": self.style("section") + Style(italic=True),
            "markdown.h5": self.style("text") + Style(italic=True),
            "markdown.h6": self.style("muted"),
            "markdown.h7": self.style("muted") + Style(italic=True),
        }
        for name, role in (
            ("block_quote", "quote"),
            ("list", "list"),
            ("item", "text"),
            ("item.bullet", "bullet"),
            ("item.number", "list_number"),
            ("hr", "muted"),
            ("h1.border", "text"),
            ("link", "link"),
            ("link_url", "link_url"),
            ("table.border", "table_border"),
            ("table.header", "table_header"),
            ("kbd", "kbd"),
        ):
            styles[f"markdown.{name}"] = self.style(role)
        styles["markdown.code"] = self.style("code") + self._styles["inline_code"]
        styles["markdown.code_block"] = self.style("code") + self._styles["block_code"]
        styles["markdown.link_url"] += Style(underline=True)
        styles["markdown.kbd"] += Style(bold=True)
        return styles


def _prompt(style: Style) -> str:
    parts = [f"fg:{_color(style.color)}", f"bg:{_color(style.bgcolor)}"]
    for name, enabled in (
        ("bold", style.bold),
        ("italic", style.italic),
        ("underline", style.underline),
        ("blink", style.blink),
        ("reverse", style.reverse),
        ("hidden", style.conceal),
        ("strike", style.strike),
        ("dim", style.dim),
    ):
        if enabled is not None:
            parts.append(name if enabled else f"no{name}")
    return " ".join(parts)


def _color(color: Color | None) -> str:
    if color is None or color.is_default:
        return "default"
    if color.is_system_defined and color.number is not None:
        return _ANSI[color.number]
    return color.get_truecolor().hex


def load_theme(name: str = "default") -> Theme:
    """Load a bundled preset or a partial custom theme by safe basename.

    Args:
      name: A preset or custom YAML basename; default restores the original colors.

    Returns:
      theme: The resolved semantic palette and its rendering adapters.

    Raises:
      ValueError: The name, file, schema, role, or style is invalid.

    """
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid theme name: {name!r}")
    if name in _PRESETS:
        return _bundled(name)
    path = config_dir() / "rekursiv-ai" / "sagent" / "themes" / f"{name}.yaml"
    try:
        return _custom(name, path)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"theme {name!r} ({path}): {error}") from error


def _custom(name: str, path: Path) -> Theme:
    if path.is_symlink():
        raise ValueError("theme files must not be symbolic links")
    document = _read(path)
    if document.preset not in _PRESETS:
        raise ValueError(f"unknown theme preset: {document.preset!r}")
    base = _bundled(document.preset)
    return Theme(
        name,
        {**base.styles, **_styles(document.styles)},
        syntax=base.syntax if document.syntax is None else document.syntax,
        source_tokens={**base.source_tokens, **_source_styles(document.source_tokens)},
    )


@cache
def _bundled(name: str) -> Theme:
    path = _CWD / "assets" / "themes" / f"{name}.yaml"
    document = _read(path)
    styles = _styles(document.styles)
    if name != "terminal":
        styles = {**_bundled("terminal").styles, **styles}
    if set(styles) != set(_ROLES):
        raise ValueError(f"theme {name!r} is missing semantic roles")
    return Theme(
        name,
        styles,
        syntax=document.syntax or "",
        source_tokens=_source_styles(document.source_tokens),
    )


def _read(path: Path) -> _Document:
    try:
        given = parse(path.read_text(), source=str(path))
    except YAMLError as error:
        raise ValueError(str(error)) from error
    if given is None or given == {}:
        raise ValueError("expected preset and styles theme fields")
    return from_plain(given, _Document)


def _source_styles(given: Mapping[str, str]) -> dict[str, Style]:
    for token in given:
        if not re.fullmatch(r"[A-Z][A-Za-z]*(?:\.[A-Z][A-Za-z]*)*", token):
            raise ValueError(f"invalid syntax token: {token!r}")
    return {token: _style(token, value) for token, value in given.items()}


def _styles(given: Mapping[str, str]) -> dict[str, Style]:
    unknown = set(given) - set(_ROLES)
    if unknown:
        raise ValueError(f"unknown theme roles: {', '.join(sorted(unknown))}")
    return {role: _style(role, value) for role, value in given.items()}


def _style(role: str, value: str) -> Style:
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"invalid theme style for {role!r}")
    for match in re.finditer(r"rgb\(([^)]*)\)", value):
        channels = match[1].split(",")
        if len(channels) != 3 or any(
            not c.strip().isdigit() or int(c) > 255 for c in channels
        ):
            raise ValueError(f"invalid theme color for {role!r}")
    for color in re.finditer(r"#[^\s]+", value):
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color.group()):
            raise ValueError(f"invalid theme color for {role!r}")
    try:
        style = Style.parse(value)
    except errors.StyleSyntaxError as error:
        raise ValueError(f"invalid theme style for {role!r}: {error}") from error
    if style.link is not None:
        raise ValueError("theme styles cannot contain hyperlinks")
    return style


def theme_choices() -> tuple[str, ...]:
    """List presets followed by safe custom names in the user configuration.

    Returns:
      names: Builtin presets and custom YAML basenames.

    """
    directory = config_dir() / "rekursiv-ai" / "sagent" / "themes"
    names = sorted(
        path.stem
        for path in directory.glob("*.yaml")
        if path.is_file()
        and not path.is_symlink()
        and _NAME.fullmatch(path.stem)
        and path.stem not in _PRESETS
    )
    return (*_PRESETS, *names)


def _remembered() -> Theme:
    path = state_dir() / "rekursiv-ai" / "sagent" / "theme"
    try:
        name = path.read_text().strip()
    except FileNotFoundError:
        return load_theme()
    except (OSError, UnicodeError) as error:
        logging.getLogger(__name__).warning("Cannot read saved theme: %s", error)
        return load_theme()
    try:
        return load_theme(name)
    except ValueError as error:
        logging.getLogger(__name__).warning("Cannot restore saved theme: %s", error)
        return load_theme()


@dataclass(slots=True, kw_only=True)
class ThemeState:
    """The remembered palette, shared by terminal renderers; presets are read-only."""

    current: Theme = field(default_factory=_remembered)

    def select(self, name: str) -> None:
        """Select and remember a theme, copying the current appearance if missing.

        Args:
          name: A preset or safe custom YAML basename.

        Raises:
          ValueError: The theme is invalid or cannot be saved; current is unchanged.

        """
        with _theme_lock():
            try:
                theme = load_theme(name)
            except ValueError as error:
                if not isinstance(error.__cause__, FileNotFoundError):
                    raise
                theme = Theme(
                    name,
                    self.current.styles,
                    syntax=self.current.syntax,
                    source_tokens=self.current.source_tokens,
                )
                self._save(theme, write=True)
            else:
                self._save(theme, write=False)

    def edit(self, name: str, role: str, value: str) -> None:
        """Save one raw semantic style, select it, and fork presets as userN.

        Args:
          name: An existing theme; editing it also selects it.
          role: A semantic role, such as thinking or status.
          value: A Rich style; an empty string clears the role's override.

        Raises:
          ValueError: Validation or saving failed; current is unchanged.

        """
        styles = _styles({role: value})
        with _theme_lock():
            base = load_theme(name)
            if name in _PRESETS:
                index = 1
                directory = config_dir() / "rekursiv-ai" / "sagent" / "themes"
                while (directory / f"user{index}.yaml").exists() or (
                    directory / f"user{index}.yaml"
                ).is_symlink():
                    index += 1
                name = f"user{index}"
            theme = Theme(
                name,
                {**base.styles, **styles},
                syntax=base.syntax,
                source_tokens=base.source_tokens,
            )
            self._save(theme, write=True)

    def prompt_style(self) -> PromptStyle:
        """Resolve the current palette when prompt-toolkit redraws."""
        return self.current.prompt_style()

    def _save(self, theme: Theme, *, write: bool) -> None:
        path = config_dir() / "rekursiv-ai" / "sagent" / "themes" / f"{theme.name}.yaml"
        previous = path.read_bytes() if write and path.exists() else None
        if write:
            write_private(
                path,
                safe_dump(
                    {
                        "preset": "terminal",
                        "styles": {
                            role: str(style) if style else ""
                            for role, style in theme.styles.items()
                        },
                        "syntax": theme.syntax,
                        "source_tokens": {
                            token: str(style) if style else ""
                            for token, style in theme.source_tokens.items()
                        },
                    },
                    sort_keys=False,
                ),
            )
        try:
            write_private(
                state_dir() / "rekursiv-ai" / "sagent" / "theme",
                theme.name + "\n",
            )
        except OSError:
            if write:
                if previous is None:
                    path.unlink()
                else:
                    write_private(path, previous)
            raise
        self.current = theme


@contextmanager
def _theme_lock() -> Generator[None]:
    path = config_dir() / "rekursiv-ai" / "sagent" / "themes" / ".lock"
    try:
        with os.fdopen(open_private(path, os.O_RDWR), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
    except OSError as error:
        raise ValueError(f"cannot save theme: {error}") from error
