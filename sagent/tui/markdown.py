"""Markdown for the scrollback: Rich's, with three defects fixed, and a stream splitter.

1. Rich yields a blank line before a list or quote that opens the text; here
   a blank line comes only between rendered blocks.
2. CommonMark lets only a list starting at 1 interrupt a paragraph, so a
   continued numbering under a bold heading ran into the paragraph as prose.
3. Rich fills a code block to the window, which reflows when it shrinks;
   here code wraps at 100 and the fill stops at the longest line, or 80.

``TightMarkdown.__rich_console__`` is a copy of Rich's, changed for (1); a
test renders both over a corpus to catch the copy drifting from Rich.

Typical usage example:

  console.print(TightMarkdown(text))
  cut = stable_boundary(streamed)       # Render streamed[:cut] now.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from markdown_it import MarkdownIt
from markdown_it.rules_block.list import list_block, skipOrderedListMarker
from rich.cells import cell_len
from rich.markdown import (
    CodeBlock,
    Link,
    Markdown,
    MarkdownContext,
    MarkdownElement,
    UnknownElement,
)
from rich.segment import Segment
from rich.style import Style as RichStyle
from rich.syntax import Syntax

from sagent.tui.types import FILLS, Sticky


if TYPE_CHECKING:
    from markdown_it.rules_block.state_block import StateBlock
    from markdown_it.token import Token
    from rich.console import Console, ConsoleOptions, JustifyMethod, RenderResult
    from rich.style import Style


__all__ = [
    "TightMarkdown",
    "stable_boundary",
]


class FitCodeBlock(CodeBlock):
    """A code block wrapping its code at 100, filled to 80 or, once wide, its cap."""

    @override
    def __init__(
        self,
        lexer_name: str,
        theme: str,
        sticky: Sticky | None = None,
    ) -> None:
        super().__init__(lexer_name, theme)
        self.sticky = sticky if sticky is not None else Sticky()

    @override
    @classmethod
    def create(cls, markdown: Markdown, token: Token) -> FitCodeBlock:
        lexer_name = (token.info or "").partition(" ")[0]
        sticky = markdown.sticky if isinstance(markdown, TightMarkdown) else Sticky()
        return cls(lexer_name or "text", markdown.code_theme, sticky)

    @override
    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        code = str(self.text).rstrip()
        longest = max((cell_len(line) for line in code.splitlines()), default=0)
        width = self.sticky.cells(FILLS["code"], longest, width=options.max_width)
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=1,
        )
        yield from console.render(syntax, options.update_width(width))


class TightMarkdown(Markdown):
    """Markdown without a leading blank line or swallowed ordered lists."""

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "fence": FitCodeBlock,
        "code_block": FitCodeBlock,
    }

    @override
    def __init__(
        self,
        markup: str,
        code_theme: str = "monokai",
        justify: JustifyMethod | None = None,
        style: str | Style = "none",
        hyperlinks: bool = True,
        inline_code_lexer: str | None = None,
        inline_code_theme: str | None = None,
        sticky: Sticky | None = None,
    ) -> None:
        self.sticky = sticky if sticky is not None else Sticky()
        super().__init__(
            markup,
            code_theme=code_theme,
            justify=justify,
            style=style,
            hyperlinks=hyperlinks,
            inline_code_lexer=inline_code_lexer,
            inline_code_theme=inline_code_theme,
        )
        # Rich builds its parser inline in ``__init__`` and exposes no hook,
        # so the lenient list rule can only be installed by re-parsing. The
        # enables mirror Rich's own; dropping either would silently lose
        # strikethrough or table support.
        parser = MarkdownIt().enable("strikethrough").enable("table")
        parser.block.ruler.at(
            "list",
            _lenient_list_block,
            {"alt": ["paragraph", "reference", "blockquote"]},
        )
        self.parsed = parser.parse(markup)

    @override
    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        style = console.get_style(self.style, default="none")
        options = options.update(height=None)
        context = MarkdownContext(
            console,
            options,
            style,
            inline_code_lexer=self.inline_code_lexer,
            inline_code_theme=self.inline_code_theme,
        )
        new_line = False
        first_render = True
        _new_line_segment = Segment.line()
        inline_style_tags = self.inlines

        for token in self._flatten_tokens(self.parsed):
            node_type = token.type
            tag = token.tag
            entering = token.nesting == 1
            exiting = token.nesting == -1
            self_closing = token.nesting == 0

            if node_type == "text":
                context.on_text(token.content, node_type)
            elif node_type == "hardbreak":
                context.on_text("\n", node_type)
            elif node_type == "softbreak":
                context.on_text(" ", node_type)
            elif node_type == "link_open":
                href = str(token.attrs.get("href", ""))
                if self.hyperlinks:
                    link_style = console.get_style(
                        "markdown.link_url",
                        default="none",
                    ) + RichStyle(link=href)
                    context.enter_style(link_style)
                else:
                    context.stack.push(Link.create(self, token))
            elif node_type == "html_inline":
                # Without this branch inline HTML falls through to
                # ``UnknownElement``, whose ``new_line = True`` injects a
                # blank line per tag and drops the ``<kbd>`` styling.
                if token.content == "<kbd>":
                    context.enter_style(
                        console.get_style("markdown.kbd", default="bold"),
                    )
                elif token.content == "</kbd>":
                    context.leave_style()
                else:
                    continue
            elif node_type == "link_close":
                if self.hyperlinks:
                    context.leave_style()
                else:
                    element = context.stack.pop()
                    assert isinstance(element, Link)
                    context.enter_style(
                        console.get_style("markdown.link", default="none"),
                    )
                    context.on_text(element.text.plain, node_type)
                    context.leave_style()
                    context.on_text(" (", node_type)
                    context.enter_style(
                        console.get_style("markdown.link_url", default="none"),
                    )
                    context.on_text(element.href, node_type)
                    context.leave_style()
                    context.on_text(")", node_type)
            elif tag in inline_style_tags and node_type not in ("fence", "code_block"):
                if entering:
                    context.enter_style(f"markdown.{tag}")
                elif exiting:
                    context.leave_style()
                else:
                    context.enter_style(f"markdown.{tag}")
                    # Rich guards an empty ``content`` here; CommonMark has no
                    # empty code span, and appending "" would change nothing.
                    context.on_text(token.content, node_type)
                    context.leave_style()
            else:
                element_class = self.elements.get(token.type) or UnknownElement
                element = element_class.create(self, token)

                if entering or self_closing:
                    context.stack.push(element)
                    element.on_enter(context)

                if exiting:
                    element = context.stack.pop()
                    should_render = not context.stack or (
                        context.stack
                        and context.stack.top.on_child_close(context, element)
                    )
                    if should_render:
                        if new_line and not first_render:
                            yield _new_line_segment
                        yield from console.render(element, context.options)
                        first_render = False
                elif self_closing:
                    context.stack.pop()
                    # Rich checks ``content`` for ``None``; markdown-it types it
                    # ``str`` and defaults it to "".
                    element.on_text(context, token.content)
                    should_render = not context.stack or (
                        context.stack
                        and context.stack.top.on_child_close(context, element)
                    )
                    if should_render:
                        if new_line and node_type != "inline" and not first_render:
                            yield _new_line_segment
                        yield from console.render(element, context.options)
                        first_render = False

                if exiting or self_closing:
                    element.on_leave(context)
                    new_line = element.new_line


def stable_boundary(text: str) -> int:
    """Return where the last block of streamed markdown starts.

    Everything before it is finished: more text cannot change how it renders.

    Args:
      text: The reply so far.

    Returns:
      offset: Characters that are safe to render; ``0`` before two blocks.

    """
    starts = [
        t.map[0] for t in MarkdownIt().parse(text) if t.map is not None and t.level == 0
    ]
    if len(starts) < 2:
        return 0
    lines = text.split("\n")
    return sum(len(lines[i]) + 1 for i in range(starts[-1]))


# Across 133,899 logged replies this applied 244 times; the wrapped number it guards
# against (a line wrapping onto ``2024.``) occurred zero times. ``list_block`` reads the
# restriction off ``state.parentType``, and returns before tokenizing when ``silent``,
# so no nested parse sees the swap.
def _lenient_list_block(
    state: StateBlock,
    start_line: int,
    end_line: int,
    silent: bool,
) -> bool:
    """Let an ordered list interrupt a paragraph at any start number."""
    saved = state.parentType
    # Only the start number is relaxed. Swapping for every marker also let an
    # empty item (``*`` or ``2.`` alone on a line) cut a paragraph in two.
    after = skipOrderedListMarker(state, start_line)
    if (
        silent
        and saved == "paragraph"
        and after >= 0
        and state.skipSpaces(after) < state.eMarks[start_line]
    ):
        state.parentType = "root"
    try:
        return list_block(state, start_line, end_line, silent)
    finally:
        state.parentType = saved
