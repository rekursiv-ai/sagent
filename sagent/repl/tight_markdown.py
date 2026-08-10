"""Markdown subclass that fixes two Rich rendering defects.

1. Spurious leading blank line. Rich's ``Markdown.__rich_console__``
   tracks a ``new_line`` flag that is set by *close* tokens
   (``paragraph_close``, etc.) via ``UnknownElement`` (``new_line =
   True``).  Because the flag is set before the first *rendered* block, a
   list or blockquote that opens the document gets a blank line above it.
   Fixed with a ``first_render`` guard so the ``new_line`` segment is only
   yielded between rendered blocks, never before the first one.

2. Ordered lists that do not start at 1 are swallowed into the preceding
   paragraph. CommonMark only lets an ordered list interrupt a paragraph
   when it starts at 1, so a model emitting a continued numbering run
   under a bold sub-heading renders as run-on prose. Fixed by
   :func:`_lenient_list_block`.

Because item 1 is a whole-method override, this module carries a copy of
Rich's ``__rich_console__``. ``tight_markdown_test`` renders both this
class and stock ``Markdown`` over a corpus of constructs and asserts they
agree except on an explicit allowlist -- that differential is what
detects the copy drifting from a newer Rich.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from markdown_it import MarkdownIt
from markdown_it.rules_block.list import list_block
from rich.markdown import (
    Link,
    Markdown,
    MarkdownContext,
    UnknownElement,
)
from rich.segment import Segment
from rich.style import Style as RichStyle


if TYPE_CHECKING:
    from markdown_it.rules_block.state_block import StateBlock
    from rich.console import Console, ConsoleOptions, JustifyMethod, RenderResult
    from rich.style import Style


class TightMarkdown(Markdown):
    """Markdown without a leading blank line or swallowed ordered lists."""

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
    ) -> None:
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
        self, console: Console, options: ConsoleOptions
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
                        "markdown.link_url", default="none"
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
                        console.get_style("markdown.kbd", default="bold")
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
                        console.get_style("markdown.link", default="none")
                    )
                    context.on_text(element.text.plain, node_type)
                    context.leave_style()
                    context.on_text(" (", node_type)
                    context.enter_style(
                        console.get_style("markdown.link_url", default="none")
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
                    if token.content:
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
                    text = token.content
                    if text is not None:  # pyright: ignore[reportUnnecessaryComparison] -- vendored from Rich
                        element.on_text(context, text)
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


def _lenient_list_block(
    state: StateBlock, start_line: int, end_line: int, silent: bool
) -> bool:
    """Let an ordered list interrupt a paragraph at any start number.

    CommonMark restricts a paragraph-interrupting ordered list to one
    starting at 1, so that a line wrapping onto ``2024.`` is not read as a
    list. Assistant turns break the other way: a continued numbering run
    under a bold sub-heading is common and renders as run-on prose. Across
    133,899 logged assistant messages the fix applied 244 times and the
    wrapped-number case it protects against occurred zero times.

    Upstream reads the restriction off ``state.parentType``, so clearing
    it for the probe is what disables the check. The probe is
    non-recursive -- ``list_block`` returns at ``if silent`` before it
    tokenizes children -- so no nested parse ever observes the swap.

    Args:
      state: Block-parser state; ``parentType`` is swapped and restored.
      start_line: First line of the candidate list.
      end_line: Line after the last one available to this rule.
      silent: True when upstream is only probing whether a list starts
          here, which is the only mode that consults ``parentType``.

    Returns:
      matched: True when the lines form a list, per upstream.

    """
    saved = state.parentType
    if silent and saved == "paragraph":
        state.parentType = "root"
    try:
        return list_block(state, start_line, end_line, silent)
    finally:
        state.parentType = saved
