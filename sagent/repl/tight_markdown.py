"""Markdown subclass that suppresses Rich's spurious leading blank line.

Rich's ``Markdown.__rich_console__`` tracks a ``new_line`` flag that is
set by *close* tokens (``paragraph_close``, etc.) via ``UnknownElement``
(``new_line = True``).  Because the flag is set before the first
*rendered* block, a list or blockquote that opens the document gets a
blank line above it.

Fix: add a ``first_render`` guard so the ``new_line`` segment is only
yielded between rendered blocks, never before the first one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markdown import (
    Link,
    Markdown,
    MarkdownContext,
    UnknownElement,
)
from rich.segment import Segment
from rich.style import Style as RichStyle


if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult


class TightMarkdown(Markdown):
    """Markdown without a spurious leading blank line."""

    def __rich_console__(  # pyright: ignore[reportImplicitOverride] -- vendored override
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
