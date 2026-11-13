"""Interact with wesearch's persistent Chrome through the calling session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

import asyncio

from treekle import PlainTree, json_dumps
from wesearch import browser
from wesearch.browser import BrowserAction, BrowserError, BrowserMode
from wesearch.mcp.types import BrowserParamsSchema, browser_data
from wesearch.types.errors import FetchError

from sagent.session.types import Attachment, WebBrowserResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    Arg,
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.tools.types import ToolContext


__all__ = ["WebBrowser"]


@dataclass(frozen=True, slots=True, kw_only=True)
class WebBrowser:
    """Interact with a persistent tab without owning Chrome's lifecycle."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        action: Annotated[BrowserAction, BrowserParamsSchema.action.description]
        tab_id: Annotated[str, BrowserParamsSchema.tab_id.description] = ""
        url: Annotated[str, BrowserParamsSchema.url.description] = ""
        selector: Annotated[str, BrowserParamsSchema.selector.description] = ""
        text: Annotated[str, BrowserParamsSchema.text.description] = ""
        key: Annotated[str, BrowserParamsSchema.key.description] = ""
        mode: Annotated[BrowserMode | None, BrowserParamsSchema.mode.description] = None
        restart: Annotated[bool, BrowserParamsSchema.restart.description] = False
        max_chars: Annotated[
            int,
            BrowserParamsSchema.max_chars.description,
            Arg(minimum=1),
        ] = 20_000
        timeout_sec: Annotated[float, BrowserParamsSchema.timeout_sec.description] = 30

    name = "WebBrowser"
    description = (
        "Interact with the existing persistent Chrome. List tabs, open/navigate, "
        "inspect visible text and controls, click, fill, press a key, screenshot, "
        "or explicitly close a tab. Use stable tab ids and inspection selectors. "
        "Action='mode' reports actual mode; supply mode to request headed/headless. "
        "Changing mode requires restart=True and user consent before interrupting "
        "shared Chrome. Restart invalidates tab ids, restores tabs best-effort, "
        "and preserves the profile. Otherwise tabs survive calls and disconnects. "
        "Screenshots are native images. Reads need no approval. Obtain user approval "
        "before purchases, accepting terms, account changes, or other consequential "
        "externally visible actions. "
        "Never include filled text in summaries; it may be a password or key."
    )
    schema = json_schema(Arguments)
    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return the action, target/mode, and selector, never filled text."""
        if arguments.get("action") == "mode":
            return f"WebBrowser mode {arguments.get('mode') or ''}".rstrip()
        target = arguments.get("tab_id") or arguments.get("url", "")
        return f"WebBrowser {arguments.get('action', '')} {target} {arguments.get('selector', '')}".rstrip()

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Serialize tabs independently, and an explicit restart against all calls."""
        del context
        if arguments.get("action") == "mode" and arguments.get("restart") is True:
            return "*"
        tab_id = arguments.get("tab_id")
        return f"browser:{tab_id}" if isinstance(tab_id, str) and tab_id else None

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[WebBrowserResult, Attachment | str, str]:
        """Forward one action through the calling agent's fetch session.

        Args:
          context: The calling agent's state and shared fetch session.
          arguments: What the model passed, as WebBrowser.Arguments.

        Returns:
          streaming_result: Structured inspection text with any PNG as an attachment.

        """
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[WebBrowserResult, Attachment | str, str]:
        a = parse(arguments, WebBrowser.Arguments)
        try:
            result = await asyncio.to_thread(
                browser.browser,
                a.action,
                tab_id=a.tab_id,
                url=a.url,
                selector=a.selector,
                text=a.text,
                key=a.key,
                mode=a.mode,
                restart=a.restart,
                max_chars=a.max_chars,
                timeout_sec=a.timeout_sec,
                session=context.fetch_session,
            )
        except (BrowserError, FetchError, ValueError, OSError) as error:
            return ToolOutput(
                stderr=(f"WebBrowser failed: {error}",),
                result=WebBrowserResult(action=a.action, exit_code=1),
            )
        return ToolOutput(
            stdout=(
                json_dumps(browser_data(result)),
                *(
                    (Attachment(mime_descriptor="image/png", data=result.screenshot),)
                    if result.screenshot
                    else ()
                ),
            ),
            result=WebBrowserResult(action=a.action),
        )
