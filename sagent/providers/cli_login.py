"""Sign in through a vendor CLI's own sign-in, at sagent's prompt.

A subscription's terms let only the vendor's own client sign in to it, so the
CLI does the whole sign-in -- the code exchange, the stored login, its renewal
-- and sagent only shows the address the CLI prints and hands it the code the
user pastes. A CLI that finishes on its own, as ``claude`` does when the
browser comes back to its loopback server, ends the prompt.

Typical usage example:

  await cli_sign_in("agy", "acp", "--login", browser=True, grace_sec=1.0)
  await cli_sign_in("claude", "auth", "login", env=env, browser=False, grace_sec=1.0)
"""

from __future__ import annotations

from asyncio import StreamWriter
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import TYPE_CHECKING

import asyncio
import re

from sagent.lib.posix.child import Child
from sagent.model.exceptions import AuthError
from sagent.providers.login import open_browser, read_login_input


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


__all__ = [
    "cli_sign_in",
]


async def cli_sign_in(
    *argv: str,
    env: Mapping[str, str] | None = None,
    browser: bool,
    show: Callable[[str], object] = print,
    grace_sec: float,
) -> None:
    """Run the CLI's sign-in, showing its address and passing it the user's code.

    Args:
      argv: The CLI's sign-in command, which prints an address and reads a code.
      env: Its environment; ``None`` inherits this one.
      browser: Open the address in the user's browser too; ``False`` where the
        CLI opens one itself, or none would reach the user.
      show: Where the address is shown.
      grace_sec: How long a failed CLI's last words are waited for, never to
        the end: a browser it started may hold its pipes open past its exit.

    Raises:
      AuthError: The CLI exited non-zero; its last line says why.
      EOFError: Nothing was typed at the prompt.

    """
    async with await Child.start(
        *argv,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        env=env,
    ) as child:
        said: list[str] = []
        address = asyncio.Future[str]()
        streams = (child.process.stdout, child.process.stderr)
        reading = tuple(
            asyncio.create_task(_read(s, said=said, address=address)) for s in streams
        )
        exited = asyncio.create_task(child.exited())
        try:
            await asyncio.wait((address, exited), return_when=asyncio.FIRST_COMPLETED)
            if address.done():
                url = address.result()
                show(
                    f"Sign in at this address, then paste the code it shows:\n\n  {url}\n",
                )
                if browser:
                    open_browser(url)
                await _answer(child, exited=exited)
            status = await exited
            if status:
                await asyncio.wait(reading, timeout=grace_sec)
                name = Path(argv[0]).name
                raise AuthError(
                    f"{name} sign-in failed: {said[-1]}"
                    if said
                    else f"{name} sign-in failed (exit {status}).",
                )
        finally:
            for task in (exited, *reading):
                task.cancel()
            address.cancel()


async def _read(
    stream: asyncio.StreamReader | None,
    *,
    said: list[str],
    address: asyncio.Future[str],
) -> None:
    """Keep each whole line ``stream`` says in ``said``; set ``address`` at the first."""
    assert isinstance(stream, asyncio.StreamReader)
    async for raw in stream:
        line = raw.decode(errors="replace").strip() if raw.endswith(b"\n") else ""
        if line:
            said.append(line)
        if not address.done() and (found := re.search(r"https://\S+", line)):
            address.set_result(found[0])


async def _answer(child: Child, *, exited: asyncio.Task[int]) -> None:
    """Pass the CLI the code typed at the prompt, unless it ``exited`` first."""
    prompt = asyncio.create_task(read_login_input("Code: "))
    try:
        await asyncio.wait((prompt, exited), return_when=asyncio.FIRST_COMPLETED)
        if exited.done():
            return
        stdin = child.process.stdin
        assert isinstance(stdin, StreamWriter)
        stdin.write(f"{prompt.result()}\n".encode())
        await stdin.drain()
    finally:
        prompt.cancel()
        _ = await asyncio.wait((prompt,))
