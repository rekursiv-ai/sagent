"""Antigravity CLI configuration for the generic ACP client."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import os
import shutil

from sagent.acp.client import AcpClient, AcpPromptResult, PermissionDecision
from sagent.lib.userdirs import data_dir


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sagent.acp.client import PermissionHandler, RequestHandler, UpdateHandler


def find_agy_binary(command_entry: str = "agy") -> str:
    """Find the path to the agy executable.

    Checks PATH first, then standard installation paths like ~/.local/bin/agy.
    """
    found = shutil.which(command_entry)
    if found is not None:
        return found
    fallback = data_dir().parent / "bin" / command_entry
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    local_bin = Path.home() / ".local" / "bin" / command_entry
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin)
    return command_entry


@dataclass(frozen=True, slots=True, kw_only=True)
class AgyAcp:
    """Build an ACP client for the subscription-backed ``agy acp`` server.

    ``agy acp`` reuses the credentials cached by an interactive agy login.
    """

    command: tuple[str, ...] = ("agy", "acp")
    dangerously_skip_permissions: bool = False
    agent: str | None = None
    model: str | None = None
    effort: str | None = None
    add_dirs: tuple[Path, ...] = ()
    client_name: str = "sagent"
    client_version: str = "0.2.1"

    def resolved_command(self) -> tuple[str, ...]:
        """Return command argv with flags and resolved binary path."""
        base_cmd = list(self.command)
        if base_cmd and base_cmd[0] == "agy":
            base_cmd[0] = find_agy_binary("agy")
        args: list[str] = list(base_cmd)
        if self.dangerously_skip_permissions and "--dangerously-skip-permissions" not in args:
            args.append("--dangerously-skip-permissions")
        if self.agent is not None and "--agent" not in args:
            args.extend(("--agent", self.agent))
        for directory in self.add_dirs:
            args.extend(("--add-dir", str(directory)))
        return tuple(args)

    def client(
        self,
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        on_update: UpdateHandler | None = None,
        on_request: RequestHandler | None = None,
        on_permission: PermissionHandler | None = None,
    ) -> AcpClient:
        """Return an unopened ACP client for agy."""
        perm_handler = on_permission
        if perm_handler is None and self.dangerously_skip_permissions:
            perm_handler = PermissionDecision.auto_allow
        return AcpClient(
            command=self.resolved_command(),
            cwd=cwd,
            env=env,
            client_name=self.client_name,
            client_version=self.client_version,
            on_update=on_update,
            on_request=on_request,
            on_permission=perm_handler,
        )

    async def prompt(
        self,
        prompt: Sequence[Mapping[str, Any]] | str,
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        on_update: UpdateHandler | None = None,
        on_permission: PermissionHandler | None = None,
    ) -> AcpPromptResult:
        """Run a single prompt in a new ACP session and return its result."""
        async with self.client(
            cwd=cwd,
            env=env,
            on_update=on_update,
            on_permission=on_permission,
        ) as client:
            session = await client.new_session()
            return await client.prompt(session, prompt)


__all__ = ["AgyAcp", "find_agy_binary"]
