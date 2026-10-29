"""OpenAI models over the Responses API, billed to a ChatGPT plan.

The login is the one ``codex login`` keeps, and requests reach the endpoint
the Codex CLI uses. Costs are priced at API rates, as what the session would
have cost on a key; the plan itself is billed flat.

Typical usage example:

  async with OpenAISub() as openai:             # ~/.codex/auth.json
      await openai.login()                       # Once, or after it is revoked.
      sol = openai["sol"]
      openai.usage                               # The plan's windows, as reported.

  work = OpenAISub(credentials=Path("~/.codex/auth-work.json"))
  child = OpenAISub(lease=parent_login.lend)   # Borrows; never renews.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, NotRequired, Protocol, TypedDict, override

from treekle import json_dumps, json_loads

from sagent.model.types import Request
from sagent.providers.openai import token_count, wire
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.lease import BorrowedLogin
from sagent.providers.openai.login import ORIGINATOR, CodexLogin


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from sagent.providers.openai.compat import OpenAIModel
    from sagent.providers.openai.lease import TokenLease
    from sagent.providers.types import ProviderSharedResources


__all__ = [
    "OpenAISub",
]


class OpenAISub(OpenAI):
    """OpenAI's models, billed to a ChatGPT plan.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      credentials: A login file; ``None`` uses the login ``codex`` keeps.
      lease: Lends tokens from a login another process holds; ``None`` holds
        the login here, and renews it.

    """

    section: ClassVar[str] = "openai-sub"
    url: ClassVar[str] = "https://chatgpt.com/backend-api/codex"
    key_variable: ClassVar[str] = ""
    # The endpoint answers 400 "Unsupported parameter: temperature" for any model.
    temperature: ClassVar[float | None] = None

    codex_version: ClassVar[str] = "0.153.0"
    """The Codex CLI release claimed."""

    user_agent: ClassVar[str] = (
        f"{ORIGINATOR}/{codex_version} (Ubuntu 26.4.0; x86_64) gnome-terminal"
    )
    """What ``codex`` sends, as captured from that release on 2026-10-27."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
        credentials: Path | None = None,
        lease: TokenLease | None = None,
    ) -> None:
        super().__init__(name=name, resources=resources)
        self._login: _Login = (
            CodexLogin(credentials, resources=self.resources)
            if lease is None
            else BorrowedLogin(lease)
        )
        # By agent: its turn's id, and the token the server routes that turn by.
        self._turn_states: dict[str, tuple[str, str]] = {}

    async def login(self, *, headless: bool = False) -> None:
        """Sign in through the browser, or with the redirect pasted from it."""
        await self._login.sign_in(headless=headless)

    @override
    async def credentials(self) -> dict[str, str]:
        token, account = await self._login.token()
        # The endpoint answers without the last three.
        return {
            "authorization": f"Bearer {token}",
            "chatgpt-account-id": account,
            "originator": ORIGINATOR,
            "user-agent": self.user_agent,
        }

    @override
    async def reauthenticate(self) -> bool:
        await self._login.renew()
        return True

    @override
    def headers(self, request: Request) -> dict[str, str]:
        # The endpoint routes its prompt cache by ``session_id``, beside the key in
        # the body; the key alone, or beside Codex's newer ``session-id``,
        # cached nothing when probed on 2026-10-27.
        headers = {
            "session_id": wire.cache_key(request),
            "x-codex-turn-metadata": json_dumps(_turn_metadata(request)),
        }
        if request.session:
            headers["session-id"] = request.session
            headers["x-client-request-id"] = request.call_id
            headers["x-codex-window-id"] = f"{request.agent}:{request.window}"
            turn, state = self._turn_states.get(request.agent, ("", ""))
            if turn == request.prompt_id and state:
                headers["x-codex-turn-state"] = state
        if request.parent:
            headers["x-openai-subagent"] = "collab_spawn"
            headers["x-codex-parent-thread-id"] = request.parent
        return headers

    @override
    def answered(
        self,
        sent: Mapping[str, str],
        received: Mapping[str, str],
    ) -> None:
        state = received.get("x-codex-turn-state", "")
        if state and "session-id" in sent:
            metadata = json_loads(sent["x-codex-turn-metadata"], _TurnMetadata)
            thread, turn = metadata.get("thread_id"), metadata.get("turn_id")
            # A turn's token routes only that turn, so each agent keeps its
            # latest turn's.
            if thread is not None and turn is not None:
                self._turn_states[thread] = (turn, state)

    @override
    async def count(self, request: Request, *, model: OpenAIModel) -> int:
        """Return an estimate of the tokens ``request`` would send; this endpoint cannot count."""
        return token_count.estimate(
            request,
            capability=model.capability,
            encoding="o200k_base",
        )


class _Compaction(TypedDict):
    trigger: str


class _TurnMetadata(TypedDict):
    """The fields of Codex's ``x-codex-turn-metadata`` this provider sends."""

    session_id: NotRequired[str]
    thread_id: NotRequired[str]
    agent_name: NotRequired[str]
    turn_id: NotRequired[str]
    root_turn_id: NotRequired[str]
    turn_started_at_unix_ms: NotRequired[int]
    window_id: NotRequired[str]
    window_number: NotRequired[int]
    request_kind: str
    compaction: NotRequired[_Compaction]
    parent_thread_id: NotRequired[str]
    parent_turn_id: NotRequired[str]
    subagent_kind: NotRequired[str]
    thread_source: NotRequired[str]


# As Codex 0.153.0 sends it, captured on 2026-10-27; every field is optional to
# the server (codex-rs core/src/responses_metadata.rs).
def _turn_metadata(request: Request) -> _TurnMetadata:
    """Return what Codex says of the turn ``request`` is a call of."""
    metadata: _TurnMetadata = {
        "request_kind": "turn" if request.purpose == "main" else request.purpose,
    }
    if request.session:
        metadata["session_id"] = request.session
        metadata["thread_id"] = request.agent
        metadata["agent_name"] = request.name
        metadata["turn_id"] = request.prompt_id
        metadata["root_turn_id"] = request.root_prompt or request.prompt_id
        metadata["turn_started_at_unix_ms"] = request.turn_started_ms
        metadata["window_id"] = f"{request.agent}:{request.window}"
        metadata["window_number"] = request.window
    if request.purpose == "compaction" and request.compaction:
        metadata["compaction"] = {"trigger": request.compaction}
    if request.parent:
        metadata["parent_thread_id"] = request.parent
        metadata["parent_turn_id"] = Request(
            agent=request.parent,
            turn=request.parent_turn,
        ).prompt_id
        metadata["subagent_kind"] = "thread_spawn"
        metadata["thread_source"] = "subagent"
    return metadata


class _Login(Protocol):
    """What this provider asks of a login, held here or borrowed."""

    async def token(self) -> tuple[str, str]:
        """Return an access token, and the account it is for."""
        ...

    async def renew(self) -> None:
        """Replace the access token the server refused."""
        ...

    async def sign_in(self, *, headless: bool = False) -> None:
        """Sign in, interactively; ``headless`` skips the browser."""
        ...
