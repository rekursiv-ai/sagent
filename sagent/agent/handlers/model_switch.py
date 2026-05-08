"""``ModelSwitchHandler``: handle ``/model`` slash command.

Parses the ``/model [args...]`` argument string, builds a new
provider/model, and calls ``agent.swap_model``. With no args it
reports the current spec to the printer's tool-error sink.

REPL-local commands like ``/model`` write directly to the printer
rather than round-tripping through the inbox -- that way their
output appears immediately, before any queued user messages or
model calls run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import dataclasses
import shlex

from sagent.agent.handlers.base import InlineHandler
from sagent.providers import build_provider, infer_provider


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Message, ModelSpec
    from sagent.repl.render import Printer


class ModelSwitchHandler(InlineHandler):
    """Handle ``text/x-model-switch-request``: swap provider/model in place.

    Constructor takes an optional ``Printer``: when supplied, status
    lines (current spec, swap confirmation, errors) write to it
    directly. Without a printer, the handler is a no-op for output
    (model swap still happens for valid args).
    """

    descriptors: tuple[str, ...] = ("text/x-model-switch-request",)

    def __init__(self, printer: Printer | None = None) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        """Parse args, build new model, swap; report status to the printer."""
        args_str = str(msg.content)
        spec = agent.model_spec
        if spec is None:
            self._write("[/model] agent has no model spec; cannot swap.")
            return
        try:
            tokens = shlex.split(args_str)
        except ValueError as e:
            self._write(f"[/model] parse error: {e}")
            return
        if not tokens:
            self._write(
                f"[/model] provider={spec.provider} auth={spec.auth} "
                f"model={spec.model_id} account={spec.account or 'default'}",
            )
            return
        result = _parse_args(tokens, spec)
        if isinstance(result, str):
            self._write(result)
            return
        prov_name, auth, account, model_id = result
        if model_id and prov_name == spec.provider:
            inferred = infer_provider(model_id, prov_name)
            if inferred is not None:
                prov_name, auth = inferred
        try:
            provider = build_provider(prov_name, auth, account=account)
            new_model = provider.model(model_id)
        except (AttributeError, RuntimeError, ValueError) as exc:
            self._write(f"[/model] {exc}")
            return
        old_id = agent.model.model_id
        agent.swap_model(
            new_model,
            spec=dataclasses.replace(
                spec,
                provider=prov_name,
                auth=auth,
                model_id=new_model.model_id,
                account=account,
            ),
        )
        if prov_name != spec.provider:
            label = f"{spec.provider}/{old_id} -> {prov_name}/{new_model.model_id}"
        else:
            label = f"{old_id} -> {new_model.model_id}"
        self._write(f"[/model] {label}")

    def _write(self, line: str) -> None:
        """Forward a status line to the printer if configured."""
        if self._printer is not None:
            self._printer.write_line(line)


_FLAG_PROVIDER = ("--provider", "-p")
_FLAG_AUTH = ("--auth", "-a")
_FLAG_ACCOUNT = ("--account",)
_KV_KEYS = {"provider", "auth", "account", "model", "model_id"}


def _parse_args(
    tokens: list[str],
    spec: ModelSpec,
) -> tuple[str, str, str | None, str | None] | str:
    """Parse ``/model`` arguments.

    Accepts three interchangeable forms (mix-and-match in one call):
      - ``--provider P --auth A --account ACCT MODEL_ID`` (flag form)
      - ``provider=P auth=A account=ACCT model=MODEL_ID`` (kv form;
        re-parseable from the status line ``/model`` prints when called
        with no args)
      - bare ``MODEL_ID`` (positional; inherits other fields from spec)

    Returns ``(prov_name, auth, account, model_id)`` on success, or an
    error string on failure.
    """
    prov_name = spec.provider
    auth = spec.auth
    account = spec.account
    model_id: str | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _FLAG_PROVIDER and i + 1 < len(tokens):
            prov_name = tokens[i + 1]
            i += 2
            continue
        if tok in _FLAG_AUTH and i + 1 < len(tokens):
            auth = tokens[i + 1]
            i += 2
            continue
        if tok in _FLAG_ACCOUNT and i + 1 < len(tokens):
            account = tokens[i + 1]
            i += 2
            continue
        if "=" in tok and not tok.startswith("-"):
            key, value = tok.split("=", 1)
            if key not in _KV_KEYS:
                return f"[/model] unknown key: {key}"
            if key == "provider":
                prov_name = value
            elif key == "auth":
                auth = value
            elif key == "account":
                # Reverse the display alias: ``account=default`` <->
                # ``account is None``. Also accept empty.
                account = None if value in ("", "default") else value
            else:  # "model" or "model_id"
                model_id = value
            i += 1
            continue
        if not tok.startswith("-"):
            model_id = tok
            i += 1
            continue
        return f"[/model] unknown flag: {tok}"
    if not model_id and prov_name == spec.provider and auth == spec.auth:
        return (
            "[/model] usage: /model [provider=P] [auth=A] [account=ACCT]"
            " [model=MODEL_ID]   (or --provider/--auth/--account flags,"
            " or a bare model_id)"
        )
    return prov_name, auth, account, model_id
