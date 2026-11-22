"""Keep prompt measurements without retaining copies of conversation history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hashlib
import json

from treekle import mutable, to_plain

from sagent.model.types import Response
from sagent.session.types import (
    AgentMessage,
    AssistantMessage,
    RequestMeasurement,
    SessionContext,
    SessionRecord,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UserMessage,
    stored_plain,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.model.cost import TokenCount
    from sagent.model.types import Request
    from sagent.providers.types import Model


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestSnapshot:
    """Capture framing and record hashes before streaming; never retain record text."""

    tokenizer: str
    model: str
    frame: str
    hashes: tuple[str, ...]


class TokenAccounting:
    """Reuse measured unchanged requests and appended groups before estimating text.

    It holds the records it last hashed, and no older ones, to reuse their hashes
    in the next sizing; and the hashes of measured-group candidates, until the next
    measured call, to reuse them across the many sizings of one compaction.
    """

    def __init__(self) -> None:
        self._prompts: dict[tuple[str, str, str], RequestMeasurement] = {}
        self._blocks: dict[tuple[int, str], dict[str, int]] = {}
        self._last: RequestMeasurement | None = None
        self._hashed: tuple[tuple[SessionRecord, str], ...] = ()
        self._slices: dict[tuple[str, str], str] = {}

    @property
    def has_measurements(self) -> bool:
        """Return whether any prompt measurements have been retained."""
        return bool(self._prompts)

    def capture(self, request: Request, *, model: Model) -> RequestSnapshot:
        """Capture request framing and content before streaming.

        Args:
          request: The exact request about to stream.
          model: The model about to answer, with its settings still unchanged.

        Returns:
          snapshot: Hashes independent of later request or model mutations.

        """
        return RequestSnapshot(
            tokenizer=_tokenizer(model),
            model=f"{model.provider.name}/{model.capability.model_id}",
            frame=_digest(
                (request.system, _tools(request), request.allowed, model.settings),
            ),
            hashes=self._hashes(request.records),
        )

    # Records are frozen values, so an equal record hashes the same. A request is
    # rebuilt for every call, a later message told as a new ``UserMessage`` each
    # time, so reuse by identity alone hashes the whole window again per call; and
    # a window and the requests of it put the same records at other places, so
    # reuse by place alone does too. ``held`` keeps each record alive, so no id in
    # it names another object.
    def _hashes(self, records: Sequence[SessionRecord]) -> tuple[str, ...]:
        """Return each record's hash, reusing the last sizing's by identity or place."""
        held = self._hashed
        by_id = {id(r): digest for r, digest in held}
        hashes = tuple(
            by_id.get(id(r))
            or (held[i][1] if i < len(held) and held[i][0] == r else None)
            or _digest(stored_plain(r))
            for i, r in enumerate(records)
        )
        self._hashed = tuple(zip(records, hashes, strict=True))
        return hashes

    def observe(
        self,
        snapshot: RequestSnapshot,
        *,
        tokens: TokenCount,
    ) -> RequestMeasurement:
        """Retain one prompt measurement and a compatible appended-group delta.

        Args:
          snapshot: The request captured before streaming.
          tokens: Provider usage; the prompt includes all cache pools, not output.

        Returns:
          measurement: Compact journal data. A missing usage report stores no size.
            A delta measures the entire appended group, never its individual records.

        """
        self._slices.clear()
        last = self._last
        start = 0
        delta = None
        if (
            last is not None
            and (last.model, last.tokenizer, last.frame)
            == (snapshot.model, snapshot.tokenizer, snapshot.frame)
            and last.records < len(snapshot.hashes)
            and last.prefix == _prefix(snapshot.hashes[: last.records])
            and tokens.prompt >= last.prompt
        ):
            start, delta = last.records, tokens.prompt - last.prompt
        measurement = RequestMeasurement(
            tokenizer=snapshot.tokenizer,
            model=snapshot.model,
            frame=snapshot.frame,
            prefix=_prefix(snapshot.hashes),
            records=len(snapshot.hashes),
            prompt=tokens.prompt,
            block=_prefix(snapshot.hashes[start:]) if delta is not None else "",
            block_records=len(snapshot.hashes) - start if delta is not None else 0,
            block_tokens=delta,
        )
        self.restore(measurement)
        return measurement

    def restore(self, measurement: RequestMeasurement) -> None:
        """Restore one journaled measurement in chronological order.

        Args:
          measurement: A record produced by ``observe``. Zero prompt counts mean
            usage was unavailable, and must not overwrite measured sizes.

        """
        self._last = measurement if measurement.prompt > 0 else None
        if measurement.prompt <= 0:
            return
        key = (measurement.frame, measurement.prefix, measurement.tokenizer)
        self._prompts[key] = measurement
        if measurement.block_tokens is not None:
            block = (measurement.block_records, measurement.block)
            self._blocks.setdefault(block, {})[measurement.tokenizer] = (
                measurement.block_tokens
            )

    def estimate_request(self, request: Request, *, model: Model) -> int:
        """Estimate a whole request from compatible measurements, then characters.

        Args:
          request: The complete candidate, including its system prompt and tools.
          model: The destination model; matching tokenizer measurements win.

        Returns:
          tokens: A matching measured prefix plus the remaining blocks. No count
            from a modified prefix is reused; changed framing uses block estimates.

        """
        snapshot = self.capture(request, model=model)
        chain = _chain(snapshot.hashes)
        prefixes = set(chain)
        matching = [
            m
            for m in self._prompts.values()
            if m.frame == snapshot.frame and m.prefix in prefixes
        ]
        if matching:
            best = max(
                matching,
                key=lambda m: (m.tokenizer == snapshot.tokenizer, m.records),
            )
            return best.prompt + self._block_tokens(
                request.records[best.records :],
                snapshot.hashes[best.records :],
                chain[best.records :],
                model=model,
            )
        chars = sum(map(len, request.system)) + sum(
            len(_encoded(tool)) for tool in _tools(request)
        )
        return int(
            chars / model.capability.approx_chars_per_token,
        ) + self._block_tokens(request.records, snapshot.hashes, chain, model=model)

    def estimate_block(
        self,
        entries: Sequence[Response | SessionRecord],
        *,
        model: Model,
    ) -> int:
        """Estimate contiguous content using measured groups without splitting them.

        Args:
          entries: An unchanged contiguous block; ``Response`` wrappers are flattened.
          model: The destination model. Other-tokenizer measurements remain estimates.

        Returns:
          tokens: Whole measured groups plus character estimates only for unmeasured
            records. Subsets of a measured group are not exact individual counts.

        """
        records = tuple(
            r
            for entry in entries
            for r in (entry.records if isinstance(entry, Response) else (entry,))
        )
        hashes = self._hashes(records)
        return self._block_tokens(records, hashes, _chain(hashes), model=model)

    def _block_tokens(
        self,
        records: Sequence[SessionRecord],
        hashes: tuple[str, ...],
        chain: tuple[str, ...],
        *,
        model: Model,
    ) -> int:
        """Return ``estimate_block`` of ``records``, whose hashes and chain are given."""
        tokenizer = _tokenizer(model)
        total = at = 0
        sizes = sorted({size for size, _ in self._blocks}, reverse=True)
        while at < len(records):
            found = self._block(
                hashes,
                chain,
                start=at,
                tokenizer=tokenizer,
                sizes=sizes,
            )
            if found is None:
                total += _estimated(records[at], model=model)
                at += 1
            else:
                size, tokens = found
                total += tokens
                at += size
        return total

    def _block(
        self,
        hashes: tuple[str, ...],
        chain: tuple[str, ...],
        *,
        start: int,
        tokenizer: str,
        sizes: list[int],
    ) -> tuple[int, int] | None:
        choices = [
            (
                size,
                self._blocks.get((size, self._slice(hashes, chain, start, size)), {}),
            )
            for size in sizes
            if start + size <= len(hashes)
        ]
        for size, counts in choices:
            if tokenizer in counts:
                return size, counts[tokenizer]
        for size, counts in choices:
            if counts:
                return size, next(reversed(counts.values()))
        return None

    # The chain at both ends names the slice and everything before it, so a key
    # never names two slices; a compaction's sizings share their leading records,
    # and with them the slices of every group probed among them.
    def _slice(
        self,
        hashes: tuple[str, ...],
        chain: tuple[str, ...],
        start: int,
        size: int,
    ) -> str:
        """Return the hash of ``hashes[start : start + size]``, kept for reuse."""
        key = (chain[start], chain[start + size])
        found = self._slices.get(key)
        if found is None:
            found = self._slices[key] = _prefix(hashes[start : start + size])
        return found


def _tokenizer(model: Model) -> str:
    return (
        model.capability.tokenizer
        or f"{model.provider.name}/{model.capability.model_id}"
    )


def _encoded(value: object) -> str:
    return json.dumps(to_plain(value), sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_encoded(value).encode()).hexdigest()


def _prefix(hashes: tuple[str, ...]) -> str:
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def _chain(hashes: tuple[str, ...]) -> tuple[str, ...]:
    """Return ``_prefix`` of each leading run of ``hashes``, the empty one first."""
    hasher = hashlib.sha256()
    chain = [hasher.hexdigest()]
    for digest in hashes:
        hasher.update(digest.encode())
        chain.append(hasher.hexdigest())
    return tuple(chain)


def _tools(request: Request) -> tuple[tuple[object, ...], ...]:
    return tuple((t.name, t.description, t.schema) for t in request.tools)


def _estimated(record: SessionRecord, *, model: Model) -> int:
    attachments = ()
    if isinstance(record, ToolMessage):
        text, attachments = record.text(), record.attachments
    elif isinstance(record, SessionContext):
        text = record.text()
    elif isinstance(record, ToolCall):
        text = record.name + _encoded(mutable(record.arguments))
    elif isinstance(record, Thinking):
        text = record.content or record.summary or record.encrypted or ""
    elif isinstance(
        record,
        (UserMessage, AssistantMessage, SystemMessage, AgentMessage),
    ):
        text, attachments = record.content or "", record.attachments
    else:
        text = ""
    return int(len(text) / model.capability.approx_chars_per_token) + sum(
        model.attachment_tokens(a) for a in attachments
    )
