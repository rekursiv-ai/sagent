"""The status line under the prompt: time, tokens, the window, spend, and the wait.

``[53s 22↑ 2144↓ 72K↟ 383K↡ (84%↡ 36%↔) $0.48; waiting on model.]``, with a
spinner while a model call or a compaction runs. The tokens are the session's
sums: request, response, cache written, cache read. Then the share of prompt
tokens the cache served, and how full the window is now.

Typical usage example:

  status = Status(agent.runtime)
  agent.runtime.observers.append(status)
  toolbar = status.frame                   # Polled by the prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import time

from sagent.model.cost import TokenCount
from sagent.runtime import types


if TYPE_CHECKING:
    from sagent.runtime.runtime import Runtime


__all__ = [
    "Status",
    "count",
    "elapsed",
]


class Status:
    """Totals of the session's model calls; an observer of the runtime.

    Args:
      runtime: Whose calls it counts, and whose state names the wait.
      wait_after_sec: How long one call runs before the line names its wait.
      spinner: The frames turned while a call or compaction runs.
      spend: What the session spent before, in USD, as a resume restores it.

    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        wait_after_sec: float = 15.0,
        spinner: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
        spend: float = 0.0,
    ) -> None:
        self.runtime = runtime
        self.wait_after_sec = wait_after_sec
        self.spinner = spinner
        self.tokens = TokenCount()
        self.spend = spend
        # What the window holds: the last reply's prompt and reply, or what a
        # compaction left.
        self.held = 0
        # Time spent in model calls that ended.
        self.seconds = 0.0
        self._call: float | None = None
        self._compact: float | None = None
        # When the calls the model waits on started; the runtime says when they end.
        self._tools: float | None = None
        self._streamed = 0

    def __call__(self, event: types.RuntimeEvent) -> None:
        """Count ``event`` toward the totals."""
        now = time.monotonic()
        if isinstance(event, types.ModelCallStarted):
            self._call, self._streamed = now, 0
        elif isinstance(event, types.ModelResponsePartial):
            self._streamed += len(event.text)
        elif isinstance(
            event,
            (
                types.ModelResponseComplete,
                types.ModelResponseCancelled,
                types.ModelResponseError,
            ),
        ):
            if isinstance(event, types.ModelResponseComplete):
                tokens = event.response.tokens
                self.tokens += tokens
                self.spend += event.response.spend.total
                self.held = tokens.prompt + tokens.response
            if self._call is not None:
                self.seconds += now - self._call
            self._call = None
        elif isinstance(event, types.CohortStarted):
            self._tools = now
        elif isinstance(event, types.CompactStarted):
            self._compact = now
        elif isinstance(event, (types.CompactComplete, types.CompactFailed)):
            self._compact = None
            if isinstance(event, types.CompactComplete) and event.token_after:
                self.held = event.token_after
        elif isinstance(event, types.ClearComplete):
            self.held = 0
        # A child's calls arrive here as they are charged; its ``ChildDoneEvent``
        # repeats their sum.
        elif isinstance(event, types.Charged):
            self.spend += event.usd

    def frame(self) -> str:
        """Return the status line.

        Returns:
          line: The totals, the wait once named, and the calls running in the
            background; ``""`` before the first call of a session that has
            spent nothing.

        """
        rt = self.runtime
        # Detached or killed, calls leave the cohort without a ``CohortComplete``.
        if not (rt.cohort or rt.running_tools):
            self._tools = None
        if (
            not self.seconds
            and not self.spend
            and self._call is None
            and self._compact is None
            and self._tools is None
        ):
            return ""
        now = time.monotonic()
        running = now - self._call if self._call is not None else 0.0
        # Only while the call runs: once it ends, its billed tokens include
        # what streamed, and adding the estimate too counted the reply twice.
        streamed = (
            int(self._streamed / self.runtime.model.capability.approx_chars_per_token)
            if self._call is not None
            else 0
        )
        t = self.tokens
        settings = rt.model.settings
        window = settings.context_size.tokens
        cached = f"{t.cache_read / t.prompt if t.prompt else 0:.0%}↡"
        filled = f" {self.held / window:.0%}↔" if window > 0 else ""
        metrics = (
            f"{elapsed(self.seconds + running)}"
            f" {count(t.request)}↑"
            f" {count(t.response + streamed)}↓"
            f" {count(t.cache_write + t.cache_write_1h)}↟"
            f" {count(t.cache_read)}↡"
            f" ({cached}{filled})"
            f" ${self.spend:.2f}"
        )
        background = sum(job.state != "running" for job in rt.jobs.values())
        reason = " ".join(
            part
            for part in (
                self._wait(running, now=now),
                f"{background} in background." if background else "",
            )
            if part
        )
        line = f"[{metrics}{f'; {reason}' if reason else ''}]"
        start = next(
            (t for t in (self._compact, self._call, self._tools) if t is not None),
            None,
        )
        if start is None:
            return line
        return f"{self.spinner[int((now - start) * 5) % len(self.spinner)]} {line}"

    def _wait(self, running: float, *, now: float) -> str:
        """Return what the agent waits on, once that is worth saying."""
        rt = self.runtime
        until = rt.service_suspended_until
        if until is not None and until > time.time():
            return f"retrying in {elapsed(until - time.time())}."
        if self._compact is not None:
            return "compacting."
        if rt.inbox.gate_armed:
            return "waiting for input."
        if self._call is None:
            if self._tools is None or now - self._tools < self.wait_after_sec:
                return ""
            return "waiting on tools."
        if running < self.wait_after_sec:
            return ""
        if rt.running_tools or rt.cohort:
            return "waiting on tools."
        return "waiting on model."


def elapsed(seconds: float) -> str:
    """Return ``seconds`` as ``12s``, ``1m 23s``, ``2h 17m 4s``, or ``3d 1h 2m 5s``.

    Args:
      seconds: A duration; a fraction floors.

    Returns:
      text: The duration, seconds always shown.

    """
    days, rem = divmod(int(seconds), 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    parts = [
        f"{days}d" if days else "",
        f"{hours}h" if days or hours else "",
        f"{minutes}m" if days or hours or minutes else "",
        f"{secs}s",
    ]
    return " ".join(p for p in parts if p)


def count(n: int) -> str:
    """Return ``n`` as ``412``, ``12K``, or ``1.8M``.

    Args:
      n: A token count.

    Returns:
      text: Whole under 10K; ``1.0M``, never ``1000K``, at 999,500.

    """
    if n < 10_000:
        return str(n)
    if n < 999_500:
        return f"{n / 1000:.0f}K"
    return f"{n / 1_000_000:.1f}M"
