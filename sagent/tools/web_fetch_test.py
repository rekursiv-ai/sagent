"""Tests for WebFetch: arguments, shell takeover, caching, and errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from wesearch.fetch import FetchSession, PolicyParams
from wesearch.types.errors import CloudflareChallengeError, FetchError
from wesearch.types.params import NO_BODY, NoBody
from wesearch.web import FetchResult

import pytest

from sagent.session.types import UncategorizedToolResult, WebFetchResult
from sagent.tools import web_fetch
from sagent.tools.command_analyzer import invocations
from sagent.tools.types import ToolContext
from sagent.tools.web_fetch import WebFetch


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree


@dataclass(slots=True, kw_only=True)
class Fetches:
    """Stands in for the network: records each fetch, answers or raises."""

    text: str = "page text\n"
    error: Exception | None = None
    seen: list[dict[str, object]] = field(default_factory=list)

    def __call__(
        self,
        url: str,
        *,
        method: str,
        json_body: PlainTree | NoBody,
        form_body: dict[str, str] | None,
        policy: PolicyParams,
        session: FetchSession,
    ) -> FetchResult:
        self.seen.append(
            {
                "url": url,
                "method": method,
                "json_body": json_body,
                "form_body": form_body,
                "transport": policy.transport,
                "extractor": policy.extractor,
                "session": session,
            },
        )
        if self.error is not None:
            raise self.error
        return FetchResult(text=self.text, url=url, kind="html", truncated=False)


@pytest.fixture
def fetches(monkeypatch: pytest.MonkeyPatch) -> Fetches:
    fake = Fetches()
    monkeypatch.setattr(web_fetch, "fetch_web", fake)
    return fake


_CONTEXT = ToolContext(cwd=Path("/"))


def _error(text: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=text, is_error=True)


@pytest.mark.parametrize(
    "flag",
    [
        *("-s", "-S", "-L", "-f", "-sS", "-sL", "-fsSL", "-sSL", "-fsS", "-fL"),
        *("--silent", "--show-error", "--location", "--fail"),
    ],
)
def test_curl_may_be_quiet_follow_and_fail(flag: str) -> None:
    assert _shell(f"curl {flag} https://example.com/a") == {
        "url": "https://example.com/a",
    }


def _shell(command: str) -> Mapping[str, PlainTree] | str | None:
    found = invocations(command)
    assert found is not None
    (invocation,) = found
    return WebFetch(session=FetchSession()).from_shell(invocation)


# --- Description ------------------------------------------------------------


def test_webfetch_describes_itself() -> None:
    tool = WebFetch(session=FetchSession())
    assert tool.name == "WebFetch"
    assert tool.description.strip()
    assert tool.schema["required"] == ["url"]
    assert (
        tool.summary({"url": "https://a.example/x"}) == "WebFetch https://a.example/x"
    )
    assert tool.summary({}) == "WebFetch "
    assert tool.serialize_key(_CONTEXT, {"url": "https://a.example"}) is None


# --- Shell takeover -----------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "curl https://example.com/a",
        "curl -fsSL https://example.com/a",
        "curl --silent --location https://example.com/a",
        "wget -qO- https://example.com/a",
        "wget -q -O - https://example.com/a",
        "wget -O- https://example.com/a",
        "curl -s http://8.8.8.8/a",
        "wget -qO - https://example.com/a",
        "wget -O - https://example.com/a",
        (
            "curl -s -S -L -f -sS -sL -fsSL -sSL -fsS -fL --silent --show-error"
            " --location --fail https://example.com/a"
        ),
    ],
)
def test_a_plain_download_becomes_a_fetch(command: str) -> None:
    assert _shell(command) == {"url": command.rsplit(maxsplit=1)[-1]}


@pytest.mark.parametrize(
    "command",
    [
        "wget https://example.com/a",
        'curl "https://example.com/$P"',
        "curl -o page.html https://example.com/a",
        "curl -H 'X: y' https://example.com/a",
        "curl https://example.com/a https://example.com/b",
        "curl -s",
        "curl http://localhost:8000/a",
        "curl http://127.0.0.1/a",
        "curl http://10.0.0.5/a",
        "curl http://printer.local/a",
        "curl http://db.internal/a",
        "ls https://example.com/a",
        "curl https://example.com/a > page.html",
    ],
)
def test_anything_else_stays_a_shell_command(command: str) -> None:
    assert _shell(command) is None


# --- Fetching ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_get_returns_the_pages_text(fetches: Fetches) -> None:
    tool = WebFetch(session=FetchSession())
    result = await tool.call(_CONTEXT, {"url": "https://example.com"})
    assert isinstance(result, WebFetchResult)
    assert (result.url, result.content, result.size) == (
        "https://example.com",
        "page text\n",
        10,
    )
    assert result.call_id == ""
    assert result.duration_sec is not None
    assert 0 <= result.duration_sec < 1
    assert fetches.seen == [
        {
            "url": "https://example.com",
            "method": "GET",
            "json_body": NO_BODY,
            "form_body": None,
            "transport": "auto",
            "extractor": "html2text",
            "session": tool.session,
        },
    ]


@pytest.mark.asyncio
async def test_a_lowercase_method_is_accepted(fetches: Fetches) -> None:
    result = await WebFetch(session=FetchSession()).call(
        _CONTEXT,
        {"url": "https://e.com", "method": "post"},
    )
    assert isinstance(result, WebFetchResult)
    assert fetches.seen[0]["method"] == "POST"


@pytest.mark.asyncio
async def test_an_unknown_method_is_an_argument_error(fetches: Fetches) -> None:
    with pytest.raises(ValueError, match="'method' must be one of"):
        _ = await WebFetch(session=FetchSession()).call(
            _CONTEXT,
            {"url": "https://e.com", "method": "PUT"},
        )
    assert fetches.seen == []


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"json": {"a": 1}}, "'json' and 'form' require method='POST'."),
        ({"form": {"a": "1"}}, "'json' and 'form' require method='POST'."),
        (
            {"method": "POST", "json": {"a": 1}, "form": {"a": "1"}},
            "'json' and 'form' are mutually exclusive.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_body_must_fit_the_method(
    fetches: Fetches,
    arguments: dict[str, PlainTree],
    error: str,
) -> None:
    result = await WebFetch(session=FetchSession()).call(
        _CONTEXT,
        {"url": "https://e.com", **arguments},
    )
    assert result == _error(error)
    assert fetches.seen == []


@pytest.mark.asyncio
async def test_a_post_sends_its_body(fetches: Fetches) -> None:
    tool = WebFetch(session=FetchSession())
    _ = await tool.call(
        _CONTEXT,
        {"url": "https://e.com", "method": "POST", "json": None},
    )
    _ = await tool.call(
        _CONTEXT,
        {"url": "https://e.com", "method": "POST", "json": [1]},
    )
    _ = await tool.call(
        _CONTEXT,
        {"url": "https://e.com", "method": "POST", "form": {"q": "x"}},
    )
    assert [(s["json_body"], s["form_body"]) for s in fetches.seen] == [
        (NO_BODY, None),
        ([1], None),
        (NO_BODY, {"q": "x"}),
    ]


@pytest.mark.asyncio
async def test_a_call_may_choose_transport_and_extractor(fetches: Fetches) -> None:
    tool = WebFetch(
        session=FetchSession(),
        policy=PolicyParams(transport="curl", extractor="raw"),
    )
    _ = await tool.call(_CONTEXT, {"url": "https://e.com"})
    _ = await tool.call(
        _CONTEXT,
        {"url": "https://e.com", "transport": "stdlib", "extractor": "trafilatura"},
    )
    assert [(s["transport"], s["extractor"]) for s in fetches.seen] == [
        ("curl", "raw"),
        ("stdlib", "trafilatura"),
    ]


@pytest.mark.asyncio
async def test_a_get_is_cached_by_url_transport_and_extractor(fetches: Fetches) -> None:
    tool = WebFetch(session=FetchSession())
    for arguments in (
        {"url": "https://e.com"},
        {"url": "https://e.com"},
        {"url": "https://e.com", "transport": "curl"},
        {"url": "https://e.com", "extractor": "raw"},
        {"url": "https://e.com/other"},
    ):
        result = await tool.call(_CONTEXT, arguments)
        assert isinstance(result, WebFetchResult)
        assert result.content == "page text\n"
    assert len(fetches.seen) == 4


@pytest.mark.asyncio
async def test_a_post_is_never_replayed(fetches: Fetches) -> None:
    tool = WebFetch(session=FetchSession())
    post: dict[str, PlainTree] = {"url": "https://e.com", "method": "POST"}
    _ = await tool.call(_CONTEXT, {"url": "https://e.com"})
    _ = await tool.call(_CONTEXT, post)
    _ = await tool.call(_CONTEXT, post)
    _ = await tool.call(_CONTEXT, {"url": "https://e.com"})
    assert [s["method"] for s in fetches.seen] == ["GET", "POST", "POST"]


@pytest.mark.asyncio
async def test_a_long_page_is_cut_with_a_note(fetches: Fetches) -> None:
    fetches.text = "".join(f"line {i:03}\n" for i in range(100))
    context = ToolContext(cwd=Path("/"), max_result_chars=200)
    result = await WebFetch(session=FetchSession()).call(
        context,
        {"url": "https://e.com"},
    )
    assert isinstance(result, WebFetchResult)
    assert result.content is not None
    body, note = result.content.rsplit("\n... ", 1)
    kept = body.splitlines()
    assert kept == [f"line {i:03}" for i in range(len(kept))]
    assert note == f"({100 - len(kept):,} more lines omitted)"
    assert len(result.content) <= 200
    assert result.size == len(fetches.text)


@pytest.mark.asyncio
async def test_a_bot_block_says_how_to_clear_it(fetches: Fetches) -> None:
    fetches.error = CloudflareChallengeError(url="https://e.com")
    result = await WebFetch(session=FetchSession()).call(
        _CONTEXT,
        {"url": "https://e.com"},
    )
    assert result == _error(CloudflareChallengeError.explain("https://e.com"))


@pytest.mark.parametrize(
    "error",
    [
        FetchError("https://e.com", 404, {}, b""),
        ValueError("refusing a private host"),
        OSError("network unreachable"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_fetch_says_why(fetches: Fetches, error: Exception) -> None:
    fetches.error = error
    tool = WebFetch(session=FetchSession())
    result = await tool.call(_CONTEXT, {"url": "https://e.com"})
    assert result == _error(f"Fetch failed: {error}")
    fetches.error = None
    second = await tool.call(_CONTEXT, {"url": "https://e.com"})
    assert isinstance(second, WebFetchResult)
    assert len(fetches.seen) == 2


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
