"""Shared test doubles for fetch transport tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from curl_cffi import requests as cc_requests


def lower_headers(kw: dict[str, Any]) -> dict[str, str]:
    """Lower-cased request headers from a curl ``request`` mock's kwargs."""
    headers = cast("dict[str, str] | None", kw.get("headers")) or {}
    return {k.lower(): v for k, v in headers.items()}


def const_curl_session(stub: Any) -> Callable[..., Any]:
    """A ``curl_session`` replacement that always returns ``stub`` (typed)."""

    def factory(*_args: object) -> Any:
        return stub

    return factory


class StubCookie:
    """A minimal jar entry: just a name/value, enough for the pooled-path tests."""

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value


class StubCookies:
    """Minimal curl-cookies stand-in: a recording jar plus a ``set`` that stores."""

    def __init__(self) -> None:
        self.jar: list[Any] = []

    def set(
        self,
        name: str,
        value: str,
        *,
        domain: str = "",
        path: str = "/",
        secure: bool = False,
    ) -> None:
        del domain, path, secure
        self.jar = [c for c in self.jar if getattr(c, "name", None) != name]
        self.jar.append(StubCookie(name, value))


class StubSession:
    """A pooled-Session stand-in whose request delegates to the module-level
    ``curl_cffi.requests.request`` -- so one ``patch("curl_cffi.requests.request")``
    intercepts both the identity (session) and keyless paths.
    """

    def __init__(self) -> None:
        self.cookies = StubCookies()

    def request(self, *args: Any, **kwargs: Any) -> Any:
        return cc_requests.request(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType] -- curl_cffi's **RequestParams TypedDict is unstubbed

    def close(self) -> None:
        pass
