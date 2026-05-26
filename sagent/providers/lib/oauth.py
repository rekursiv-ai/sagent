"""OAuth helpers shared across subscription providers.

Includes:
  - :class:`AuthCodeListener`: localhost callback server for the OAuth
    authorization-code redirect flow.
  - :func:`pkce_pair`: PKCE verifier/challenge generator.
  - :func:`resolve_account` and :func:`credentials_path`: per-account
    credentials-file routing shared by every subscription provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import base64
import hashlib
import http.server as http_server
import re
import secrets
import threading
import urllib.parse


_DEFAULT_ACCOUNT = "default"
_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def resolve_account(account: str | None) -> str:
    """Normalise ``account`` and reject names that could escape the dir.

    Args:
      account: User-supplied account name. ``None`` maps to the legacy
        unnamed account ``"default"``.

    Returns:
      name: Canonical account name matching ``[A-Za-z0-9][A-Za-z0-9_-]*``.

    Raises:
      ValueError: If ``account`` contains characters that could escape
        the credentials directory (path-traversal guard).

    """
    name = account or _DEFAULT_ACCOUNT
    if not _ACCOUNT_RE.match(name):
        raise ValueError(
            f"Invalid account name {name!r}: use alphanumerics, ``_``, or ``-``.",
        )
    return name


def credentials_path(default_path: Path, account: str | None) -> Path:
    """Return per-account credentials path, suffixed with ``-<account>``.

    ``None`` or ``"default"`` returns ``default_path`` unchanged
    (legacy, backward-compatible). Named accounts insert ``-<name>``
    before the suffix: e.g. ``oauth_creds.json`` →
    ``oauth_creds-work.json``.

    Args:
      default_path: The legacy single-account file path.
      account: Account name. ``None``/``"default"`` selects the legacy
        path; any other value selects the per-account variant.

    Returns:
      path: Resolved per-account credentials file path.

    Raises:
      ValueError: Via :func:`resolve_account` for malformed names.

    """
    name = resolve_account(account)
    if name == _DEFAULT_ACCOUNT:
        return default_path
    return default_path.with_stem(f"{default_path.stem}-{name}")


def parse_manual_auth_code(value: str, expected_state: str) -> str:
    """Extract an OAuth code from pasted manual-login input.

    Args:
      value: Raw terminal paste. Accepts a code, ``code#state``, or a full
        redirect URL containing ``code`` and ``state`` query parameters.
      expected_state: State originally sent in the authorization URL.

    Returns:
      code: Authorization code to exchange.

    Raises:
      ValueError: If no code is present or a pasted state does not match.

    """
    text = value.strip()
    parsed = urllib.parse.urlparse(text)
    code = ""
    state = ""
    if parsed.scheme and parsed.netloc:
        params = urllib.parse.parse_qs(parsed.query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]
    else:
        code, sep, state = text.partition("#")
        if not sep:
            state = ""
    if not code:
        raise ValueError("Authorization code not found")
    if not state:
        raise ValueError("State missing in pasted authorization code")
    if state != expected_state:
        raise ValueError("State mismatch in pasted authorization code")
    return code


class AuthCodeServer(http_server.HTTPServer):
    """HTTPServer carrying a reference to the owning listener.

    ``BaseRequestHandler.server`` gives :class:`AuthCodeHandler`
    access to the listener's mutable state without closing over it.
    The ``listener`` attribute is set by ``AuthCodeListener.start``
    immediately after construction; the handler runs only on requests
    arriving after that point, so the attribute is always set when
    read.

    Args:
      server_address: ``(host, port)`` bind address.
      handler: HTTP request handler class.
      listener: Owning :class:`AuthCodeListener`.

    """

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[http_server.BaseHTTPRequestHandler],
        listener: AuthCodeListener,
    ) -> None:
        super().__init__(server_address, handler)
        self.listener = listener


class AuthCodeHandler(http_server.BaseHTTPRequestHandler):
    """Single-shot callback handler for the OAuth redirect URI."""

    @override
    def log_message(self, format: str, *args: object) -> None:
        """Suppress default stderr logging."""
        del format, args

    def do_GET(self) -> None:
        """Handle the OAuth redirect GET request."""
        parsed = urllib.parse.urlparse(self.path)
        expected_path = cast(AuthCodeServer, self.server).listener.callback_path
        if parsed.path != expected_path:
            self.send_response(404)
            self.end_headers()
            return
        listener = cast(AuthCodeServer, self.server).listener
        params = urllib.parse.parse_qs(parsed.query)
        state = (params.get("state") or [""])[0]
        code = (params.get("code") or [""])[0]
        err = (params.get("error") or [""])[0]
        if err:
            listener._error = err  # noqa: SLF001 -- handler callback pokes listener internals
        elif state != listener._expected_state:  # noqa: SLF001 -- handler callback pokes listener internals
            listener._error = "state mismatch"  # noqa: SLF001 -- handler callback pokes listener internals
        else:
            listener._code = code  # noqa: SLF001 -- handler callback pokes listener internals
        ok = not listener._error  # noqa: SLF001 -- handler callback pokes listener internals
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        body = (
            b"<html><body><h2>Authentication complete.</h2>"
            b"<p>You can close this tab.</p></body></html>"
            if ok
            else (
                b"<html><body><h2>Authentication failed.</h2>"
                b"<p>Return to the terminal.</p></body></html>"
            )
        )
        self.wfile.write(body)
        listener._done.set()  # noqa: SLF001 -- handler callback pokes listener internals


class AuthCodeListener:
    """Tiny localhost HTTP callback for OAuth authorization_code flow."""

    def __init__(
        self,
        expected_state: str,
        *,
        port: int = 0,
        callback_path: str = "/callback",
    ) -> None:
        self._expected_state = expected_state
        self._port = port
        self.callback_path = callback_path
        self._server: AuthCodeServer | None = None
        self._thread: threading.Thread | None = None
        self._code = ""
        self._error = ""
        self._done = threading.Event()

    @property
    def redirect_uri(self) -> str:
        """Return the full redirect URI including the bound port.

        Returns:
          uri: ``http://127.0.0.1:<port><callback_path>`` string.

        Uses the literal IPv4 loopback rather than ``localhost`` so the
        URI in the authorize request matches our server's bind address
        on every host. ``localhost`` resolution can route to ``::1`` on
        IPv6-first systems, where our IPv4-only ``AuthCodeServer``
        ("127.0.0.1", 0) silently never sees the callback.

        """
        assert self._server is not None
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}{self.callback_path}"

    def start(self) -> None:
        """Start the localhost HTTP server in a background thread."""
        if self._server is not None:
            raise RuntimeError("auth code listener already started")
        self._server = AuthCodeServer(("127.0.0.1", self._port), AuthCodeHandler, self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            args=(0.05,),
            daemon=True,
        )
        self._thread.start()

    def wait(self, timeout_sec: float) -> str:
        """Block until the authorization code arrives or timeout expires.

        Args:
          timeout_sec: Maximum seconds to wait.

        Returns:
          code: The authorization code from the callback.

        Raises:
          TimeoutError: If no callback arrives within ``timeout_sec``.
          RuntimeError: If the OAuth callback returned an error.

        """
        if not self._done.wait(timeout=timeout_sec):
            raise TimeoutError("auth code listener timed out")
        if self._error:
            raise RuntimeError(f"OAuth callback error: {self._error}")
        return self._code

    def stop(self) -> None:
        """Shut down the HTTP server and join the background thread."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code_verifier and S256 code_challenge.

    Returns:
      verifier: URL-safe random verifier string.
      challenge: Base64url-encoded SHA-256 digest of the verifier.

    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge
