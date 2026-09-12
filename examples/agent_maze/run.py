#!/bin/sh
# ruff: noqa: EXE003, D300, D205, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Entry point for the agent-maze coordination demo (decentralized vs centralized).

Default is **replay** -- it serves the webpage, which replays a captured run from
``web/data.js`` (two arms side by side, no API key needed):

    uv run python -m examples.agent_maze.run              # replay (serves web/)
    uv run python -m examples.agent_maze.run --live       # re-capture all 4 conditions
    uv run python -m examples.agent_maze.run --live --locks 4 --k 2

Same foggy maze, same paired-lock coordination task -- only the comms topology differs
(mesh = any-to-any + broadcast + anyone spawns; tree = hub-and-spoke relay, only the
coordinator spawns). The agents are autonomous sagent ``Agent``s acting through tools;
the World is a reactive feedback service on a logical clock. Engine + mechanic live in
``engine.py`` / ``world.py``; one arm in ``arena.py``; capture + metrics in ``capture.py``.
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Final

import argparse
import asyncio
import contextlib
import functools
import http.server
import socket
import webbrowser

from examples.agent_maze.capture import capture


_CWD: Final = Path(__file__).resolve().parent
PORT = 8001  # config-globals: ignore -- serve port, meant to be overridden


def serve(port: int = PORT, host: str = "127.0.0.1") -> None:
    """Serve web/ and open the replay.

    Args:
      port: Local TCP port for the web server.
      host: Bind address for the web server.

    """
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(_CWD / "web"),
    )
    httpd = http.server.HTTPServer((host, port), handler)
    url = f"http://localhost:{port}/index.html"
    host_name = socket.gethostname()
    print(f"\n  ▶  Report served on this machine at:  {url}\n")
    print("  Viewing over SSH? Forward the port from your laptop:")
    print(f"      ssh -L {port}:localhost:{port}  {host_name}   # or  <user>@<host>")
    print(f"  then open  {url}  in your LOCAL browser.\n")
    print("  (replaying the captured run -- press Ctrl-C to stop)")
    with contextlib.suppress(Exception):
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def main() -> int:
    """Serve the agent-maze web interface and return the exit code.

    Returns:
      status: Process exit code.

    """
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
    )
    _add_arguments(ap)
    args = ap.parse_args()

    if args.live:
        asyncio.run(capture(num_locks=args.locks, k=args.k))
    elif not (_CWD / "web" / "data.js").exists():
        print("no web/data.js yet -- run with --live to capture one.")
        return 0

    if args.no_serve:
        return 0
    serve(args.port, args.host)
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register agent-maze command-line flags."""
    parser.add_argument(
        "--live",
        action="store_true",
        help="re-capture all 4 conditions (mesh/tree x told/discover), then serve",
    )
    parser.add_argument("--locks", type=int, default=3, help="locks per maze (--live)")
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="runs per cell to cherry-pick (--live)",
    )
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-serve", action="store_true")


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
