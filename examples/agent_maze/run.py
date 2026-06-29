"""Entry point for the agent-mesh maze coordination demo (mesh vs tree topology).

Default is **replay** — it serves the webpage, which replays a captured run from
``web/data.js`` (two arms side by side, no API key needed):

    uv run python -m examples.agent_maze.run             # replay (serves web/)
    uv run python -m examples.agent_maze.run --live      # re-capture both arms, then serve
    uv run python -m examples.agent_maze.run --live --discover   # hide the topology

Same maze, same paired-lock coordination task, same agents — only the comms topology
differs (mesh = any-to-any, tree = hub-and-spoke). The capture engine + mechanic live in
``lock_lockstep.py`` and ``world.py``.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import asyncio
import contextlib
import functools
import http.server
import os
import socket
import webbrowser

from examples.agent_maze.lock_lockstep import _key, capture


HERE = Path(__file__).parent
PORT = 8001


def serve(port: int = PORT, host: str = "127.0.0.1") -> None:
    """Serve web/ and open the replay (prints an ssh -L line for remote viewing)."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(HERE / "web")
    )
    httpd = http.server.HTTPServer((host, port), handler)
    url = f"http://localhost:{port}/index.html"
    host_name = socket.gethostname()
    print(f"\n  ▶  Report served on this machine at:  {url}\n")  # noqa: T201
    print("  Viewing over SSH? Forward the port from your laptop:")  # noqa: T201
    print(f"      ssh -L {port}:localhost:{port}  {host_name}   # or  <user>@<host>")  # noqa: T201
    print(f"  then open  {url}  in your LOCAL browser.\n")  # noqa: T201
    print("  (replaying the captured run — press Ctrl-C to stop)")  # noqa: T201
    with contextlib.suppress(Exception):
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description="agent-mesh maze coordination demo")
    ap.add_argument(
        "--live",
        action="store_true",
        help="re-capture all 4 conditions (mesh/tree x told/discover), then serve",
    )
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    if args.live:
        os.environ["ANTHROPIC_API_KEY"] = _key()
        asyncio.run(capture())
    elif not (HERE / "web" / "data.js").exists():
        print("no web/data.js yet — run with --live to capture one.")  # noqa: T201
        return

    if args.no_serve:
        return
    serve(args.port, args.host)


if __name__ == "__main__":
    main()
