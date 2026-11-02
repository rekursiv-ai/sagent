"""What tests of MCP servers run: a stdio server with one tool, ``shout``.

The protocol is written by hand, stdlib only: importing the ``mcp`` server
package costs the child most of a second, and a client speaks to this the same
way. It says ``server started`` on stderr, so a test can find its log.

Typical usage example:

  script = tmp_path / "server.py"
  script.write_text(SHOUT_SERVER)
  StdioServerParameters(command=sys.executable, args=[str(script)])
"""

from __future__ import annotations

from typing import Final


__all__ = ["SHOUT_SERVER"]


SHOUT_SERVER: Final = """\
import json, sys

print("server started", file=sys.stderr, flush=True)
SHOUT = {
    "name": "shout",
    "description": "Say the text loudly.",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method, params = request["method"], request.get("params", {})
    if method == "initialize":
        result = {
            "protocolVersion": params["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "good", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [SHOUT]}
    elif method == "tools/call":
        text = params["arguments"]["text"].upper()
        result = {"content": [{"type": "text", "text": text}], "isError": False}
    elif method == "ping":
        result = {}
    else:
        error = {"code": -32601, "message": f"no method {method}"}
        reply = {"jsonrpc": "2.0", "id": request["id"], "error": error}
        print(json.dumps(reply), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""
"""A Python script serving ``shout``, which returns its ``text`` upper-cased."""
