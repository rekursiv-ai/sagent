#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Executable entry point. Dispatches to typecheck's ``main``.
'''
# fmt: on

from __future__ import annotations

from tools.typecheck.typecheck import main


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
