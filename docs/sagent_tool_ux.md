# Per-Tool UX

`⎿` extends what is otherwise *input* -- the command, the diff receipt.
Output is always indented, never glyphed.

Errors and hints are always shown, regardless of the output setting.

Default policy when output is on: first 2 lines, `⋯ N lines ⋯`, last 2
lines. Both counts are per-tool defaults and CLI-configurable.

```
--tool Bash.output_head_rows=2 --tool Bash.output_tail_rows=2
--tool Read.output=on
--tool Bash.output=off
/tool Bash.output_tail_rows=20
```

| tool | output default |
|---|---|
| Read | off |
| Grep | off |
| Glob | off |
| List | off |
| Write | off |
| Edit | on |
| Bash | on |
| WebSearch | off |
| WebFetch | off |

## Read

`output=off` by default. With `--tool Read.output=on`:

```
Read check_dataclass.py:1-50
   #!/bin/sh
   # ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
   ⋯ 44 lines ⋯
   REQUIRED: Final = {"kw_only": True, "slots": True}
   if not isinstance(node, ast.Call):

Read missing.py
   ✗ File not found: missing.py
```

## Grep

`output=off` by default. With `--tool Grep.output=on`:

```
Grep 'coerce_kwargs' in sagent
   tools/tool_spec.py:69
   bin/cli.py:230
   ⋯ 4 lines ⋯
   repl/input_pane.py:476
   tools/tool_spec_test.py:49
```

## Glob

`output=off` by default. With `--tool Glob.output=on`:

```
Glob '**/*.py' in sagent
   agent/agent.py
   agent/background.py
   ⋯ 138 lines ⋯
   types/runtime.py
   types/tools.py
```

## List

`output=off` by default. With `--tool List.output=on`:

```
List sagent
   agent/
   bin/
   ⋯ 14 lines ⋯
   tools/
   types/
```

## Write

`output=off` by default. With `--tool Write.output=on`:

```
Write bash_test.py
   412 lines
```

## Edit

`output=on` by default.

```
Edit bash.py
⎿  Added 5 lines, removed 2 lines
  158 - def __init__(self, *, peers=(), description="on"):
  158 + @dataclass(frozen=True, slots=True, kw_only=True)
  159 + class Bash:
```

## Bash

`output=on` by default.

```
Bash Fetch MW pronunciation spans
⎿  python3 -c 'import urllib.request; print(fetch(url))'
   ✗ HTTP Error 403: Forbidden

Bash List web tool files
⎿  ls sagent/tools/web*.py
   hint: ls glob via Bash is a bad UX. Use the Glob tool.
   web_fetch.py
   web_search.py
   ⋯ 22 lines ⋯
   errors_test.py
   README.md

Bash Print mocked UX
⎿  uv run python /opt/scratch/scripts/probe_mock.py
   (no output)

Bash Run the full sagent suite
⎿  uv run pytest sagent -q
   ⠹ 12s

Bash Run the full sagent suite
⎿  uv run pytest sagent -q
   ........................................ [ 24%]
   ........................................ [ 48%]
   ⋯ 49 lines ⋯
   0.20s call  agent/runtime_test.py::test_user_queued
   4551 passed, 53 skipped in 28.96s

Bash Check worktree state
⎿  git status --short
   M sagent/tools/bash.py

Bash Run the gates
⎿  uv run pytest -q
   ........................................ [ 24%]
   F....................................... [ 48%]
   ⋯ 61 lines ⋯
   E   TypeError: frozen instance
   ✗ 1 failed, 4550 passed in 29.14s
```

## WebSearch

`output=off` by default. With `--tool WebSearch.output=on`:

```
WebSearch 'pep 695 type alias get_origin'
   peps.python.org/pep-0695 — Type Parameter Syntax
   docs.python.org/3/library/typing.html — typing.get_origin
   ⋯ 8 lines ⋯
   discuss.python.org/t/pep-695-typealiastype — Discussion
   bugs.python.org/issue45607 — get_origin and aliases
```

## WebFetch

`output=off` by default. With `--tool WebFetch.output=on`:

```
WebFetch https://peps.python.org/pep-0695/
   # PEP 695 – Type Parameter Syntax

   ⋯ 214 lines ⋯

   Copyright: This document is placed in the public domain.

WebFetch https://example.com/gone
   ✗ HTTP Error 404: Not Found
```
