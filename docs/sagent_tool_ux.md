# Per-Tool UX

`⎿` extends what is otherwise *input* -- the command, the diff receipt.
Output is always indented, never glyphed.

Errors and hints are always shown, regardless of the output setting.

## Compact commands and diffs

In the interactive REPL, command previews apply their row budget after
wrapping to the available terminal width. Bash keeps its configured first
3 and last 1 display rows by default. A long single-line command therefore
has the same visible budget as a multiline command. Output-body settings
still count logical lines as before.

Edit diffs show the first 8 and last 4 display rows, with added/removed
totals computed from the complete diff. Short diffs render completely.
An omission marker counts hidden display rows and points to the original:

```text
  … 148 rows omitted · /details 2
```

Use `/details 2` to open that complete command or diff. `/details` opens
the latest item with a compact preview. Ctrl+O opens the same view while
preserving the draft and cursor in the input pane.

Inside the details view:

- Click an item header, or use Enter/Space, to expand or collapse it.
- Up/Down select another item. Tab switches between the list and text.
- Ctrl+F searches the complete item, expanding it first if necessary.
- Ctrl+Y copies a selection, or the complete original item when nothing
  is selected. macOS uses `pbcopy`; other terminals receive OSC 52.
- F2 disables mouse handling for native terminal text selection. Clipboard
  and selection support depend on the terminal.
- Esc, q, or Ctrl+C closes the view. During search, Esc first exits search.
- The footer shows the original text file for inspection with another tool.

The view is read only. Opening or expanding an item does not dispatch
input, call a model, rerun a command, or change the session tape. Agent
tasks continue while the view is open. Transcript output is buffered and
printed when it closes; newly arriving items appear in the list without
changing the selected item. Mouse interaction is confined to this view.
Already printed terminal scrollback stays ordinary selectable text.

Original commands and diffs are saved in private temporary files for the
REPL invocation and removed on normal exit, cancellation, or failure.
IDs and raw-file paths are local to that invocation. Session replay
rebuilds details from the retained tape; these files are not a permanent
archive of messages removed by clear or compaction. Terminal control
characters are displayed visibly in the inspector, while the raw files
and complete-item copying retain the exact original text.

If storage fails, the renderer reports the failure and prints the full
payload. Non-interactive printers keep their existing rendering. When
inspection runs without a terminal, it prints the selected complete item.

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
