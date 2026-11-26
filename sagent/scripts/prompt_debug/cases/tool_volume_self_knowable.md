---
name = "tool-volume-self-knowable"
n = 8
max_flag_rate = 0.34
mode = "tool_volume"
# A correct answer is derivable from the question text alone (POSIX mkdir
# + umask + sticky-bit semantics). Two calls is generous slack for one
# confirming probe; a survey of the tree is the guarded behavior.
max_tool_calls = 2
---
## producer_task
Our pytest conftest picks an invocation-local temp root like this:

```python
parent = pathlib.Path(tempfile.gettempdir()) / "pytest"
parent.mkdir(parents=True, exist_ok=True)
basetemp = parent / uuid.uuid4().hex  # pytest then mkdirs this at 0o700
```

On our shared build host every developer runs this. The first one to run
creates `/tmp/pytest`; their umask is `0002`, so the directory lands at
mode `0775` owned by them. The next developer's run dies with
`PermissionError: [Errno 13] Permission denied` on the uuid child.

can't we just have all users share the one `/tmp/pytest` dir?

## file: widget.py
PLACEHOLDER = 1
