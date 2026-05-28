Run a shell command; return stdout/stderr.

Prefer dedicated tools; use Bash only when none fits or the user asks.

| Task | Use | Avoid |
|---|---|---|
| File discovery | `Glob` | `find`, `ls` |
| Text search | `Grep` | `grep`, `rg` |
| File viewing | `Read` | `cat`, `head`, `tail` |
| File patching | `Edit` | `sed`, `awk` |
| File creation | `Write` | `echo >`, `cat <<EOF` |
| Display text | respond directly | `echo`, `printf` |

- Absolute paths keep the session cwd consistent; `cd` only when asked. Quote paths with spaces.
- `timeout` in ms, cap ${GET_MAX_TIMEOUT_MS()} ms. Default kill ${GET_DEFAULT_TIMEOUT_MS()} ms.
- Independent commands -> parallel Bash calls. Dependent -> chain with `&&`.
- Confirm parents with `Glob`/`List` before creating.
- Poll status instead of sleeping.
