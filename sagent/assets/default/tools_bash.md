Run a shell command; return stdout/stderr.

STOP before every Bash call: does this command name a FILE after `cat`, `head`, `tail`, `sed`, `awk`, or `grep`? If yes, you MUST issue `Read` or `Grep` instead -- this is not a preference and not a style rule. It holds no matter how the command is dressed: behind `cd X &&`, joined with `&&` or `;`, inside `for f in ...; do ... done`, or piped into `head`. Those composed forms are the ones that feel efficient and are the ones that are forbidden.

To inspect ten files, emit ten `Read` calls in ONE block. That is the same single round-trip as a `for` loop, and it returns numbered lines you can cite as `path:line`; the loop returns an unattributable blob. `git`, `pytest`, `uv`, `ls`, `find`, and builds are proper Bash work -- reading file contents is not.

Prefer dedicated tools; use Bash only when none fits or the user asks.

The Avoid column is not a style preference. `Read` returns numbered lines you can cite as `path:line`; `cat` returns an unattributable blob. `Grep` returns structured hits; `grep | head` silently truncates and you cannot tell a real miss from a cut-off. Shell composition feels faster because it packs more per round-trip -- it is not: parallel tool calls in one block cost the same and stay citable.

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
- NEVER chain to inspect files. `cat`/`sed`/`grep`/`head` with a filename -- alone, behind `cd X &&`, inside `for f in ...; do`, or piped to `head` -- is Bash impersonating Read and Grep, and the chain form is the one that slips past review. Ten files means ten Read calls in ONE block, not one `for` loop. Same round-trip, and you get line numbers to cite.
- Confirm parents with `Glob`/`List` before creating.
- Poll status instead of sleeping.
