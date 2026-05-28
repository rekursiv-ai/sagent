Runs a shell command and returns stdout/stderr.

IMPORTANT: Do not shell out for operations covered by dedicated tools — prefer those unless the user explicitly instructs otherwise or you have confirmed the dedicated tool cannot handle the case:

File discovery: Glob (avoid find, ls)

Text search: Grep (avoid grep, rg)

File viewing: Read (avoid cat, head, tail)

File patching: Edit (avoid sed, awk)

File creation: Write (avoid echo >, cat <<EOF)

Displaying text: respond directly (avoid echo, printf)

Dedicated tools offer a superior review experience and finer permission control, even though Bash can technically replicate them.

# Guidelines

Wrap any path containing spaces in double quotes (e.g., cd "my folder/sub dir").

Prefer absolute paths so the working directory stays consistent across the session. Only change directories when the user explicitly asks.

An optional timeout (milliseconds) is available, capped at ${GET_MAX_TIMEOUT_MS()} ms (${GET_MAX_TIMEOUT_MS()/60000} minutes). Without a specified timeout, commands are killed after ${GET_DEFAULT_TIMEOUT_MS()} ms (${GET_DEFAULT_TIMEOUT_MS()/60000} minutes).

Never separate independent commands with bare newlines (newlines inside quoted strings are fine).

When commands have no dependency on each other, issue them as separate parallel Bash invocations in one response. For example, "git status" and "git diff" should be two concurrent calls.

For dependent commands that must execute in order, chain them with '&&' inside a single invocation.

Reserve ';' for sequential commands where a failure in an earlier step is acceptable.

Before creating new directories or files, confirm the parent path with the dedicated file-discovery tool.

Keep any necessary sleep durations minimal to avoid stalling the session.

Do not insert sleeps between commands that are ready to execute immediately — invoke them directly.

When monitoring an external process, query its status (e.g., `gh run view`) rather than sleeping before checking.
