Manage background tasks, analogous to bash job control (`&`,
`jobs`, `kill %N`, `fg %N`).

Any tool call can be backgrounded by setting `background: true` in
its parameters. The tool returns immediately with a task id; the
actual work runs asynchronously. When a detached tool finishes, you
receive a later message with its result. Wait for that message instead
of polling for completion. An optional `delay: N` (seconds) makes the
task sleep before executing (implies `background: true`).

Operations:
- `list` -- show all background tasks with id, tool name, phase
  (sleeping/running/completed), and elapsed time.
- `cancel` -- cancel a task by id. Tasks that are sleeping are
  cancelled before they start; running tasks are interrupted.
- `foreground` -- await a specific task by id, then return its
  result as this tool's result. Blocks until the task completes.

Arguments:
- `operation` (required) -- one of `list`, `cancel`, `foreground`.
- `id` (required for `cancel` and `foreground`) -- the task id
  returned when the tool call was backgrounded.
