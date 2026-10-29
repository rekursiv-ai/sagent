Manage tool calls running in the background, like bash job control (`&`, `jobs`, `kill %N`, `fg %N`).

Any other tool call may set `background: true` to run in the background, or `delay: N` to start after N seconds (which also runs it in the background). The call is answered at once with `[Running in background: <tool> as job <id>]`; its result arrives in a later message on its own. Wait for that message rather than polling.

Operations:
- `list` -- each background job: its id, tool, phase (`sleeping` until its delay ends, then `running`), and seconds since it was asked for.
- `cancel` -- stop the job `id`. A sleeping job never starts; a running one is interrupted. Its cancellation arrives in a later message.
- `foreground` -- wait for the job `id` and return its result as this call's result, instead of in a later message.

`id` is required for `cancel` and `foreground`.
