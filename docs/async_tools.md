# Async tool calls: three ways to close an interrupted `tool_use`

## The shared constraint

Every major model provider (Anthropic and peers) enforces a pairing rule
on the wire: each `tool_use` block **must** be followed by a matching
`tool_result` before the next user or assistant turn. An unanswered
`tool_use` is rejected (HTTP 400).

This creates a problem the moment a user redirects **mid-tool-call** —
pressing an interrupt key, or typing a new message while a command is
still running. To process the user's input *now*, the runtime must close
the still-open `tool_use` *now*, even though the tool has not produced a
result.

Three agents solve this differently. The axis they differ on is **what
happens to the interrupted tool's eventual result**:

- **codex** — the result is dropped; the slot is filled with a synthetic
  `"aborted"` output.
- **claude-code** — the result is preserved, but relocated out of the
  redirected conversation into an isolated background task.
- **sagent** — the result is preserved and delivered *forward* into the
  same conversation when it arrives.

The one thing all three agree on: **never silently back-patch a slot the
model has already read.** (See "The failure mode all three avoid" below.)

---

## codex: synthesize `"aborted"` in-slot, keep one linear history

codex keeps a single, linear conversation history and repairs it before
every request. When a turn is interrupted, the in-flight `FunctionCall`
is left with no output. Before the next send, `normalize_history` walks
the history and fills any gap:

- `ensure_call_outputs_present`
  (`core/src/context_manager/normalize.rs:17`) inserts a synthetic
  `FunctionCallOutput` immediately after any call that lacks one, with
  the literal text `"aborted"` (`normalize.rs:55`, spliced at
  `normalize.rs:125`).
- `remove_orphan_outputs` handles the reverse case — an output whose
  call is gone (`core/src/context_manager/normalize.rs:144`).
- Both are driven from `normalize_history`
  (`core/src/context_manager/history.rs:361`).

The model is then told what happened via a `<turn_aborted>` user-role
fragment (`core/src/context/turn_aborted.rs:29`):

> "The user interrupted the previous turn on purpose. Any running unified
> exec processes may still be running in the background. If any
> tools/commands were aborted, they may have partially executed."
> (`core/src/context/turn_aborted.rs:9`)

**Sequence** (interrupt, then new message):

1. Abort cancels in-flight tasks; the `tool_use` is left output-less.
2. `normalize_history` writes `"aborted"` into the slot, in place.
3. The `<turn_aborted>` guidance fragment is appended.
4. The user's new message follows. History is now wire-legal, single,
   and linear.

**What happens to the interrupted result:** it is **lost**. codex does
not capture the orphaned process's output and never re-injects it. The
`<turn_aborted>` fragment openly admits the command "may still be running
in the background" and "may have partially executed" — but if the model
needs the outcome, it must re-run or re-check.

**Cost profile:** the simplest of the three. No conversation fork, no
synthetic result-delivery machinery, no separate re-read tool. The cost
is paid by the *model*, which loses the interrupted work and must
reconstruct if it still matters. The `"aborted"` output is written once
and never mutated, so the "silently-edited past" failure (below) cannot
occur.

**Best fit:** hard-interrupt semantics — the user pressed stop *because*
they want the running thing abandoned and the direction changed. Losing
the in-flight result is the desired behavior, and the repair is cheap.

---

## claude-code: fork the conversation, isolate the interrupted work

Based on observed behavior (claude-code is not open source), when a user
backgrounds an in-flight turn (and then types), claude-code appears to **fork
the conversation into two independent transcripts**:

- The foreground turn is aborted. The pending `tool_use` is closed with a
  real interrupted/aborted `tool_result` in the foreground transcript,
  which then continues with the user's new message on a clean slate.
- The in-flight work is **relocated to a background task** carrying a
  *copy* of the transcript up to that point, including the pending
  `tool_use`. That copy runs to completion on its own, resolving the tool
  call in its own message array.

Because the two transcripts are separate arrays, no single slot ever
holds two conflicting truths. Wire-legality is bought by **duplication
and isolation** rather than by any in-place repair.

**What happens to the interrupted result:** it is **preserved, but not in
the redirected thread.** It completes inside the background task. To act
on it, the foreground agent must notice the background task and read its
output through a separate mechanism — it does not automatically re-enter
the conversation the user redirected.

**Cost profile:** conceptually simple on the wire (copy an array, abort
one loop), but the result is stranded from the thread that may need it.
The model must remember to go look; forgetting is an *omission* failure.

**Best fit:** genuine context switches — "drop that, look at this
instead" — where you *want* the interrupted work gone from the current
thread and its eventual result is noise.

---

## sagent: permanent honest stub, forward-delivered result

sagent keeps a **single** conversation and, unlike codex, does not
discard the interrupted result.

On detach, sagent appends a **permanent stub** as the answer to the open
call:

> `[detached: tool still running; real result arrives in a later
> message]`

When the real result eventually arrives, sagent does **not** rewrite the
stub's slot. Instead it appends the result *forward*, at the tail, as a
fresh synthetic `DetachedArrived` `tool_use`/`tool_result` pair. Delivery
is keyed off runtime membership (a `detached` set), not off tape anchors,
so it survives compaction by construction.

**What happens to the interrupted result:** it is **preserved and
delivered into the same conversation**, forward in time, as a real
tool-role result (so `is_error`, attachments, and structure survive).

**Cost profile:** the most machinery of the three. The synthetic pair
needs an alternation guard, and because the arrival-id scheme is
model-visible, the runtime must own that id namespace and rewrite any
model-forged `DetachedArrived` call (otherwise a forged id collides with
a real arrival and wedges the loop). The payoff is one coherent thread
that never loses a result.

**Best fit:** redirects that *refine* rather than abandon — "also run the
linter" while tests are still running. The test result arrives forward in
the same thread and the agent uses it immediately.

---

## The failure mode all three avoid: silent back-patching

The tempting-but-wrong design is to **back-patch**: when the real result
arrives, overwrite the placeholder slot the model already read, in place.

sagent's design doc records why this fails — the *silently-edited past*.
The conversational and thinking text written *while* the slot said
"still running" stays in context verbatim, asserting "I'm still
waiting." The slot they refer to has since been rewritten to the finished
result. The model now holds two contradictory truths with **no event
marking the transition**, and tends to keep trusting its own earlier
narrative. Observed live in sagent: a resolved test result sat paired in
context while the agent insisted across 15+ turns that it "never received
the result."

Each of the three approaches sidesteps this, differently:

- **codex** writes the slot **once** (`"aborted"`) and never mutates it;
  the past stays true.
- **claude-code** never shares a slot between the "waiting" prose and the
  finished result — they live in **separate transcripts**.
- **sagent** keeps the stub permanent and delivers the result **forward**
  as a new record, so the past is never rewritten.

---

## Summary

| | codex | claude-code | sagent |
|---|---|---|---|
| History topology | one linear stream | fork into 2 transcripts | one linear stream |
| Pending call closed by | synthetic `"aborted"` in-slot | foreground abort-result | permanent `[detached]` stub |
| Interrupted result | **dropped** | preserved, isolated in bg task | preserved, delivered forward |
| Transition signal to model | `<turn_aborted>` fragment | (separate thread) | stub + `DetachedArrived` pair |
| Slot ever rewritten? | no | n/a (separate arrays) | no |
| Machinery cost | lowest | low | highest |
| Best-fit redirect | hard interrupt / abandon | context switch / abandon | refine current task |
| Silent-back-patch bug | avoided (write-once) | avoided (separate threads) | avoided (forward delivery) |

There is no universal winner. The choice tracks a single question: **when
a user redirects mid-call, should the interrupted tool's result rejoin
the conversation?** codex says no (cheapest, right for hard stops);
sagent says yes, here (most machinery, right for refinements);
claude-code says yes, but elsewhere (a middle point that keeps the result
without polluting the redirected thread).
