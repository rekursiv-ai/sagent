# Worker message display

The before and after images are exports from Sagent's actual Rich terminal
renderer using the same controlled inputs. They are not native operating-system
screenshots or separate model runs. Rich supplies the terminal window decoration.

The sequence contains a human prompt, one individual worker delivery, then a
batch containing another worker's delivery and the first worker's idle notice.
Both workers legitimately use the same wording: `Ready for review.`

Before, the batch uses the grey human input bar. After, each part shows its own
sender. The human prompt keeps the input bar. Both deliveries and the idle notice
remain visible.

The before export used upstream revision
`bbe50f532452fc44f79f5b075ab318f78becefd0`. The after export used this fix. The
subsequent upstream change in `a756247` only adds a Torch typing declaration;
the renderer and runtime files used for the before export are unchanged.

## Live check

A separate, isolated CLI run on macOS used OpenAISubscription and `sol-6.1`.
Two native workers, `review-a` and `review-b`, each called `AgentSend` with
`Ready for review.` Their deliveries arrived during the parent's response and
were batched into one user-role message. The terminal showed:

```text
[from review-a]: Ready for review.
[from review-b]: Ready for review.

[from review-a]: [review-a is idle] Ready for review.
[from review-b]: [review-b is idle] Ready for review.
```

The run was recorded from a PTY. A read-only event observer confirmed both
native senders, the combined message's sender metadata, and no model errors.
The observer did not inject messages or replace rendering. The input history
contained the initial request and `/quit`, with no pasted worker replies.
These images illustrate the controlled renderer comparison; they are not
screenshots of that live run. Other providers and platforms were not exercised
live.

## Regression checks

```bash
uv run pytest -q sagent/repl/message_attribution_test.py
```

The 18 cases cover live rendering, replay, save/load, direct and queued runtime
deliveries, mixed human and worker batches, hidden parts, repeated merges,
child output, plain terminals at different widths, and unchanged provider text.
A human typing a sender label remains human input. Older saved batches without
sender metadata retain their previous display.
