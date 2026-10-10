# Progress and pending work in the CLI

An agent can report a checkpoint with `AgentSelf`:

```json
{
  "checkpoint": {
    "progress": "Two inputs passed the official checker",
    "pending": "Result from the third input",
    "user_action": ""
  }
}
```

The latest report stays beside the composer, including after the agent's turn ends.
`/pending` shows the full progress, pending work, required human action, sender and report time without calling the model.
The toolbar clips terminal cells after measuring the current width; the complete fields remain available in `/pending` and the saved session.
If multiple workers report, each keeps its own checkpoint. A report with user action takes priority in the toolbar; `/pending` lists all reports.

Use an empty `pending` or `user_action` for none reported. Use explicit wording when a state is unknown.
Update the checkpoint when evidence, pending work or required action changes. Do not repeat unchanged waiting prose or poll merely to restate it.
Use `checkpoint: null` to clear the report when it is no longer relevant.

Changed checkpoints produce a transcript entry and an append-only `checkpoint_changed` runtime record in `session.jsonl`.
Setting the same checkpoint again keeps the persistent display without a second transition. Tool calls and their receipts remain separate events.
Ordinary assistant replies, delivered messages, errors and results are never deduplicated by their text.
The mechanism gives agents a quieter reporting channel; it does not filter existing prose or guarantee that every model chooses it.

Checkpoints are agent reports, not independently verified job state.
Agent idleness does not establish assignment completion. A job can outlive an agent turn.
On resume, checkpoints are labelled as saved and not rechecked until the agent explicitly reports again.
Child reports retain their event provenance and remain inspectable after the child stops serving.

The presentation follows the persistent status and separate detail pattern documented in the [Hermes CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli#status-bar).
It adds no task scheduler or team dashboard.
