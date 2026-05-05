Send a text message to another live agent's inbox.

Use this tool to:
- Coordinate with sibling agents without going through a parent.
- Share partial results, status updates, or requests between agents.
- Set a delayed reminder for yourself: send a message to your own
  label with a `delay` to wake yourself up in the future.

Arguments:
- `to` (required) — label of the target agent. Active agents are
  listed above; messaging a non-existent label is an error.
- `content` (required) — the message text.
- `delay` (optional) — seconds to wait before delivering. The
  message is delivered asynchronously; the tool returns immediately.
  The delivered message notes how long ago it was sent.

Behavior: fire-and-forget. The message lands in the target's inbox
and is surfaced at the start of their next response. The sender does not
block waiting for a reply. Your own label is attached automatically
as the sender.
