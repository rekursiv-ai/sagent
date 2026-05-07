Interact with Slack through the Web API.

Requires a Slack bot token configured when constructing the tool. Use Slack only when the user explicitly asks for Slack work or when the workflow requires sending or reading Slack messages.

Operations:
- `send` — post `text` to `channel`; optional `thread_ts` replies in a thread.
- `list_channels` — list channels visible to the bot; accepts `limit`.
- `list_messages` — list recent messages from `channel`; accepts `limit`.
- `read_thread` — read replies for `thread_ts` in `channel`; accepts `limit`.
- `list_users` — list workspace users; accepts `limit`.
- `create_channel` — create `channel_name`.

Slack calls are externally visible shared-state mutations. Get explicit user approval before sending messages, creating channels, or otherwise changing Slack state unless the user already authorized that exact action.
