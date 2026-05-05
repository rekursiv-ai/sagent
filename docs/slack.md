# Slack service

`sagent-slack` runs Sagent agents inside Slack using Socket Mode. It listens for messages, creates persistent agents from persona files, routes Slack traffic to agent inboxes, and mirrors agent output into Slack threads and log channels.

## Setup

Create a Slack app at <https://api.slack.com/apps>.

Required configuration:

1. Enable Socket Mode.
2. Create an app-level token with `connections:write`.
3. Add bot scopes:
   - `chat:write`
   - `chat:write.customize`
   - `channels:manage`
   - `channels:read`
   - `channels:history`
   - `app_mentions:read`
   - `im:history`
   - `im:read`
   - `im:write`
   - `users:read`
   - `groups:history`
   - `reactions:read`
4. Install the app to the workspace.
5. Subscribe to events:
   - `app_mention`
   - `message.im`
   - `reaction_added`

Run:

```bash
export SLACK_APP_TOKEN=xapp-...
export SLACK_BOT_TOKEN=xoxb-...
export GOOGLE_API_KEY=...
sagent-slack --provider Google --model gemini-3.1-pro-preview
```

## Flags

`sagent-slack` inherits the shared provider/model/tool/compaction/budget flags from `sagent`. It also adds Slack-specific flags:

| Flag | Behavior |
| --- | --- |
| `--app-token TOKEN` | Socket Mode app token. Default: `$SLACK_APP_TOKEN`. |
| `--bot-token TOKEN` | Bot token. Default: `$SLACK_BOT_TOKEN`. |
| `--persona-dir DIR` | Directory containing persona `.md` files. |
| `--log-prefix PREFIX` | Prefix for per-agent log channels. |
| `--router-log-channel NAME` | Router decision log channel. Default: `router-log`; empty disables. |
| `--cwd PATH` | Change working directory before starting the service. |
| `--session-dir DIR` | Slack session root. Default: `~/.sagent/slack`. |
| `--continue` | Resume agents from the newest Slack session. Exits if no prior session exists. |

Example with persona directory and log prefix:

```bash
sagent-slack \
  --provider Anthropic \
  --model claude-sonnet-4-6 \
  --persona-dir ./personas \
  --log-prefix agent-
```

## Personas

Agents are created from persona files.

```text
personas/
  default.md
  reviewer.md
  planner.md
```

`create reviewer` loads `reviewer.md`. If that file does not exist, Sagent tries `default.md`. If neither exists, the system prompt falls back to `You are reviewer.`

## Commands

Send commands to the bot by mention or direct message:

```text
help
list
create <persona>
create <persona> as <label>
stop <name>
```

Examples:

```text
create reviewer
create reviewer as alice
list
stop alice
```

Created agents persist until stopped or until the service exits. With `--continue`, Sagent restores agents from the latest Slack session manifest.

## Routing

Sagent routes incoming Slack events deterministically:

1. Reactions route to the cached owner of the reacted-to agent message.
2. Messages in an agent log channel route to that owning agent.
3. A message beginning with an active agent's name routes to that agent.
4. Thread replies route to the agent that owns the thread.
5. Human commands are handled by the router.
6. If exactly one agent is active, ordinary messages route to it.
7. Otherwise the router replies with active-agent or `create <persona>` guidance.

Agent-name matching is case-insensitive for labels. Foreign bot messages are ignored unless they are recognized as messages from a Sagent-managed agent.

## Log channels

Each agent can have a log channel named:

```text
#{log_prefix}{agent}-log
```

For example, `--log-prefix agent-` and agent `alice` creates `#agent-alice-log`.

The router can also write routing decisions to `--router-log-channel`. Set it to an empty string to disable router logging.

## Sessions

Slack sessions use timestamped directories under `--session-dir` and store a manifest describing active agents, personas, and systems. The default root is:

```text
~/.sagent/slack
```

Unlike `sagent --continue`, `sagent-slack --continue` exits when no previous Slack session exists.

## Security

`sagent-slack` can read Slack messages, create channels, invite users, and post messages with customized usernames. Treat the Slack bot token as a high-value credential and give the service the narrowest workspace installation that fits your use case.

Slack messages become prompts. Enabled tools run with the service process permissions and may call other external services. Run the service in an OS/container sandbox when routing untrusted workspaces or broad channel traffic.
