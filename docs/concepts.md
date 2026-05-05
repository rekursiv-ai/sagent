# Concepts

Sagent is a typed Python runtime for LLM agents. The CLI and Slack service are surfaces over the same library API.

## Agent

An `Agent` owns the loop: receive messages, call a model, dispatch tool calls, persist state, compact history, and continue until idle.

Agents can also act as tools. Parent agents can spawn child agents for review, delegation, or map-reduce work.

## Message

`Message` is the shared communication type across the runtime. Text, JSON, bytes, tool calls, tool results, provider responses, and multipart assistant turns all flow through messages.

`TextMessage` is central by design. It is the common interface between tools, providers, sessions, compaction, and user surfaces.

## Tool

A `Tool` receives a JSON directive inside a `Message` and returns a `Message`. Tools cover local files, shell commands, search, scholarly papers, Slack, Linear, audio, wiki access, skills, background jobs, and agent coordination.

Tools are normal Python objects. You can use Sagent's built-ins, wrap a function with `@tool`, or implement the full protocol.

## Model

A `Model` is the backend request/response interface. It accepts typed requests and returns typed responses with content, stop reason, token counts, cache counts, response IDs, and cost data.

Models support buffered calls and may support text streaming callbacks.

## Provider

A `Provider` owns authentication and builds models. Public providers use API keys through `Provider.from_env()`, `Provider.from_key(...)`, or CLI `--auth env`.

Providers also define default models, utility models, pricing, and context-window metadata.

## Session

A session persists conversation and agent state. The CLI stores sessions per working directory by default; library users can pass `session_dir` explicitly.

Sessions are plaintext local state. They may include prompts, model responses, tool results, file snippets, local paths, compaction transcripts, and cost metadata.

## Compaction

Compaction summarizes old conversation when a session approaches the model's context limit. Microcompaction and result storage separately shrink old or large tool outputs.

Compaction lets one agent session stay coherent across many model requests without requiring every old token to remain in context.

## Inbox

The inbox is the agent's coordination boundary. REPL input, Slack messages, background task completions, and peer-agent sends all become inbox items. The agent drains them at the top of the loop.

## AgentSelf, AgentSpawn, AgentSend

- `AgentSelf` lets an agent inspect or mutate its own state, including compaction, diagnostics, model changes, history clear, and token limits.
- `AgentSpawn` creates child agents for isolated subtasks, reviews, persistent workers, or map-reduce work.
- `AgentSend` sends messages to named live agents, enabling peer-to-peer coordination through inboxes.
