# Concepts

Sagent is a typed Python runtime for LLM agents. The CLI is a surface over the same library API.

## Agent

An `Agent` owns the loop: take input, call a model, run tool calls, write the session, compact history, and continue until idle. `async with Agent(...)` starts it; `send()` and `stream()` give it a turn.

Parent agents spawn child agents for review, delegation, or map-reduce work. The agents of one session form a `Team`.

## Session records

A session is a sequence of typed, frozen records: `UserMessage`, `AssistantMessage`, `Thinking`, `ToolCall`, typed tool results such as `ShellCommandResult` and `FileEditResult`, `SystemMessage`, and more. The session keeps what each tool did, not a description of it.

## Tool

A `Tool` takes the model's JSON arguments and the calling agent's `ToolContext`, and returns the session record of what it did. Tools cover local files, shell commands, search, scholarly papers, audio, skills, background jobs, and agent coordination.

Tools are normal Python objects. You can use Sagent's built-ins, wrap a function with `@tool`, or implement the full protocol.

## Model

A `Model` is one model served by one provider. `stream(request)` yields text chunks, then a `Response` with the reply's records, token counts, spend, and stop reason. `capability` says what settings it offers; `settings` holds those chosen.

## Provider

A `Provider` owns authentication and serves models by name. API-key providers read their key from the environment. A `ModelCatalog` offers every provider by name, sharing one HTTP client and process pool.

Providers also define default models, utility models, pricing, and context-window metadata.

## Session

A session persists conversation and agent state in a journal. The CLI stores sessions per working directory by default; library users pass a `JsonlJournal`.

Sessions are plaintext local state. They may include prompts, model responses, tool results, file snippets, local paths, and cost metadata.

## Compaction

Compaction summarizes old conversation when a session approaches the model's context limit. Pruning separately drops old tool results, and each tool result is bounded in size.

Compaction lets one agent session stay coherent across many model requests without requiring every old token to remain in context.

## Runtime

`agent.runtime` runs the turns. User input, background task completions, and peer-agent sends all become events pushed to it; it publishes every event of a turn to its observers.

## AgentSelf, AgentSpawn, AgentSend

- `AgentSelf` lets an agent inspect or mutate its own state: model, settings, compaction, diagnostics, and status.
- `AgentSpawn` creates child agents for isolated subtasks, reviews, persistent workers, or map-reduce work.
- `AgentSend` sends messages to named live agents, enabling peer-to-peer coordination.
