# Changelog

All notable sagent changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.19 - 2026-10-07

### Added

- `Agent.aclose()` shuts down the agent and awaits model and owned-provider
  cleanup. It is safe to call repeatedly; cancelling a waiter does not
  cancel the cleanup. `shutdown()` remains the synchronous entrypoint.
- Custom tools can declare `max_result_chars` through the optional
  `ResultBounded` protocol. `sagent.types.tools.DEFAULT_MAX_RESULT_CHARS`
  is 50,000, and declarations are capped at that ceiling. Tools that omit
  the property get the default; `0` exempts a self-bounding tool such as
  `Read` from disk off-loading.
- `Agent.prepare_model_change()` builds a model without applying it, and
  `commit_model_change()` queues the swap after the current call finishes.
  A prepared `ModelChange` can be discarded to release its resources.
  This lets callers validate settings against the destination model
  before committing a change.
- Models: Gemini Flash 3.5 through 3.8 (`gemini-flash-3.5` through
  `gemini-flash-3.8`), Flash-Lite 3.1 and 3.5
  (`gemini-flash-lite-3.1`, `gemini-flash-lite-3.5`); Qwen 3.8 Max and
  Flash (`qwen-max-3.8`, `qwen-flash-3.8`), and MiniMax M3
  (`minimax-3.0`). Catalogs include their prices, limits and reasoning
  capabilities.
- `sagent.agent.retry.RetryDeferredError` carries the retry time for a
  service failure with a long advertised wait, such as a 503 with
  `Retry-After`. `RateLimitError` is its subclass for actual throttling;
  service outages no longer appear as rate limits.
- `Agent.own_spend` exposes the agent's own cumulative cost separately
  from the root's aggregate cost for its subagent tree.

### Changed

- `Agent.run()` keeps the model open when a run finishes or is interrupted,
  so the same agent can handle subsequent requests. It stops and awaits
  the run's outstanding model and compaction calls without performing
  terminal cleanup; call `await agent.aclose()` when finished with it.
- Tool results use fixed bounds instead of a per-result threshold derived
  from the context window. The agent off-loads results above their tool's
  character cap, the 200,000-character round budget, or the space left in
  context, preserving a preview and the full output on disk. An explicit
  `ToolResultPolicy.persist_tokens` can impose a tighter token cap; `0`
  leaves the fixed limits in place instead of disabling off-loading.
- `Bash` results have a 30,000-character cap and `Grep` results a
  20,000-character cap. `Read` pages its own output rather than repeatedly
  off-loading a read of an off-loaded file. A page that cannot fit the
  remaining context returns an error asking for `offset` and `limit`.
- After a model refusal or `PolicyBlockedError`, the agent withholds the
  input since the last assistant turn from subsequent requests. Refused
  tool results become error stubs with their call IDs intact, and earlier
  conversation remains available. The blocked request still raises its
  typed error; do not retry the blocked context unchanged.
- **Breaking:** major-only model names now select the newest release in
  that family, not a pinned `.0` model: `sol-6` resolves to `sol-6.1` and
  `opus-5` to `opus-5.5`. Use `sol-6.0` or `opus-5.0` to pin the former
  selections. Family names such as `sol`, `opus` and `gemini-flash` also
  resolve to their newest row, as do roles that name a family.
- Catalog IDs consistently use family/version names, including
  `astra-6.0`, `luna-6.0`, `gemini-pro-3.1`, `qwen-plus-3.7`,
  `kimi-2.6` and `minimax-2.7`. Vendor wire IDs remain accepted:
  `gemini-3.1-pro-preview`, for example, still resolves to
  `gemini-pro-3.1`. Saved sessions and displays use the canonical name.
- Utility models follow the refreshed catalogs: Anthropic uses
  `sonnet-5.5`, Google `gemini-flash-lite-3.5`, and DashScope
  `qwen-flash-3.8`. DashScope defaults to `qwen-plus-3.7`; MiniMax's
  default and utility roles both select `minimax-3.0`.
- Length-truncated responses retain usable prose instead of always raising
  `ModelTerminationError`; a truncated response containing tool calls
  still fails rather than executing incomplete arguments. Malformed tool
  calls are retried, and an unknown vendor stop reason warns instead of
  rejecting the entire response.
- Tool argument validation uses cached, compiled JSON Schema validators
  and requires `fastjsonschema>=2.22.2`. Validation errors distinguish
  missing fields, unexpected fields and invalid values, and do not execute
  the rejected call.
- **Breaking:** custom-tool `directive_schema` and `tool(schema=...)` use
  `Mapping[str, PlainTree]` instead of the former `JSON` alias. Import
  `PlainTree` and `immutable` from `sagent.lib.codec`; examples now use
  `immutable(...)` in place of `json_freeze(...)`.
- **Breaking:** `ModelCatalog`, `UnknownModelError` and
  `UnsupportedTagError` moved from `sagent.types.providers` to
  `sagent.catalog.table`. Construct a catalog with `models=ModelTable(...)`
  instead of `rows=...`. `CONTEXT_TAGS`, `base_model_id` and
  `split_model_id` moved there from `sagent.types.model`.
- **Breaking:** providers must implement `close_sdk()` as part of the
  `Provider` contract; the optional `ProviderCloseable` protocol is gone.
  A model closes only resources it owns; shared SDK clients belong to the
  provider, so closing one model does not strand its siblings.
- **Breaking:** CLI session-source flags are mutually exclusive:
  `--session`, `--ephemeral`, `--resume`, `--continue`, `--resume-all`
  and `--continue-all` can no longer silently override each other. A
  missing `--resume HASH` exits 1 instead of creating a fresh session.
  Token and tool-round limits must be positive, and `--max-budget-usd`
  must be finite and greater than zero.

### Fixed

- Anthropic safeguard classification no longer treats a rejection that
  merely mentions a caching, retention or versioning "policy" as a
  `PolicyBlockedError`. Actual safeguard blocks retain the actionable
  error path rather than being retried as transient failures.
- Model switches and `AgentSelf` settings changes validate against the
  destination model before applying anything, then apply settings after
  the queued swap. Request and response caps above the destination's
  limits are rejected instead of leaving a partially changed agent.
- Cross-provider switches can fall back to the destination's default
  credentials when an inherited named account is missing. Session resume
  can do the same for a saved account. Explicit account selections and
  same-provider model changes remain strict.
- Resuming a model compares provider, authentication mode and account as
  well as model ID, so an identical ID no longer hides a different saved
  credential selection. The resolved fallback account is recorded.
- Only one driver can claim an agent across `run()`, `serve_forever()` and
  `drive_until_first_idle()`. A competing driver fails before stealing
  inbox events or releasing another driver's claim.
- Per-agent budget spend persists across session resume, separately from
  tree-wide spend. Compaction, scrunch and advisor side calls contribute
  tokens and cost without replacing the conversation's prompt-size or
  cache-miss baseline; discarded responses remain accounted for.
- Rewritten history after a failed turn no longer counts the conversation
  twice and triggers premature compaction. Changing only the account or
  authentication mode preserves the same model's measured prompt size.
- Compaction bounds tool-result bodies on the first summary request, not
  just after an overflow. Overflow recovery reduces the cap when that
  changes the request, otherwise drops an old group, so retries make
  progress instead of resending the same oversized prompt.
- Tools that declare their own `background` or `delay` parameter keep it;
  scheduler injection no longer steals `AgentSend`'s delay. Background
  cancellation translates call IDs to job IDs, and finished detached jobs
  are removed from the job-ID mappings.
- Gated-out tools no longer contribute instructions to the system prompt.
- Write-capable `Bash` calls serialize against keyed `Read`, `Edit` and
  `Write` calls, not only other Bash writers. Timeouts kill the process
  group even after its leader exits and bound output draining when a
  surviving descendant keeps a pipe open.
- `Read` rejects oversized image files before loading and sending them,
  with advice to downscale first. Anthropic tool results retain PDF
  attachments, and hyphenated tool names accepted by its API are no
  longer discarded as placeholders.
- Switching to Anthropic drops thinking blocks without a signature or
  body instead of replaying invalid provider-specific history. Thinking
  requests honor show/hide output explicitly and keep fixed budgets at or
  above Anthropic's 1,024-token minimum.
- OpenAI subscription token refresh stays under the cross-process
  credential lock, preventing competing refreshes of one rotating token.
  Tokens without a JWT expiry use the grant's `expires_in`; a grant with
  neither returns an actionable authentication error.
- Wrongly typed scalar metadata in a saved session takes its default
  instead of aborting resume. Off-loaded results without a session use a
  private temporary directory rather than a shared, pre-creatable path.

### Removed

- The `msgspec` runtime dependency; the released package does not require
  it.
- **Breaking:** `sagent.lib.custom_json` was replaced by
  `sagent.lib.codec`. Direct imports of the old module must migrate to the
  shared codec's `from_plain`, `to_plain`, `immutable` and `mutable` APIs.
- **Breaking:** Google's `gemini-1.5-flash`, `gemini-1.5-pro` and
  `gemini-2.0-flash` catalog entries. Selecting these IDs now raises
  an unknown-model error.

## 0.1.18 - 2026-10-03

### Added

- `AgentSpawn` accepts `hot: bool` (default `false`). A hot spawn freezes
  the child's system prompt to a byte-identical snapshot of the parent's
  current one instead of letting the child rebuild a dynamic prompt from
  its own tools (which auto-gains `BackgroundTask` and shows its own
  spawn-depth text), so the child's first request can land on the
  provider's already-warmed prompt cache instead of a guaranteed miss
  (#361). A hot persistent child stays frozen when it is resumed.
- `Agent` accepts `frozen_system: bool` (default `false`); when set,
  `system_prompt()` returns the configured system spec verbatim, skipping
  per-tool prompt contributions and the detached-activity note. Backs
  `AgentSpawn`'s hot mode.
- `sagent.agent.cache_waste`: per-turn prompt-cache miss detection.
  `Agent.record_response` flags an avoidable miss whenever a turn's
  `cache_read` tokens fall short of what the prior turn's prefix should
  have made servable from cache, distinguishing an expected TTL expiry
  from a mutated prefix. Models without a prompt cache, and turns whose
  prompt shrank (compaction, `clear`), are not reported. Detected misses
  accumulate on `CostTracker.cache_misses` (bounded to 500 entries); roll
  them up with `summarize_cache_waste`.
- Models: Claude Fable 5.1 (`fable-5.1`), Opus 5.5 (`opus-5.5`) and
  Sonnet 5.5 (`sonnet-5.5`); GPT-6 Astra (`astra-6`), Sol (`sol-6`) and
  Luna (`luna-6`), and GPT-6.1 Sol (`sol-6.1`). Astra accepts no `none`
  effort, so its effort floors at `low`.
- Every catalog answers the role names `default` and `utility` (and, on
  Anthropic, `best`) wherever a model ID is accepted: `--model best`,
  `/model utility`, `provider.model("utility")`.
- A short model ID selects its provider by catalog membership, so
  `/model`, `AgentSelf` and `AgentSpawn` can switch to `opus-4.8` or
  `luna-6` without naming a provider. Vendor prefixes remain a fallback.
- `--resume-limit N` caps how many sessions the `--resume` picker lists
  (default 20). Listing N rows reads at most N+1 transcripts.
- A session repair command restores sessions whose resumed conversation
  had collapsed to one message (see Fixed):
  `python -m sagent.bin.repair_truncated_sessions [--dry-run] [DIR ...]`.
  It backs up each file before appending the repair, and with no
  arguments it scans every known session.
- The `Skill` tool also discovers project skills in `.claude` and
  `.agents` skill directories, alongside `.sagent`, at every level from
  the filesystem root down to the working directory.
- `Agent(allow_background=False)` offers tools without their
  `background`/`delay` keys and runs such calls in the foreground, so no
  job outlives the turn. `Agent(tool_results=...)` and
  `AgentSpawn(tool_results=...)` take a `ToolResultPolicy` for the
  per-result off-load threshold and the aggregate tool-result budget.
  `Agent.tool_gate` limits which tools the next requests offer.
- When Anthropic's safeguards reject a conversation, the run stops with
  `PolicyBlockedError`, which carries the provider's message and request
  ID, instead of a generic request error. The REPL halts with advice not
  to retry that context: run `/clear`, `/model` or `/quit`.

### Changed

- Model IDs use short catalog names: `opus-4.8`, `sonnet-4.6`,
  `haiku-4.5`, `sol-5.6`, `terra-5.6`, `luna-5.6`. Vendor IDs such as
  `claude-opus-4-8` and `gpt-5.6-sol` still resolve as aliases; the
  status line, prompts and saved sessions show the short form.
- A bare model ID selects the model's largest context window: 1M tokens
  for Fable, Opus 4.6 and newer, and Sonnet 4.6 and newer, and 1M to
  1.05M for GPT-5.4 and newer on the OpenAI API-key provider, where
  0.1.17 defaulted to 200K and 272K. Append `+200k` or `+272k`, or pass
  `--max-request-tokens`, to keep the smaller window; an explicit `+1m`
  is still accepted where the bare ID already provides it. Sessions now
  grow larger before compacting, and OpenAI bills prompts above 272K at
  its long-context rate. `OpenAISubscription` still caps at 272K.
- Default models: Anthropic `default` is `opus-5.5` (was
  `claude-opus-5`) and `utility` is `sonnet-5` (was `claude-haiku-4-5`).
  OpenAI and `OpenAISubscription` default to `astra-6` (was
  `gpt-5.6-sol+1m` and `gpt-5.6-sol`) with `luna-6` as utility (was
  `gpt-5.4-mini`). DashScope, Moonshot and MiniMax name their own utility
  models (`qwen3.6-flash`, `kimi-k2-0905-preview`, `MiniMax-Text-01`)
  instead of falling back to the default model.
- Both OpenAI providers use the Responses API. The API-key provider no
  longer forces effort to `none` when tools are present, so reasoning and
  function tools work together. Requests replay local history with
  `store: false`, keep encrypted reasoning and image-bearing tool
  results, and allow parallel tool calls. Existing sessions need no
  conversion.
- `Glob` returns files only and skips paths `.gitignore` excludes, as
  `fd` does, and runs `fd` when it is installed. Without ripgrep, `Grep`
  skips the same ignored files ripgrep does, so both backends agree. Both
  walk trees with the new `rignore` dependency.
- **Breaking:** fast serving moved from the `+fast` model-ID tag to the
  `priority` service tier. Request it with
  `model_options={"service_tier": "priority"}` on `AgentSelf` or
  `AgentSpawn`, or set `model.settings.service_tier` in Python; there is
  no CLI flag for it. An ID ending in `+fast` is now an unknown model.
  Anthropic's `standard_only` tier is spelled `default`; the request
  still carries `standard_only`, and fast mode still sends only `speed`.
- **Breaking:** `--thinking` and `/thinking` take one word at a time:
  `adaptive`, `on`, `off`, `redact`, `show` or `hide`. The combined
  states (`adaptive-show`, `adaptive-hide`, `on-show`, `on-hide`,
  `off-hide`, `redact-hide`) are rejected; in the REPL, issue
  `/thinking on` and then `/thinking hide`.
- **Breaking:** reasoning effort is spelled `none`, `min`, `low`,
  `medium`, `high`, `xhigh`, `max`. `--effort off` and `--effort minimal`
  now fail validation; use `none` and `min`. `/effort off` still works.
- Context budgets count tokens instead of characters. Tool-result
  bounds, the disk off-load threshold and the compaction trigger use the
  model's token estimate, re-measured at about 2.4 characters per token
  for current Claude models and 3.7 for OpenAI. No tool is exempt from
  off-loading any more: an oversized `Read` or error result is saved to
  disk with a preview, like any other.
- With a compactor configured (the CLI default), older tool results are
  no longer re-truncated on every turn once half the window fills. They
  stay byte-identical until compaction, which keeps the provider's prompt
  cache warm through long tool-heavy sessions; a recorded ten-turn replay
  went from a 69% to a 90% cache-hit rate.
- Compaction fires at one threshold everywhere: the window minus the
  response reservation and buffer. Request and response caps left unset
  follow the active model, so `/model` rescales them while keeping
  explicit `--max-request-tokens`/`--max-response-tokens`, and a model
  switch no longer compacts on its own.
- Pricing: GPT-5.6 Sol, Terra and Luna are billed at $4/$20, $2/$12 and
  $0.20/$1.20 per million input/output tokens (they had been $5/$30,
  $2.50/$15 and $1/$6). Each request is billed at the service tier the
  server reports serving, priority-tier cache traffic at the priority
  rate, and 1-hour cache writes are metered apart from 5-minute ones.
  These figures feed the status line, `AgentSelf` diagnostics and
  `--max-budget-usd`.
- The system prompt's environment block names the model by its ID and
  states a knowledge cutoff only where the catalog records one. The
  default prompt adds rules for following an invoked skill's steps,
  obeying tool nudges, and backgrounding (`cmd &` inside `Bash` does not
  detach; use the tool's `background` parameter), and it narrows when to
  check a fact before answering.
- Transcript files are created owner-only (`0600`, directories `0700`),
  and existing ones are tightened when migrated.
- Requires `httpx2` instead of `httpx`. `pyturbojpeg` 2.x is now allowed
  (the `<2` cap is gone); it needs the libjpeg-turbo 3.x shared library,
  which Ubuntu 24.04 does not package. The repository's
  `install-libjpeg-turbo.sh` installs it.
- **Breaking:** model capability and per-instance settings are separate
  types. `ModelSpec`, `ContextBudget` and `ProviderOptions` are gone: a
  model exposes `capability`, `settings`, `limits` and `tagged_model_id`,
  and `Agent(budget=...)` takes an `AgentSettings`
  (`sagent.types.settings`). `Agent` no longer accepts `thinking`,
  `thinking_state`, `effort`, `show_thinking` or `provider_options`, and
  drops the `effort`, `thinking`, `thinking_state`, `show_thinking`,
  `cache_ttl`, `service_tier`, `latency` and `provider_options`
  properties; set these on `agent.model.settings`. `AgentSpawn` drops its
  `thinking`, `thinking_state` and `effort` arguments.
- **Breaking:** each provider class carries a `catalog` (`ModelCatalog`)
  in place of `DEFAULT_MODEL`, `DEFAULT_UTILITY_MODEL` and its model
  table. `provider.utility_model()` is now `provider.model("utility")`,
  and `model()` no longer takes `max_request_tokens`. An `OpenAICompat`
  subclass sets `ENV_VAR`, `BASE_URL` and `catalog`. The per-vendor
  `sagent.providers.<vendor>.catalog` modules moved to
  `sagent.catalog.<vendor>`, and `supported_provider_options` is gone.
- **Breaking:** `from sagent import Agent` no longer works, and
  `import sagent` no longer binds `sagent.providers` or `sagent.tools`;
  import `sagent.agent.Agent` and the subpackages directly. `sagent.repl`
  no longer re-exports `run_repl` or the pane classes; import them from
  their modules (`sagent.repl.run_repl`).
- **Breaking:** `sagent.lib.custom_json` is codec-based.
  `dataclass_to_json`/`dataclass_from_json` become
  `DataclassCodec.to_json`/`DataclassCodec.from_json`, and the `*_val`
  readers (`dict_val`, `str_val`, `int_val`, ...) become `DictCodec`,
  `StrCodec`, `IntCodec`, and so on. Encoding rejects a value that does
  not match its declared type and keeps a datetime's named time zone;
  `loads` parses at `json.loads` speed while still refusing non-finite
  numbers. `sagent.lib.absent` provides the `ABSENT` sentinel.
- **Breaking:** tool-result sizing is in tokens. `TOOL_RESULT_MAX_CHARS`
  and `sagent.tools.core.truncate` give way to `bound_by_tokens` and
  `truncate_to_budget`, and `Agent.max_result_chars` is now
  `max_result_tokens`. The `BudgetReset` runtime event is gone.
  `PriceKey` replaces `PriceCatalogProduct`, and `TokenCount` and
  `TokenPrice` carry a `cache_write_1h` meter.
- **Breaking:** a model response that ends any way other than finishing
  or calling tools raises `ModelTerminationError`, now a
  `UserFacingError`, instead of coming back as an empty or partial
  assistant turn: a refusal, a token-limit cut-off, a malformed tool call
  or an unrecognized reason. Gemini's `MALFORMED_FUNCTION_CALL` and
  `OTHER` no longer pass as a refusal and a normal finish. A stream that
  announces tool calls but delivers none raises after its retries instead
  of returning what arrived.

### Fixed

- A resumed session could come back with its conversation collapsed to a
  single message when a merged user turn landed on a resume-repair or
  compaction barrier. An append that would drop absorbed content is now
  refused before it persists, and the repair command (see Added)
  restores sessions already affected.
- Concurrent writers to one session file, such as persistent subagents
  and background tasks, serialize their appends, and a torn trailing line
  left by a crash is closed before the next write instead of corrupting
  it.
- A headless run whose session could not be written at all exits 1 with
  an error (`{"error": ...}` under `--output-format json`) instead of
  reporting success.
- A saved tool call with an empty ID or name is dropped at load instead of
  colliding with others and wedging dispatch. Very long call IDs are
  shortened with a hash so their result files fit filesystem name limits.
- An oversized tool result keeps its head and an offset to resume from
  instead of being elided, and the compactor caps one before sending
  rather than overflowing the request. OpenAI's per-item "string too long"
  400 counts as a context overflow, so it shrinks and retries.
- OpenAI token estimates no longer fail on text that contains a
  special-token string such as `<|endoftext|>`.
- The `--resume` picker no longer parses every transcript, a
  `--resume HASH` prefix match reads directory names only, and resolving
  a long session's context is no longer quadratic in its length.
- Re-reading a file whose earlier `Read` result was off-loaded to disk no
  longer grows the stored copy with nested line-number gutters.
- A detached task keeps its claim until it exits, so a second driver can
  no longer attach to the same inbox, and a message arriving mid-stream
  no longer races a backgrounded tool group's edits.
- Compaction calls, and responses that end in an error, count toward the
  session's cost and its `max_budget_usd` cap.

### Removed

- `AgentSpawn(session_root_dir=...)`, `AgentSpawn.on_persistent_spawn`,
  `AgentSpawn.on_persistent_stop`, and `Agent.rebuild`. Child session
  dirs now always derive from the parent's `session_dir`. Removing the
  `session_root_dir` rebuild branch also fixes a resumed persistent child
  receiving the IPC rule twice.
- **Breaking:** the OpenAI models `gpt-5.4-nano`, `gpt-5.3-codex`,
  `gpt-5.3-codex-spark`, `gpt-5.3-chat-latest` and `o1-mini` are no
  longer in the catalog; selecting one raises an unknown-model error.
- `discover(import_roots=...)` in `sagent.tools.skill` (all three skill
  directories are always scanned) and `sagent.sessions.parse_jsonl`.

## 0.1.17 - 2026-08-19

### Added

- `sagent.lib.custom_json` round-trips sets: a `set`, `frozenset`, or
  `Set`-annotated field encodes to a JSON array and decodes back to the
  declared container. A `frozenset` field used to come back as a list,
  leaving an unhashable, mutable value on a frozen dataclass.

### Changed

- Requires wesearch 0.1.11 or newer.
- `WebFetch` and `WebSearch` build their directive schemas from wesearch's
  shared parameter spec instead of restating every parameter's name, type,
  and default locally, so the tool and MCP surfaces can no longer drift
  apart. `WebFetch` also accepts a lowercase `method`, which a model writes
  as readily as the uppercase form.
- `WebSearch` no longer rewrites an explicit `backend` when `categories`
  names a non-general tab. Asking for `duckduckgo` with `science` silently
  ran against SearXNG and returned results the caller had not asked for;
  that combination is now rejected, and an unnamed backend is left for
  wesearch to resolve.
- The `WebFetch` description now gives a measured extractor rule: reach for
  `trafilatura` only when the page is one contiguous prose body. Over an
  11-page corpus it dropped 12 of 37 content probes where `html2text`
  dropped none, and no property of the page predicts which.
- The default system prompt is firmer about evidence gathered to confirm a
  position already held, and about treating an opinion request as an input
  to build on rather than a claim to adjudicate.
- `sagent.lib.userdirs` functions take no application name. `data_dir()`
  and its siblings return the base directory and the caller joins its own
  namespace (`data_dir() / "rekursiv-ai" / "sagent"`), spelled the same way
  as any other vendor's directory; `resolve_working_dir` is gone. Existing
  files stay where they are.
- `dataclass_from_json` raises `SchemaError` (a `ValueError`) when the
  payload names a field the target cannot accept. Dropping the key turned
  every misspelling into a silent default, so a caller reading
  `{"min_digit": 7}` got the default and no indication of it.

### Fixed

- An optional nested-dataclass field accepts a null: the `None` check runs
  before annotation dispatch, which `_strip_optional` had been reducing
  past and then rejecting.
- `dataclass_to_json` skips `init=False` fields. The generated `__init__`
  rejects them by name, so the encoder had been emitting a payload its own
  decoder could not read.
- An ambiguous scalar union keeps its type tag when `None` is a member, so
  a `Path | bytes | None` field no longer encodes to a bare string that
  decode cannot attribute to either member.

## 0.1.16 - 2026-08-11

### Added

- `WebFetch` takes an `extractor` argument (`trafilatura` by default,
  plus `html2text`, `markdownify`, `raw`). trafilatura returns only what it
  scores as the article, which is wrong for a page that is not
  article-shaped -- a Q&A thread came back without its answers.
- `Read` takes `last_lines`, for content expected near the end of a file.
- Bash lint nudges a file-inspection shell command toward `Read` or `Grep`,
  and names the equivalent call.

### Changed

- Requires wesearch 0.1.10 or newer, which folds the former `[extract]`
  extra into its core dependencies; drop the suffix from any install of
  sagent's dependency.
- Config and data live under a single org namespace, resolved through the
  XDG user directories rather than hand-built paths.
- Tool results are bounded by a character budget instead of per-tool
  truncation caps, so diagnostics survive the cut.

### Fixed

- Rendered bash fragments are escaped and delimited, so a command
  containing quotes or newlines no longer corrupts the surrounding text.

### Changed

- Requires wesearch 0.1.8 or newer.

### Fixed

- `WebFetch` validates the target host against wesearch's `public_host`,
  so a URL resolving to a private or link-local address is refused rather
  than fetched from inside the network the agent runs in.

## 0.1.14 - 2026-08-01

### Changed

- Requires wesearch 0.1.7 or newer.
- README carries a one-line description below the badges; PyPI renders the
  README, so the project page had been showing the previous text.

## 0.1.13 - 2026-08-01

### Changed

- Requires wesearch 0.1.6 or newer.
- Model metadata moved into per-provider capability catalogs, and the
  provider modules are grouped into one subpackage per family.

### Fixed

- Every effort a model advertises now reaches the wire, and a model that
  advertises none no longer receives a thinking knob it would reject.
- Entitlement errors are classified as fatal rather than retried, and the
  provider's message is surfaced in the suspension banner.

## 0.1.12 - 2026-07-29

- Fixed CLI subscription authentication: `AnthropicCLI` now recognizes Claude
  Code's native macOS login, `OpenAISubscription` honors `$CODEX_HOME`, and
  zero-flag startup only tries allowed subscription providers.
- Added GPT-5.6 Sol, Terra, and Luna to the OpenAI API-key and subscription
  providers, with GPT-5.6-specific `xhigh` and `max` effort handling. API-key
  users can select the 1.05M-window variants; subscription auth accepts the
  same IDs but clamps them to its 272K backend contract. GPT-5.6 Sol is now the
  OpenAI provider default. Because GPT-5.6 Chat Completions rejects function
  tools with reasoning enabled, the API-key transport forces effort to `none`
  for tool-using requests; use the subscription Responses transport for
  reasoning and tools together.
- Fixed `OpenAISubscription` credential loading to report API-key-shaped Codex
  auth files as a clean configuration error instead of raising `KeyError`.
- Fixed headless runs to exit nonzero and emit an error payload when the model
  call fails, instead of returning an empty successful result.
- **Breaking:** removed the `--provider-arg Class.key=JSON` CLI flag and the
  untyped `Agent(provider_args=...)` bag. Provider construction knobs are now
  typed fields on `types.providers.ProviderOptions`
  (`Agent(provider_options=...)`), validated against each provider class's
  `supported_options` declaration -- an unsupported explicitly-set option
  raises instead of being dropped with a warning. The
  server-side-context-management opt-in moved to an explicit
  `--server-side-context-management` flag, and the `Class.thinking=...`
  pseudo-key is gone (use `--thinking`). Programmatic `from_key` construction
  is unchanged (`Anthropic.from_key(...)`); `build_provider` no longer
  forwards arbitrary kwargs. Legacy session records with `provider_args`
  still load (known keys map onto `ProviderOptions`).
- Added fast mode as a model-id option tag, mirroring `+1m`:
  `claude-opus-4-8+fast` (composable: `claude-opus-4-8+1m+fast`) works
  everywhere a model id does -- `--model`, `/model`, subagent specs,
  session persistence. Providers validate the tag at `model()`
  construction: models without a fast path (and the CLI-wrapping
  provider) reject it with a `ValueError`. `Agent.latency` is now
  read-only, derived from the model id.

## 0.1.3 - 2026-05-07

- Fixed SelfHosted tool-call allowlist matching so CLI tool names such as
  `Bash` dispatch model-emitted tool calls such as `bash`.
- Added SelfHosted generation throughput diagnostics with
  `output_tokens_per_sec` in DEBUG logs.
- Added opt-in `torch.compile` support for SelfHosted models via
  `SelfHosted.from_hf(..., compile_model=True)` and the inline `+compile`
  model option.
- Documented SelfHosted inline device, dtype, and compile configuration.

## 0.1.2 - 2026-05-05

- Added public SelfHosted provider support for Hugging Face causal language
  models, including Qwen 3.6 examples and local/cache path loading.
- Added SelfHosted auto-device selection: MPS first, then CUDA, then PyTorch's
  CPU default.
- Added CLI controls for model-only local runs with `--tools none` and
  `--max-response-tokens`.
- Added SelfHosted Qwen effort support so `--effort none` disables local
  thinking traces for short smoke tests.
- Added attention masks for SelfHosted generation when pad and EOS tokens share
  the same ID.
- Added SelfHosted guardrails for unsupported chat-template options,
  malformed tool-call blocks, and unadvertised tool names.
- Added SelfHosted load/generation diagnostics, CLI log-level controls, and
  moved blocking local generation off the asyncio event loop for REPL
  responsiveness.
- Updated documentation and examples to current model names.
- Updated Torch dependencies to `torch==2.11.0`, `torchvision==0.26.0`,
  `torchaudio==2.11.0`, and `torchao==0.17.0`.

## 0.1.1 - 2026-05-05

- Initial public release.
- Added a typed Python agent runtime with importable `Agent`, `Model`,
  `Provider`, `Tool`, and `Message` contracts.
- Added CLI, headless JSON output, session persistence, context compaction,
  streaming, and Slack entry points over the same agent loop.
- Added providers for Anthropic, OpenAI, Google, Moonshot, DashScope, MiniMax,
  and OpenAI-compatible endpoints.
- Added local file, shell, web, paper-search, audio, skill, wiki, and
  agent-coordination tools.
- Added multi-agent primitives for self-inspection, child-agent spawning, and
  peer messaging.
- Added public documentation, examples, security notes, packaging metadata, and
  GitHub release/publish workflows.
