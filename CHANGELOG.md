# Changelog

All notable Sagent changes are documented here.

## Unreleased

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
