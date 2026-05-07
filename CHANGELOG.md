# Changelog

All notable Sagent changes are documented here.

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
