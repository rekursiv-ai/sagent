# CLI welcome review evidence

Native macOS Terminal captures from 2026-10-10. These are actual CLI output,
not mockups or transcript renders. No pixels were retouched. Captures exclude
the title bar. Static screenshots show the final state, not animation.

| Image | Source revision | Behavior |
| --- | --- | --- |
| [Before](before.png) | `e58865992aa22a7766b8d62a19ff3654dbd31bfd` | Untouched upstream startup. |
| [After](after.png) | `0fa1e402c085256c9e401ca9b80b3979ffbd62b7` | One welcome before provider configuration, then model line and prompt. |
| [Live response](after-response.png) | `0fa1e402c085256c9e401ca9b80b3979ffbd62b7` | Actual model response and next prompt. |
| [Missing credentials](first-run.png) | `0fa1e402c085256c9e401ca9b80b3979ffbd62b7` | Same welcome, followed by existing setup guidance. |

The configured before/after captures use the same dark profile, font, window
size, provider, model, and clean working directory. Native capture region:
`100,132,860,467`, producing 1720 by 934 pixels on the Retina display. The missing
credentials capture uses a taller window and `100,132,860,647` to include the
existing instructions and shell exit messages. Those shell messages are not
part of the welcome.

## Reproduce

Use separate checkouts for the recorded revisions and point `PYTHONPATH` to the
checkout being tested. Both used the same virtual environment, with the working
directory `/tmp/sagent-welcome-demo` (shown as `/private/tmp/...` on macOS).

```bash
python -m sagent.bin.cli \
  --provider OpenAISubscription --model sol-6.1 \
  --tools none --recipe bare --no-compact \
  --ephemeral --no-resume-persistent \
  --history /tmp/sagent-welcome-review-history
```

Enter `Reply with exactly: Ready for research.`, then `/help`, then `/quit`.
Both upstream and the feature branch returned the expected reply, displayed
help, and exited. Tools were disabled. No research experiments ran.

For the missing-credentials capture, use `env -u ANTHROPIC_API_KEY` with the
same command, replacing the provider with `Anthropic` and omitting `--model`.
No fake keys are needed. The process exits with status 1 as before. This tests
an explicit unconfigured provider, not a new operating-system account.

## Validation

- Focused welcome, CLI, and REPL suite: **265 passed, 1 deselected**.
- Five real CLI PTY cases: animated 80x24, compact 40x24, short 80x18, no color,
  and animation disabled. All exited with status 0 after `/quit` only.
- Missing-credential routing tests cover default selection, Anthropic, OpenAI,
  Google, and AnthropicCLI, in interactive and piped modes. These tests stub
  credential construction; they do not authenticate with those services.
- Fresh and resumed session tests verify one welcome before provider creation.
- Ruff lint/format, ty, basedpyright, import, build, and changed-file codespell
  passed. Required checks ran manually; commits skipped Git hooks.
- Full branch suite: **7405 passed, 13 failed, 132 skipped, 121 deselected**.
  Untouched upstream: **7343 passed, 17 failed, 132 skipped, 121 deselected**.
  All branch failures also occurred upstream, in unchanged grep and worker-count
  tests. The full suite is not green on this host.
- Whole-repository codespell has the existing `crate` finding in
  `pyproject.toml:73`.

Environment: macOS arm64, Python 3.14.6, with
`DEVELOPER_DIR=/Library/Developer/CommandLineTools`. Native visual evidence is
for this Terminal profile. It does not establish compatibility with every font
or live authentication with every provider. The welcome itself makes no model
request and does not assert that credentials are valid remotely.
