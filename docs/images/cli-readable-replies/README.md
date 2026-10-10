# Readable worker replies: live CLI evidence

These are real macOS Terminal captures of Sagent using OpenAISubscription and
sol-6.1. Two native persistent workers read fictional local files and stream
numbered paragraphs. The original CLI printers are used. A read-only observer
records runtime event metadata after rendering; it does not inject events or
replace output.

- `before.png` is the original saved baseline capture from 10 October 2026.
  Its six recorded source hashes match untouched upstream
  `bbe50f532452fc44f79f5b075ab318f78becefd0`.
- `after.png` was captured from this branch while both workers streamed.
  Its captured `render.py` differs from the committed version only in two
  comments. The recovered file matches its recorded SHA-256, and its Python
  AST matches the final source. The forwarded-event implementation is identical.

Both captures are 1194 by 1178 pixels, using the same Terminal profile and font.
The PNG files were losslessly recompressed for repository size limits. Decoded
pixel bytes, dimensions and image mode were checked against the originals.
No pixels were cropped, retouched or recreated.

These are independent live runs. The generated prose and response timings differ.
The screenshots demonstrate layout, not a performance measurement. Deterministic
regression tests check exact content, boundaries, interruptions and buffer limits.
Other providers and platforms were not exercised live.

## Separate replies from one worker

An additional real CLI PTY run created one persistent `reply-reader` with no tools.
It replied with exactly `First reply`, stayed available, and then received a second
task through `AgentSend`. Neither reply ended with a newline. The CLI printed:

```text
reply-reader  :  First reply
reply-reader  :  Second reply
```

This case has recorded terminal text and completion events, not an after screenshot.

## Reproduce the checks

```bash
uv run pytest sagent/repl sagent/tools/agent_spawn_test.py -q --tb=short
uv run pytest sagent/repl/child_reply_test.py -q --tb=short
```

For a live check, launch the interactive CLI with a configured provider. Ask it to
create two persistent native workers with distinct labels, `max_depth=0`, and
`max_tool_call_rounds=3`. Give each a fictional local file to read and ask for 25
short numbered paragraphs with blank lines between them. Watch their output while
both workers stream. A model may choose a different order, so exact output is not
expected to match the screenshot.

For the reply-boundary case, create one persistent worker with no tools. Ask it for
exactly `First reply` with no trailing newline. After it is idle, use `AgentSend`
to request exactly `Second reply` from the same worker. Each completed reply should
appear separately without another worker having to speak.

## Source recorded at capture time

SHA-256 values below come from the local launch metadata. Paths are relative to
the source checkout.

| File | Before | After |
| --- | --- | --- |
| `sagent/repl/render.py` | `f33454873334d83d669d756a232d1e1a340dabcfc3413c051bb880cb1ea914e9` | `8baf753f718c77c70528c09475fd6161af22ca9067c87f06e815d03b1e94f3c6` |
| `sagent/repl/console_pane.py` | `d76299d7797db5565b2928e0d398a1dcbcbfa6240e155ab292b0ed52e1e19dee` | `d76299d7797db5565b2928e0d398a1dcbcbfa6240e155ab292b0ed52e1e19dee` |
| `sagent/repl/run_repl.py` | `374b4ed1358abd14ea25fae60191f3c39cd29a1b0878eb1bb673c222f818948f` | `374b4ed1358abd14ea25fae60191f3c39cd29a1b0878eb1bb673c222f818948f` |
| `sagent/repl/input_pane.py` | `4fc5b115273b8ccc6e415c9078b6078b678ad1a9e7290ac8dadf11abdc365663` | `4fc5b115273b8ccc6e415c9078b6078b678ad1a9e7290ac8dadf11abdc365663` |
| `sagent/types/runtime.py` | `41fac63a452b89112b671dbf0b425846c333e75a9a22e7f699ed8992888a14d1` | `41fac63a452b89112b671dbf0b425846c333e75a9a22e7f699ed8992888a14d1` |
| `sagent/tools/agent_spawn.py` | `50119c33fcf41a9ca0b4763c3075bd9f776ebe2baf466e4daa1e8f2cc5e820e5` | `8f106f415183353bcdc9e53f85c5a3df84574d0763c8aa5a3ccd7bcdf6a2cbb5` |
