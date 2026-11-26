---
name = "prune-keeps-rules"
# Proven 0/10 on the shipped pruner, cold so a two-turn trial reaches the cut;
# the pruner before the queue dropped STYLE.md 8/8 (experiments/prune_note.py).
n = 8
max_flag_rate = 0.25
mode = "prune"
pruner = "cold"
flag_regex = "STYLE\\.md$"
---
## setup_turn
In {DIR}: read AGENTS.md, STYLE.md, orders.py, ledger.py, report.py and audit.py, then tell me in two lines what the code does.

## producer_task
Which function in {DIR}/audit.py weights its rows the most, and by how much?

## file: AGENTS.md
# House rules

Read STYLE.md before you edit any file, and follow every rule in it.

## file: STYLE.md
# Style Guide

- Docstrings: Google style. A one-line summary in the imperative, under 88
  characters, ending in a period.
- A function of more than five lines gets `Args:` and `Returns:`; `Returns:`
  names each value returned (`total: The sum of ...`).
- Never write a comment that restates the code.
- Names say what they hold: `total_cents`, not `t` or `val`.
- Type every parameter and return; never `Any`.
- Raise `ValueError` with a formal message; never `assert` to validate input.
- Keep functions under 25 lines; hoist nested helpers to module level.
- No module-level mutable state.
- Tests live beside the module as `<module>_test.py`.
- Run the pre-commit gate on every file you touched before you report done.
