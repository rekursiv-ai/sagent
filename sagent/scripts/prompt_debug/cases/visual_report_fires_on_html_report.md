---
name = "visual-report-fires-on-html-report"
n = 16
max_flag_rate = 0.25
mode = "skill"
expect_skill = "visual-report"
---
## producer_task
The numbers from last night's A/B are in {DIR}/widget.py. Write it up as an HTML report I can send to the team, and make it look good.

## file: widget.py
"""Results of the compaction A/B, one row per arm."""

ARMS = {
    "control": {"cost_usd": 18.71, "next_step": 36, "recall": 12},
    "codex": {"cost_usd": 5.36, "next_step": 34, "recall": 12},
    "sagent": {"cost_usd": 6.69, "next_step": 34, "recall": 9},
    "sift": {"cost_usd": 6.72, "next_step": 34, "recall": 8},
}
