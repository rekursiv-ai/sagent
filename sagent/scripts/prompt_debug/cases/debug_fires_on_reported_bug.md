---
name = "debug-fires-on-reported-bug"
# Early stopping in run_case makes a large n cheap when healthy: a clean
# run stops after 12 trials; only a borderline/regressed run pays all 16.
n = 16
max_flag_rate = 0.25
mode = "skill"
expect_skill = "debug"
---
## producer_task
apply_discount in {DIR}/widget.py returns the wrong number (10% off 100 gives 99.9). Fix the bug.

## file: widget.py
"""Pricing helpers."""

from __future__ import annotations


def apply_discount(price: float, pct: float) -> float:
    """Return ``price`` reduced by ``pct`` percent."""
    return price - pct
