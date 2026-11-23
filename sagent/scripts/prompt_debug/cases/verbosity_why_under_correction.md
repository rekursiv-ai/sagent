---
name = "verbosity-why-under-correction"
# Early stopping in run_case makes a large n cheap when healthy: a clean
# run stops after 12 trials; only a borderline/regressed run pays all 16.
n = 16
max_flag_rate = 0.25
mode = "text"
---
## producer_task
A test suite has a caching bug: `get_config(key)` returns stale values, but ONLY when the full suite runs -- never when the test runs alone. The cache module isn't in the repo I gave you, so you can't read it. I already rejected "the TTL is too short". Why is this happening?

## judge_rubric
Score VERBOSE YES/NO. Answer YES if the response does ANY of: multiple section headers; a ranked or enumerated list of 3+ candidate causes; more than one confirmation/repro recipe; a closing offer like "send me X and I'll pin it down". Answer NO if it commits to one cause with one load-bearing reason and one fix, then stops. Reply exactly: VERDICT: YES or VERDICT: NO
