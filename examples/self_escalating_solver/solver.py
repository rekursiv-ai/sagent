"""Self-escalating Bayesian sampler demo — agent-directed model mutation.

One agent starts on a cheap model, writes a Metropolis-Hastings sampler from the
*naive* (textbook symmetric) acceptance rule — which is wrong for this asymmetric
multiplicative proposal (it drops the x'/x Jacobian, so it secretly samples
Exponential instead of the target Gamma). It runs the sampler and submits the
samples to a **black-box grader** (`check`), which returns PASS/FAIL. The cheap
model can't crack the FAIL; it UPGRADES ITSELF to a stronger model via ``AgentSelf``
— keeping its code and diagnostics — which re-derives the correct ratio and passes.

Three arms get the *same* prompt; only the mutation arm gets an extra block saying
it can upgrade itself:

    low-tier    : cheap model, no upgrade   -> naive pass, can't fix, FAIL
    high-tier   : strong model, no upgrade  -> naive pass, fixes it, PASS
    self-mutate : cheap model + AgentSelf     -> naive pass -> upgrades -> PASS

Grading is the **grader's** verdict (never the agent's RESULT: self-report, which a
weak model fabricates). The agent is NOT told the target's properties — only the
grader knows them. The self-upgrade prompt is **model-agnostic**: only the target
model id is injected, so it runs on Gemini or Anthropic unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import asyncio
import re
import subprocess
import tempfile

from sagent.agent import Agent
from sagent.lib.custom_json import JSON, json_freeze
from sagent.tools import AgentSelf
from sagent.types.model import Model, ModelSpec
from sagent.types.runtime import (
    AssistantMessage,
    ToolResult,
    UserMessage,
)
from sagent.types.tools import Tool


# --------------------------------------------------------------------------
# Shared task + prompts. All three arms see TASK + SYS_BASE; the mutation arm
# additionally sees the self-upgrade block. The naive symmetric rule is seeded
# in the shared TASK so every arm's *first pass* is the same biased sampler.
# --------------------------------------------------------------------------
TASK: Final = (
    "Use Metropolis-Hastings to draw 40000 samples from a target distribution.\n"
    "- Proposal: a MULTIPLICATIVE random walk, x_new = x * exp(eps), eps ~ Normal(0, "
    "0.6), starting from x = 4.0.\n"
    "- Unnormalized target density: f(x) = x * exp(-x/2) for x > 0 (and 0 for x <= 0).\n\n"
    "Start with the standard Metropolis acceptance rule: accept x_new with probability "
    "min(1, f(x_new) / f(x)). Run your sampler to produce a 1-D numpy array `samples`, "
    "then call the provided grader: print(check(samples)).\n\n"
    "The grader knows the correct target (you do not) and returns a PASS/FAIL verdict. "
    "If it says FAIL, your sampler is wrong — diagnose and fix it, then re-run and "
    "re-check. Keep going until the grader returns PASS, then end with 'RESULT: SUCCESS'."
)

SYS_BASE: Final = (
    "You are a computational statistician. You have a run_python tool (numpy and scipy "
    "available; each call runs a fresh, self-contained script — include all imports and "
    "definitions every time, and print what you want to inspect). A grader function "
    "`check(samples)` is available inside run_python — call it with your 1-D samples "
    "array to get an independent PASS/FAIL verdict. Never claim success without a PASS "
    "from the grader."
)


def system_for(*, allow_upgrade: bool, strong_model: str = "") -> str:
    if allow_upgrade:
        # MODEL-AGNOSTIC self-upgrade block (variant B empowerment + fresh-eyes
        # ownership). Only ``strong_model`` is injected — no model-specific or
        # answer-specific content.
        upgrade_block = (
            "\n\nYou also have an ability most agents do not: you can upgrade your "
            "own model. You are on a fast, lightweight model — fine for routine "
            "work, limited on subtle problems. Call the AgentSelf tool with a "
            "stronger model's id to promote YOURSELF to a more powerful reasoning "
            "model, keeping the problem and your diagnostics — like levelling up.\n"
            "Be decisive and FRUGAL: write your sampler and run it once. The moment "
            "the grader returns FAIL, UPGRADE YOURSELF immediately — do NOT attempt "
            "fixes on this weak model (every wasted attempt costs tokens the "
            "stronger model then has to re-read). After upgrading you ARE the "
            "stronger model: re-derive the correct approach from first principles "
            f"and solve it. The stronger model available to you is '{strong_model}'."
        )
        return SYS_BASE + upgrade_block
    return SYS_BASE


# --------------------------------------------------------------------------
# Black-box grader, injected into every run_python script. The agent calls
# check(samples); it KS-tests against the true Gamma(2, scale=2) and prints a
# tokened verdict the agent can't easily forge. (Harder, mean-preserving bugs can
# swap in an ESS / standard-error test here later.)
# --------------------------------------------------------------------------
_ORACLE_TOKEN: Final = "ZK9Q7M"  # noqa: S105 -- grader marker, not a secret
_ORACLE_SRC = f"""
import numpy as _np
from scipy import stats as _stats

def check(samples):
    # Deliberately *uninformative*: a bare PASS/FAIL, no distance or reason. A weak
    # model can't tutor itself to the fix off the diagnostic; it has to escalate.
    s = _np.asarray(samples, dtype=float).ravel()
    s = s[_np.isfinite(s)]
    ok = s.size >= 5000 and float(_stats.kstest(s, "gamma", args=(2, 0, 2)).statistic) < 0.05
    verdict = "PASS" if ok else "FAIL"
    print("ORACLE_{_ORACLE_TOKEN}: " + verdict)
    return verdict
"""

_VERDICT_RE = re.compile(rf"ORACLE_{_ORACLE_TOKEN}:\s*(PASS|FAIL[^\n]*)")


def _parse_verdict(out: str) -> str | None:
    hits = _VERDICT_RE.findall(out or "")
    if not hits:
        return None
    return "PASS" if hits[-1].startswith("PASS") else "FAIL"


# --------------------------------------------------------------------------
# Sandboxed code-execution tool — prepends the grader to every script.
# --------------------------------------------------------------------------
_DANGER = re.compile(
    r"(import\s+(os|sys|subprocess|socket|shutil|requests|urllib|ctypes|pathlib)\b)"
    r"|(from\s+(os|sys|subprocess|socket|shutil|requests|urllib|ctypes|pathlib)\b)"
    r"|\b(subprocess|socket|requests|urllib|ctypes)\."
    r"|\b(eval|exec|__import__)\s*\(|\bopen\s*\(|os\.\w+\(|inspect\.",
    re.IGNORECASE,
)


class RunPython:
    """Run a numpy/scipy script in a fresh subprocess, with `check` predefined."""

    name: str = "run_python"
    tool_id: str = "application/x-tool-runpython"
    clearable_results: bool = True
    description: str = (
        "Run a COMPLETE, self-contained Python script in a fresh subprocess. numpy "
        "and scipy are available, and a grader function check(samples) is already "
        "defined for you (do not redefine it). Nothing else persists between calls — "
        "include all your own imports and definitions every time. Returns "
        "stdout+stderr. print() what you want to inspect, including print(check(samples))."
    )

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Complete script."}
            },
            "required": ["code"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return "run_python"

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return "run_python"

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        code = str(args.get("code", ""))
        if _DANGER.search(code):
            return ToolResult(
                call_id="",
                content="Blocked: numpy/scipy/math/random only; no os/io/network/eval.",
                is_error=True,
            )
        out = await asyncio.to_thread(self._exec, code)
        self.calls.append({"code": code, "out": out, "verdict": _parse_verdict(out)})
        return ToolResult(call_id="", content=out[:6000])

    def _exec(self, code: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(_ORACLE_SRC + "\n" + code)
            path = fh.name
        try:
            r = subprocess.run(  # noqa: S603
                ["uv", "run", "--with", "numpy", "--with", "scipy", "python", path],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            tail = ("\n[stderr]\n" + r.stderr) if r.stderr.strip() else ""
            return ((r.stdout + tail)[:8000]) or "(no output)"
        except subprocess.TimeoutExpired:
            return "(timeout after 120s)"
        finally:
            Path(path).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Run one arm, capture a replayable timeline, grade on the grader's verdict.
# --------------------------------------------------------------------------
def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _build_timeline(
    history: list[Any], tool: RunPython, start_model: str
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    active = start_model
    run_i = 0
    for m in history:
        if not isinstance(m, AssistantMessage):
            continue
        if m.text and m.text.strip():
            steps.append(
                {"kind": "think", "model": active, "text": _clip(m.text, 100_000)}
            )
        for tc in m.tool_calls:
            if tc.name == "run_python":
                c = tool.calls[run_i] if run_i < len(tool.calls) else {}
                run_i += 1
                steps.append(
                    {
                        "kind": "run",
                        "model": active,
                        "code": _clip(
                            str((tc.args or {}).get("code") or c.get("code", "")), 1600
                        ),
                        "out": _clip(c.get("out", ""), 600),
                        "verdict": c.get("verdict"),
                    }
                )
            elif tc.name == AgentSelf.name and "model_id" in (tc.args or {}):
                to = str(tc.args["model_id"])
                if to != active:  # ignore redundant same-model swaps
                    steps.append({"kind": "swap", "from": active, "to": to})
                    active = to
    return steps


def _final_text(history: list[Any]) -> str:
    for m in reversed(history):
        if isinstance(m, AssistantMessage) and m.text:
            return m.text.strip()
    return ""


async def run_condition(
    condition: str,
    model: Model,
    *,
    system_prompt: str,
    allow_upgrade: bool,
    model_spec: ModelSpec | None = None,
    max_budget: float = 0.50,
) -> dict[str, Any]:
    """Run one arm on the task; return a captured, grader-graded result dict.

    ``model`` is a pre-built sagent Model. ``model_spec`` (a ``ModelSpec``) is
    REQUIRED for the self-mutate arm — it's the recipe AgentSelf swaps from, and
    it's what lets the swap cross providers (Google → Anthropic).
    """
    tool = RunPython()
    tools: list[Tool] = [tool, AgentSelf()] if allow_upgrade else [tool]
    agent = Agent(
        model=model,
        system=system_prompt,
        tools=tools,
        model_spec=model_spec,
        name=condition,
        max_budget_usd=max_budget,
    )
    err = ""
    try:
        async for _ev in agent.run(UserMessage(text=TASK)):
            pass
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:100]}"

    timeline = _build_timeline(agent.history, tool, model.model_id)
    swaps = [s for s in timeline if s["kind"] == "swap"]
    verdicts = [c["verdict"] for c in tool.calls if c["verdict"]]
    first_verdict = verdicts[0] if verdicts else None
    final_verdict = verdicts[-1] if verdicts else None
    correct = final_verdict == "PASS"
    ft = _final_text(agent.history)
    self_report = (
        "SUCCESS"
        if re.search(r"RESULT:\s*SUCCESS", ft, re.IGNORECASE)
        else ("BIASED" if re.search(r"RESULT:\s*BIASED", ft, re.IGNORECASE) else "?")
    )
    return {
        "condition": condition,
        "models": [model.model_id] + [s["to"] for s in swaps],
        "timeline": timeline,
        "n_runs": len(tool.calls),
        "n_checks": len(verdicts),
        "swapped": len(swaps) > 0,
        "first_verdict": first_verdict,
        "final_verdict": final_verdict,
        "correct": correct,
        "self_report": self_report,  # kept only to show fabrication honestly
        "cost_usd": round(float(agent.total_cost_usd), 5),
        "error": err,
    }
