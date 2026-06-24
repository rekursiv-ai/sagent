"""Driver for the self-escalating Bayesian sampler demo.

Default mode is **replay** — it just points you at the webpage, which replays a
captured success run from ``web/data.js`` (deterministic, no API key needed).

``--live`` runs the three conditions for real against a provider, grades them in
the harness, measures the self-mutate success rate over ``--trials`` runs (default
4), captures one success path as the hero timeline, and overwrites ``web/data.js``.

    # replay the logged success run (no key)
    uv run python -m examples.self_escalating_solver.run

    # run live (needs the provider's key configured) and re-capture
    uv run python -m examples.self_escalating_solver.run --live --provider google --trials 4
"""

from __future__ import annotations

from pathlib import Path

import argparse
import asyncio
import json
import os
import subprocess
import tempfile

from examples.self_escalating_solver import solver


HERE = Path(__file__).parent

# Each config names the (provider, model_id) for the cheap and strong tiers. The
# prompt is model-agnostic; only these change. "cross" is the headline: a Google
# cheap model that upgrades itself ACROSS vendors to an Anthropic strong model.
# Each config: cheap tier, the high-tier baseline model, and the model the
# self-mutate agent UPGRADES to. For "cross" they differ on purpose — the
# high-tier baseline is expensive Opus, while the self-mutator upgrades only to
# the cheaper Sonnet, so adaptive self-mutation beats always-Opus on cost.
CONFIGS = {
    "google": {
        "cheap": ("Google", "gemini-2.5-flash-lite"),
        "high": ("Google", "gemini-3.1-pro-preview"),
        "mutate": ("Google", "gemini-3.1-pro-preview"),
    },
    "anthropic": {
        "cheap": ("Anthropic", "claude-haiku-4-5"),
        "high": ("Anthropic", "claude-opus-4-8"),
        "mutate": ("Anthropic", "claude-opus-4-8"),
    },
    "cross": {
        "cheap": ("Google", "gemini-2.5-flash-lite"),
        "high": ("Anthropic", "claude-opus-4-8"),
        "mutate": ("Anthropic", "claude-sonnet-4-6"),
    },
}

_KEY_FILE = {"Google": "google_api_key", "Anthropic": "anthropic_api_key"}
_KEY_ENV = {
    "Google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "Anthropic": ("ANTHROPIC_API_KEY",),
}


def _read_key(provider_name: str) -> str | None:
    # Prefer a file (so the key never has to be exported into the CLI's env);
    # fall back to env if already set for this process.
    for e in _KEY_ENV[provider_name]:
        if os.environ.get(e):
            return os.environ[e]
    kf = Path.home() / ".config" / "sagent" / _KEY_FILE[provider_name]
    return kf.read_text().strip() if kf.exists() else None


def _provider(provider_name: str):
    from sagent.providers import Anthropic, Google

    key = _read_key(provider_name)
    if provider_name == "Google":
        return Google.from_key(key) if key else Google.from_env()
    return Anthropic.from_key(key) if key else Anthropic.from_env()


def build(provider_name: str, model_id: str):
    """Return (Model, ModelSpec) for one arm. The spec is what AgentSelf swaps from."""
    from sagent.types.model import ModelSpec

    model = _provider(provider_name).model(model_id)
    spec = ModelSpec(provider=provider_name, auth="api", model_id=model_id)
    return model, spec


# Canonical biased/fixed sampler histograms for the viz (illustrative — the same
# bias the agent hits). Run in a subprocess so numpy/scipy need not be in this env.
_CANON_HIST = r"""
import json
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
logp = lambda x: stats.gamma.logpdf(x, a=2, scale=2)

def mh(correct, n=120000, sigma=0.6, x0=4.0):
    x = x0; out = np.empty(n)
    for i in range(n):
        xp = x * np.exp(rng.normal(0, sigma))
        la = logp(xp) - logp(x)
        if correct:
            la += np.log(xp) - np.log(x)   # the Jacobian the buggy version drops
        if np.log(rng.random()) < la:
            x = xp
        out[i] = x
    return out[2000:]

edges = np.linspace(0, 20, 41)
centers = (edges[:-1] + edges[1:]) / 2
binned = lambda s: np.histogram(s, bins=edges, density=True)[0].tolist()
print("HIST " + json.dumps({
    "centers": centers.tolist(),
    "biased": binned(mh(False)),
    "fixed": binned(mh(True)),
    "true_gamma": stats.gamma.pdf(centers, a=2, scale=2).tolist(),
    "exp_ref": stats.expon.pdf(centers, scale=2).tolist(),
}))
"""


def canonical_histograms() -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_CANON_HIST)
        path = fh.name
    try:
        r = subprocess.run(
            ["uv", "run", "--with", "numpy", "--with", "scipy", "python", path],
            capture_output=True,
            text=True,
            timeout=180,
        )
        for line in r.stdout.splitlines():
            if line.startswith("HIST "):
                return json.loads(line[5:])
        return {}
    finally:
        Path(path).unlink(missing_ok=True)


def _slim(s: dict) -> dict:
    return {
        k: s[k]
        for k in (
            "swapped",
            "first_verdict",
            "final_verdict",
            "correct",
            "self_report",
            "cost_usd",
            "n_runs",
            "n_checks",
            "error",
        )
    }


async def run_live(config_name: str, trials: int) -> dict:
    cfg = CONFIGS[config_name]
    cheap_prov, cheap_id = cfg["cheap"]
    high_prov, high_id = cfg["high"]  # expensive baseline (high-tier panel)
    mut_prov, mut_id = cfg["mutate"]  # what the self-mutate agent upgrades to
    # Make the Anthropic key discoverable under the canonical env name so a
    # cross-provider AgentSelf swap (Google → Anthropic) can build the provider.
    # Set on THIS process only — never the CLI's env.
    if "Anthropic" in (cheap_prov, high_prov, mut_prov):
        akey = _read_key("Anthropic")
        if akey:
            os.environ["ANTHROPIC_API_KEY"] = akey
    print(
        f"config={config_name}  cheap={cheap_prov}/{cheap_id}  high={high_prov}/{high_id}"
        f"  mutate→{mut_prov}/{mut_id}"
    )

    # low-tier and high-tier get the IDENTICAL prompt (SYS_BASE); only the model differs.
    base_prompt = solver.system_for(allow_upgrade=False)
    print("• low-tier (cheap) …", flush=True)
    low = None
    for _ in range(
        3
    ):  # a 503 can leave it with no runs; retry so it actually tries+fails
        low_model, _ = build(cheap_prov, cheap_id)
        low = await solver.run_condition(
            "low-tier",
            low_model,
            system_prompt=base_prompt,
            allow_upgrade=False,
            max_budget=0.12,
        )
        if low["n_runs"] > 0:
            break
        print("    (low-tier ran no code — likely a 503; retrying)", flush=True)
    print(
        f"    first={low['first_verdict']} final={low['final_verdict']} "
        f"correct={low['correct']} ${low['cost_usd']}"
    )

    print("• high-tier (expensive baseline) …", flush=True)
    high_model, _ = build(high_prov, high_id)
    high = await solver.run_condition(
        "high-tier",
        high_model,
        system_prompt=base_prompt,
        allow_upgrade=False,
        max_budget=0.60,
    )
    print(
        f"    first={high['first_verdict']} final={high['final_verdict']} "
        f"correct={high['correct']} ${high['cost_usd']}"
    )

    # self-mutate starts cheap, carries a ModelSpec, and upgrades to mut_id
    # (a DIFFERENT, cheaper provider/model than the high-tier baseline).
    mutate_prompt = solver.system_for(allow_upgrade=True, strong_model=mut_id)
    self_runs: list[dict] = []
    for i in range(trials):
        print(f"• self-mutate trial {i + 1}/{trials} …", flush=True)
        cheap_model, cheap_spec = build(cheap_prov, cheap_id)
        s = await solver.run_condition(
            "self-mutate",
            cheap_model,
            system_prompt=mutate_prompt,
            allow_upgrade=True,
            model_spec=cheap_spec,
            max_budget=1.00,
        )
        self_runs.append(s)
        print(
            f"    swapped={s['swapped']} models={s['models']} first={s['first_verdict']} "
            f"final={s['final_verdict']} correct={s['correct']} ${s['cost_usd']} err={s['error']!r}"
        )

    money = [s for s in self_runs if s["swapped"] and s["correct"]]
    end_correct = [s for s in self_runs if s["correct"]]
    # Hero = a clean money-path run if we have one, else the best available.
    hero = (
        money[0]
        if money
        else (next((s for s in self_runs if s["correct"]), None) or self_runs[0])
    )

    print("• canonical histograms …", flush=True)
    hist = canonical_histograms()

    data = {
        "meta": {
            "config": config_name,
            "cheap": cheap_id,
            "high": high_id,
            "mutate": mut_id,
            "cheap_provider": cheap_prov,
            "high_provider": high_prov,
            "mutate_provider": mut_prov,
            "trials": trials,
            "captured": True,
        },
        "success": {
            "trials": trials,
            "money_path": len(money),
            "end_correct": len(end_correct),
        },
        "panels": {"low_tier": low, "high_tier": high, "self_mutate": hero},
        "self_mutate_trials": [_slim(s) for s in self_runs],
        "hist": hist,
    }
    out = HERE / "web" / "data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.DEMO = " + json.dumps(data) + ";\n", encoding="utf-8")

    print("\n=== success rate (self-mutate) ===")
    print(f"  money path (upgraded AND fixed): {len(money)}/{trials}")
    print(f"  ended correct (any path):        {len(end_correct)}/{trials}")
    print(
        f"  hero captured: swapped={hero['swapped']} first={hero['first_verdict']} "
        f"final={hero['final_verdict']} correct={hero['correct']}"
    )
    print(
        f"  contrast: low={low['correct']} high={high['correct']}  (want low=False high=True)"
    )
    print(f"wrote {out}")
    return data


def serve(port: int = 8000, host: str = "127.0.0.1") -> None:
    """Serve web/ and print the URL + the ssh -L line for remote viewing.

    Binds a FIXED port (default 8000) so an SSH tunnel can be set up ahead of
    time. ``--host 0.0.0.0`` exposes it on the LAN instead (less secure).
    """
    import functools
    import http.server
    import socket
    import socketserver
    import webbrowser

    web = HERE / "web"
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(web)
    )
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer((host, port), handler)
    except OSError as exc:
        print(f"  could not bind {host}:{port} ({exc}). Try another: --port 8081")
        return
    actual = httpd.server_address[1]
    url = f"http://localhost:{actual}/index.html"
    print(f"\n  ▶  Report served on this machine at:  {url}")
    print("\n  Viewing over SSH? Forward the port from your laptop:")
    print(
        f"      ssh -L {actual}:localhost:{actual}  {socket.gethostname()}   # or  <user>@<host>"
    )
    print(f"  then open  {url}  in your LOCAL browser.")
    print("\n  (replaying the captured run — press Ctrl-C to stop)\n")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 -- headless/remote box: just print the URL
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")


def _pick_provider() -> str:
    prompt = (
        "Run live on which provider?\n"
        "  cross     - Gemini flash-lite  →  Claude Opus   (cross-vendor self-mutation)\n"
        "  google    - Gemini only\n"
        "  anthropic - Claude only\n"
        "choice [cross]: "
    )
    try:
        choice = input(prompt).strip().lower() or "cross"
    except EOFError:
        choice = "cross"
    if choice not in CONFIGS:
        print(f"unknown provider {choice!r}; using 'cross'.")
        choice = "cross"
    return choice


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Default: replay the captured run in a local webpage. "
        "--live re-runs it for real, then serves the fresh result."
    )
    ap.add_argument(
        "--live", action="store_true", help="run the experiment for real first"
    )
    ap.add_argument(
        "--provider",
        choices=["cross", "google", "anthropic"],
        default=None,
        help="cross = Google cheap -> Anthropic strong; omit (with --live) to be asked",
    )
    ap.add_argument("--trials", type=int, default=4, help="self-mutate trials (live)")
    ap.add_argument(
        "--port", type=int, default=8000, help="web server port (fixed, for ssh -L)"
    )
    ap.add_argument(
        "--host", default="127.0.0.1", help="bind host; 0.0.0.0 exposes on the LAN"
    )
    ap.add_argument("--no-serve", action="store_true", help="skip opening the webpage")
    args = ap.parse_args()

    if args.live:
        provider = args.provider or _pick_provider()
        asyncio.run(run_live(provider, args.trials))

    data_js = HERE / "web" / "data.js"
    if not data_js.exists():
        print("no web/data.js yet — run with --live to capture one.")
        return
    if args.no_serve:
        print(f"open {HERE / 'web' / 'index.html'} in a browser to view the report.")
        return
    serve(args.port, args.host)


if __name__ == "__main__":
    main()
