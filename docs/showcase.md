# Built with sagent

> *We used sagent to stand up a persistent, multi-agent dev team -- then used
> that team to ship and maintain a real library.*

[**agent-team**](https://github.com/blackjax-devs/agent-team) is a development
*channel* of five specialized agents -- a tech lead, a senior and a junior
engineer, a statistician, and a tech writer -- running as one persistent server
with a web UI. The agents message each other (`@swe`, `@statistician`) and hand
off work the way a human team does. It's a thin role profile on top of sagent;
sagent does the heavy lifting:

- **Persistent sessions + resume** keep each agent's full history across
  restarts, so the team picks a multi-day thread back up exactly where it left
  off.
- **Per-agent MCP servers** (`extra_mcp_servers`) inject a peer-messaging tool
  into every agent -- that's what turns five isolated CLIs into one channel.
- **Unified cost tracking + `--max-budget-usd`** roll every agent's spend into
  one number with a hard cap on the whole team.
- **Sub-agent spawning** lets the tech lead fan work out to ephemeral helpers
  and collect the results.

We dogfood it daily on the [BlackJAX](https://github.com/blackjax-devs/blackjax)
ecosystem. The team built and now maintains
[**tuningfork**](https://github.com/blackjax-devs/tuningfork), a BlackJAX-native
MCMC benchmark suite (14 models × 24 samplers × 10 warmups × 6 SMC methods): the
statistician agent verifies algorithm-to-paper correctness and tunes sampler
parameters, the engineers implement and run the benchmark loop, and the tech
lead coordinates hand-offs and gates merges -- all over sagent's channel.

*Built something with sagent? Open a PR adding it here.*
