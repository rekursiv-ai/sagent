Run a tool on another machine, as that machine's sagent would run it there: its files, its shell, its network.

Endpoints are the machines this session is connected to; the reminder says which are up.

- `RemoteTool()` -- each endpoint, whether it is up, and the tools it offers.
- `RemoteTool(endpoint="gpu1", tool="Bash")` -- that tool's description and schema there.
- `RemoteTool(endpoint="gpu1", tool="Bash", arguments={...})` -- run it; the result is the tool's own.

A tool listed as the same as here takes the arguments you already know. Ask for the schema of any other before calling it.

Each agent keeps its own working directory and read files on each endpoint, as here. Paths name files on that machine, never on this one.
