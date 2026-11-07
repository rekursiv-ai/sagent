# Sagent for VS Code

This extension provides a Sagent-native chat surface inside VS Code.

Sagent appears in both the Activity Bar/Primary Side Bar and the Secondary Side
Bar beside Chat and other coding-agent views. Both locations use the same
provider, conversation state, and Sagent session; they are not separate
implementations. The extension does not hide or modify Chat, Codex, or other
extensions' views.

The extension host launches one `sagent vscode-server` process for the active
workspace. The process uses Sagent's existing model catalog, tools, team, runtime,
and JSONL journal. The webview is only a presentation layer; it sends requests
through the extension host and receives streamed runtime notifications.

## Development

From this directory:

```text
npm install
npm run check
npm run compile
```

### Install the current extension in VS Code

To use Sagent in a normal VS Code window instead of an Extension Development
Host, package the extension and install the resulting VSIX:

```text
npm install
npm run vsix
code --install-extension ./sagent-vscode-0.1.0.vsix
```

You can also install the VSIX from VS Code's Extensions view with `Extensions:
Install from VSIX...`. Installing the VSIX does not disable or replace Codex;
both extensions can remain installed and active together.

The installed extension launches the configured Sagent executable when its
view is opened. The default configuration expects the Sagent CLI to be
available as `sagent` and starts it with the `vscode-server` argument. The
extension does not bundle Python or the Sagent runtime.

If Sagent is not on your PATH, configure these settings in VS Code's user or
workspace settings, adapting the executable and arguments to your local
installation:

```json
{
  "sagent.command": "uv",
  "sagent.commandArguments": [
    "run",
    "--project",
    "/path/to/sagent",
    "--frozen",
    "--no-dev",
    "python",
    "-m",
    "sagent",
    "vscode-server"
  ]
}
```

After installation, reload VS Code if prompted, open a workspace folder, and
select Sagent from the Activity Bar or Secondary Sidebar. F5 is only needed
when developing the extension source.

Open the repository root as the VS Code workspace and press `F5`. The root
`.vscode` launch configuration starts an Extension Development Host with this
extension and compiles it first, while preserving the repository as Sagent's
workspace. It starts the host with the dedicated `Sagent F5 Host` VS Code
workspace file, so the development window can have distinct UI colors without
changing the normal workspace window. That F5 workspace also asks VS Code to
start the Secondary Side Bar visible. For this source worktree, the F5
workspace launches the server through the repository's standard `uv` runner:

```text
uv run --frozen --no-dev python -m sagent vscode-server
```

This keeps the development launch independent of a globally installed
`sagent` executable and makes the current worktree the imported Python source.
The first run may synchronize the repository environment. Normal installed
usage continues to use the configurable `sagent` command shown in the
extension settings.

### Identifying the F5 window

The root `.vscode/sagent-f5.code-workspace` file contains the blue
`workbench.colorCustomizations` and title. The F5 launch configuration opens
that workspace file only in the Extension Development Host. Do not copy those
colors into `.vscode/settings.json`; ordinary folder windows should keep their
normal appearance.

## Native VS Code responsibilities

- The chat view uses `WebviewViewProvider` and VS Code's webview message bridge.
- Changed files are reviewed with the built-in Git extension API and
  `vscode.diff`.
- Workspace files, editors, and source-control state remain owned by VS Code.
- Sagent remains the source of truth for the session and agent state.
