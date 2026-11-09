// The extension host runs on Node; keep the built-in process types explicit
// because the webview TypeScript project also type-checks shared source files.
/// <reference types="node" />

import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { join } from "node:path";
import * as vscode from "vscode";
import { installSagent } from "./sagentInstaller";

import {
  parseSagentAgentSessionState,
  parseSagentCommandList,
  parseModelSettingsSnapshot,
  parseSagentSkillList,
  parseSagentSessionList,
  parseSagentSessionState,
  parseSagentTeamState,
  type SagentAgentSessionState,
  type SagentCommandList,
  type SagentEvent,
  type SagentSessionList,
  type SagentSessionState,
  type SagentTeamState,
  type SagentSkillInfo,
} from "../shared/sagentProtocol";
import type {
  ModelSettingsAction,
  ModelSettingsScope,
  ModelSettingsSnapshot,
} from "../shared/modelSettings/modelSettingsContract";

const defaultVscodeModel = "google-cli/gemini-flash-3.8";
const defaultTrackinizerUrl = "http://127.0.0.1:8765";

export const REKURSIV_ORIENTATION_PROMPT = `You are working in the Rekursiv AI stack.

Sagent coordinates agents and tools. Trackinizer preserves work, evidence,
findings, beliefs, and agent sessions. Priml runs reusable experiments and
evaluations. Configgle provides reproducible experiment configuration.
Wesearch supports web and paper research.

In Rekursiv mode, use the Skill tool to discover and load the authoritative
workflow before acting in one of those domains. Do not recreate any Rekursiv
system in the extension or in a custom tool. Preserve useful failures,
provenance, evidence, and handoffs when the native skills call for them.

Load the Trackinizer/trax router before durable work, records, evidence,
campaigns, or sessions. Load Priml before Priml experiments or evaluations,
Configgle before experiment configuration work, and Wesearch before web or
paper research. Read workspace AGENTS.md files and challenge-specific skills
before changing or running the project.`;

type SagentMessage = {
  id?: number;
  method?: string;
  params?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: { code: string; message: string };
};

type PendingRequest = {
  resolve: (result: Record<string, unknown>) => void;
  reject: (error: Error) => void;
};

export class SagentExecutableMissingError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "SagentExecutableMissingError";
  }
}

export class SagentClient implements vscode.Disposable {
  private process: ChildProcessWithoutNullStreams | undefined;
  private disposing = false;
  private lineBuffer = "";
  private nextRequestId = 1;
  private readonly pending = new Map<number, PendingRequest>();
  private readonly events = new EventEmitter();
  private launchOverride: { command: string; arguments: string[] } | undefined;

  public constructor(private readonly extensionUri: vscode.Uri) {}

  public onEvent(listener: (event: SagentEvent) => void): vscode.Disposable {
    this.events.on("event", listener);
    return new vscode.Disposable(() => this.events.off("event", listener));
  }

  public isRunning(): boolean {
    return this.process !== undefined;
  }

  public canInstallNativeSagent(): boolean {
    const configuration = vscode.workspace.getConfiguration("sagent");
    const command = configuration.get<string>("command", "sagent");
    const argumentsValue = configuration.get<unknown[]>("commandArguments", ["daemon"]);
    const commandArguments = argumentsValue.filter(
      (value): value is string => typeof value === "string",
    );
    return command === "sagent"
      && commandArguments.length === 1
      && commandArguments[0] === "daemon";
  }

  public async installNativeSagent(cwd: string): Promise<void> {
    await installSagent(cwd);
    // uv may install its executable directory after this extension host's PATH
    // was created. Run through uv for this process, without changing settings.
    this.launchOverride = { command: "uv", arguments: ["tool", "run", "sagent"] };
  }

  public start(cwd: string): void {
    if (this.process) {
      return;
    }
    const configuration = vscode.workspace.getConfiguration("sagent");
    const configuredCommand = configuration.get<string>("command", "sagent");
    const argumentsValue = configuration.get<unknown[]>("commandArguments", [
      "daemon",
    ]);
    const configuredArguments = argumentsValue.filter(
      (value): value is string => typeof value === "string",
    );
    const command = this.launchOverride?.command ?? configuredCommand;
    const commandArguments = this.launchOverride?.arguments ?? configuredArguments;
    const model = configuration.get<string>("model", defaultVscodeModel);
    const allowedProviders = configuration.get<string>("allowedProviders", "");
    const tools = configuration.get<unknown[]>("tools", []).filter(
      (value): value is string => typeof value === "string",
    );
    const mode = configuration.get<string>("mode", "standard");
    const skillDirectories = configuration.get<unknown[]>("rekursivSkillDirectories", []).filter(
      (value): value is string => typeof value === "string" && value.length > 0,
    );
    const selectedSkills = configuration.get<unknown[]>("rekursivSkills", []).filter(
      (value): value is string => typeof value === "string" && value.length > 0,
    );
    const trackinizerUrl = configuration.get<string>("trackinizerUrl", "") || defaultTrackinizerUrl;
    const bundledSkillDirectories = this.bundledSkillDirectories(
      selectedSkills.length > 0 ? selectedSkills : ["trackinizer", "priml", "configgle"],
    );
    const configuredSkillDirectories = mode === "rekursiv"
      ? [...skillDirectories, ...bundledSkillDirectories]
      : [];

    const serverArguments = [
      ...commandArguments,
      "--model",
      model,
      ...(allowedProviders ? ["--allow-providers", allowedProviders] : []),
      ...(tools.length > 0 ? ["--tools", ...tools] : []),
      ...(mode === "rekursiv" ? ["--append-system-prompt", REKURSIV_ORIENTATION_PROMPT] : []),
      ...configuredSkillDirectories.flatMap((directory) => ["--skill-dir", directory]),
    ];
    const child = spawn(command, serverArguments, {
      cwd,
      shell: false,
      env: {
        ...process.env,
        TRACKINIZER_URL: trackinizerUrl,
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.disposing = false;
    this.process = child;
    this.lineBuffer = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      // A reconnect can start a new child before the old process has fully
      // exited. Ignore late bytes from that old process rather than mixing two
      // JSONL streams in one request/response buffer.
      if (this.process === child) {
        this.readOutput(chunk);
      }
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      console.error(`[sagent] ${chunk.trimEnd()}`);
    });
    child.on("error", (error) => {
      if (this.process !== child) {
        return;
      }
      this.process = undefined;
      this.lineBuffer = "";
      const launchError = formatLaunchError(error, command, cwd);
      this.failPending(launchError);
      if (!this.disposing) {
        this.events.emit("event", { type: "connection.error", message: launchError.message });
      }
    });
    child.on("exit", (code, signal) => {
      if (this.process !== child) {
        return;
      }
      this.process = undefined;
      this.lineBuffer = "";
      this.failPending(
        new Error(`Sagent server exited (${code ?? "unknown"}${signal ? `, ${signal}` : ""})`),
      );
      if (!this.disposing) {
        this.events.emit("event", {
          type: "connection.error",
          message: `Sagent server exited (${code ?? "unknown"}${signal ? `, ${signal}` : ""})`,
        });
      }
    });
  }

  public async initialize(): Promise<Record<string, unknown>> {
    return this.request("initialize", {});
  }

  public async openSession(sessionPath?: string): Promise<SagentSessionState> {
    return parseSagentSessionState(
      await this.request("openSession", sessionPath ? { sessionPath } : {}),
    );
  }

  public async listSessions(): Promise<SagentSessionList> {
    return parseSagentSessionList(await this.request("listSessions", {}));
  }

  public async getTeam(): Promise<SagentTeamState> {
    return parseSagentTeamState(await this.request("getTeam", {}));
  }

  public async getSkills(): Promise<SagentSkillInfo[]> {
    return parseSagentSkillList(await this.request("getSkills", {}));
  }

  private bundledSkillDirectories(skillIds: string[]): string[] {
    const root = join(this.extensionUri.fsPath, "resources", "rekursiv-skills");
    return skillIds
      .map((skillId) => join(root, skillId))
      .filter((directory) => existsSync(directory));
  }

  public async getAgentSession(label: string): Promise<SagentAgentSessionState> {
    return parseSagentAgentSessionState(
      await this.request("getAgentSession", { label }),
    );
  }

  public async getCommands(agentLabel: string): Promise<SagentCommandList> {
    return parseSagentCommandList(
      await this.request("getCommands", { agentLabel }),
    );
  }

  public async executeCommand(
    agentLabel: string,
    line: string,
  ): Promise<Record<string, unknown>> {
    return this.request("executeCommand", { agentLabel, line });
  }

  public async getModelSettings(agentLabel: string): Promise<ModelSettingsSnapshot> {
    return parseModelSettingsSnapshot(
      await this.request("getModelSettings", { agentLabel }),
    );
  }

  public async setModelSettings(
    agentLabel: string,
    scope: ModelSettingsScope,
    action: ModelSettingsAction,
  ): Promise<ModelSettingsSnapshot> {
    return parseModelSettingsSnapshot(
      await this.request("setModelSettings", { agentLabel, scope, action }),
    );
  }

  public async sendMessage(
    text: string,
    agentLabel: string,
  ): Promise<Record<string, unknown>> {
    return this.request("sendMessage", { text, agentLabel });
  }

  public async cancelTurn(): Promise<Record<string, unknown>> {
    return this.request("cancelTurn", {});
  }

  public dispose(): void {
    this.disposing = true;
    const child = this.process;
    this.process = undefined;
    if (!child) {
      return;
    }
    child.stdin.end(JSON.stringify({ id: this.nextRequestId++, method: "shutdown" }) + "\n");
    const killTimer = setTimeout(() => child.kill(), 2_000);
    child.once("exit", () => clearTimeout(killTimer));
    this.failPending(new Error("Sagent client disposed"));
  }

  private request(method: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    const child = this.process;
    if (!child) {
      return Promise.reject(new Error("Sagent server is not running"));
    }
    const id = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try {
        child.stdin.write(JSON.stringify({ id, method, params }) + "\n");
      } catch (error) {
        this.pending.delete(id);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  private readOutput(chunk: string): void {
    this.lineBuffer += chunk;
    while (true) {
      const newline = this.lineBuffer.indexOf("\n");
      if (newline < 0) {
        return;
      }
      const line = this.lineBuffer.slice(0, newline).trim();
      this.lineBuffer = this.lineBuffer.slice(newline + 1);
      if (!line) {
        continue;
      }
      try {
        this.handleMessage(JSON.parse(line) as SagentMessage);
      } catch (error) {
        console.error("[sagent] invalid server message", error);
      }
    }
  }

  private handleMessage(message: SagentMessage): void {
    if (message.method === "sagent/event" && message.params) {
      const type = message.params.type;
      if (typeof type === "string") {
        this.events.emit("event", { ...message.params, type } satisfies SagentEvent);
      }
      return;
    }
    if (message.id === undefined) {
      return;
    }
    const pending = this.pending.get(message.id);
    if (!pending) {
      return;
    }
    this.pending.delete(message.id);
    if (message.error) {
      pending.reject(new Error(message.error.message));
    } else {
      pending.resolve(message.result ?? {});
    }
  }

  private failPending(error: Error): void {
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  }
}

function formatLaunchError(error: Error & { code?: string }, command: string, cwd: string): Error {
  if (error.code === "ENOENT") {
    return new SagentExecutableMissingError(
      `Sagent is not installed or is not available on this VS Code host's PATH. `
      + `The extension tried to run '${command}' from '${cwd}'. `
      + "Install Sagent natively for this host, then reload VS Code. "
      + "The extension does not cross between Windows and WSL automatically.",
    );
  }
  if (error.code === "EACCES") {
    return new Error(
      `Sagent could not be executed from '${command}' on this VS Code host `
      + `(permission denied). Check the native Sagent installation and PATH.`,
    );
  }
  return error;
}
