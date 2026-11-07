// The extension host runs on Node; keep the built-in process types explicit
// because the webview TypeScript project also type-checks shared source files.
/// <reference types="node" />

import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import * as vscode from "vscode";

import {
  parseSagentAgentSessionState,
  parseSagentCommandList,
  parseModelSettingsSnapshot,
  parseSagentSessionList,
  parseSagentSessionState,
  parseSagentTeamState,
  type SagentAgentSessionState,
  type SagentCommandList,
  type SagentEvent,
  type SagentSessionList,
  type SagentSessionState,
  type SagentTeamState,
} from "../shared/sagentProtocol";
import type {
  ModelSettingsAction,
  ModelSettingsScope,
  ModelSettingsSnapshot,
} from "../shared/modelSettings/modelSettingsContract";

const defaultVscodeModel = "google-cli/gemini-flash-3.8";

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

export class SagentClient implements vscode.Disposable {
  private process: ChildProcessWithoutNullStreams | undefined;
  private disposing = false;
  private lineBuffer = "";
  private nextRequestId = 1;
  private readonly pending = new Map<number, PendingRequest>();
  private readonly events = new EventEmitter();

  public onEvent(listener: (event: SagentEvent) => void): vscode.Disposable {
    this.events.on("event", listener);
    return new vscode.Disposable(() => this.events.off("event", listener));
  }

  public isRunning(): boolean {
    return this.process !== undefined;
  }

  public start(cwd: string): void {
    if (this.process) {
      return;
    }
    const configuration = vscode.workspace.getConfiguration("sagent");
    const command = configuration.get<string>("command", "sagent");
    const argumentsValue = configuration.get<unknown[]>("commandArguments", [
      "vscode-server",
    ]);
    const commandArguments = argumentsValue.filter(
      (value): value is string => typeof value === "string",
    );
    const model = configuration.get<string>("model", defaultVscodeModel);
    const allowedProviders = configuration.get<string>("allowedProviders", "");
    const tools = configuration.get<unknown[]>("tools", []).filter(
      (value): value is string => typeof value === "string",
    );

    const serverArguments = [
      ...commandArguments,
      "--model",
      model,
      ...(allowedProviders ? ["--allow-providers", allowedProviders] : []),
      ...(tools.length > 0 ? ["--tools", ...tools] : []),
    ];
    const child = spawn(command, serverArguments, {
      cwd,
      shell: false,
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
      this.failPending(error);
      if (!this.disposing) {
        this.events.emit("event", { type: "connection.error", message: error.message });
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
