/// <reference types="node" />

import { spawn } from "node:child_process";

export class SagentInstallError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "SagentInstallError";
  }
}

/** Install the native Sagent CLI through the host's uv installation. */
export async function installSagent(cwd: string): Promise<void> {
  await runCommand("uv", ["tool", "install", "sagent"], cwd);
}

async function runCommand(command: string, args: string[], cwd: string): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      shell: false,
      stdio: ["ignore", "ignore", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.on("error", (error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") {
        reject(new SagentInstallError(
          "Sagent could not be installed automatically because uv is not installed "
          + "on this VS Code host. Install uv, then reopen Sagent.",
        ));
      } else {
        reject(new SagentInstallError(`Sagent installation could not start: ${error.message}`));
      }
    });
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      const detail = stderr.trim();
      reject(new SagentInstallError(
        `Sagent installation failed (${code ?? `signal ${signal ?? "unknown"}`})`
        + (detail ? `: ${detail}` : "."),
      ));
    });
  });
}
