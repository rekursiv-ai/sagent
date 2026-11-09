import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

const projectRoot = resolve(import.meta.dirname, "..");
const source = join(projectRoot, "webview", "src", "assets", "sagent-logo.png");
const destination = join(projectRoot, "media", "sagent-logo.png");

await mkdir(dirname(destination), { recursive: true });
await copyFile(source, destination);
console.log(`Prepared extension logo at ${destination}.`);
