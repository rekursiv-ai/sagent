import { resolve } from "node:path";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  // React's published browser bundle still uses this Node-style build flag.
  // The webview is a browser runtime, so resolve it at build time instead of
  // shipping a fake `process` global or a development React bundle.
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  // Tailwind is presentation-only. Sagent state and behavior continue to cross
  // the extension boundary through the existing typed message contracts.
  plugins: [react(), tailwindcss()],
  build: {
    emptyOutDir: true,
    lib: {
      cssFileName: "sagent-webview",
      entry: resolve(__dirname, "webview/src/main.tsx"),
      fileName: () => "sagent-webview.js",
      formats: ["iife"],
      name: "SagentWebview",
    },
    outDir: resolve(__dirname, "media/webview"),
    rollupOptions: {
      output: {
        assetFileNames: "[name][extname]",
      },
    },
  },
});
