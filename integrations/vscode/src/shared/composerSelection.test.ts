import { strict as assert } from "node:assert";
import test from "node:test";

import { formatComposerSelection } from "./composerSelection.js";

test("formats editor selections with an editable source location", () => {
  assert.equal(
    formatComposerSelection({
      source: "editor",
      text: "const answer = 42;",
      filePath: "src/example.ts",
      languageId: "typescript",
      startLine: 4,
      endLine: 4,
    }),
    "\n\nSelected text from src/example.ts:4:\n```typescript\nconst answer = 42;\n```\n",
  );
});

test("formats chat selections as quoted editable text", () => {
  assert.equal(
    formatComposerSelection({
      source: "chat",
      text: "first line\nsecond line",
    }),
    "\n\nSelected text:\n> first line\n> second line\n",
  );
});
