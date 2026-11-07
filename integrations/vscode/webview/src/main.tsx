import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@vscode/codicons/dist/codicon.css";

import { App } from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
