import { useEffect, useRef, useState, type ReactElement } from "react";
import DOMPurify from "dompurify";
import { CopyButton } from "../ui/CopyButton";
import { Button } from "../ui/Button";
import { Icon } from "../ui/Icon";

type MermaidCodeBlockProps = {
  code: string;
  isStreaming: boolean;
};

type MermaidRenderer = typeof import("mermaid").default;

let mermaidModule: Promise<MermaidRenderer> | undefined;
let renderQueue = Promise.resolve();
let renderId = 0;

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3;
const ZOOM_STEP = 0.25;

/**
 * Render Mermaid separately from ordinary code blocks. Mermaid is loaded only
 * when a response contains a Mermaid fence, keeping the normal Markdown path
 * independent of its browser-heavy diagram runtime.
 */
export function MermaidCodeBlock({ code, isStreaming }: MermaidCodeBlockProps): ReactElement {
  const [svg, setSvg] = useState<string>();
  const [error, setError] = useState<string>();
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const diagramRef = useRef<HTMLDivElement>(null);
  const renderGeneration = useRef(0);

  useEffect(() => {
    renderGeneration.current += 1;
    const generation = renderGeneration.current;
    let cancelled = false;

    setError(undefined);

    // Incomplete fences are common during token streaming. Keep the source
    // visible until the message is complete, as Kilo does in its Markdown UI.
    if (isStreaming || !code.trim()) {
      return () => {
        cancelled = true;
      };
    }

    void renderMermaid(code).then(
      (rendered) => {
        if (!cancelled && generation === renderGeneration.current) {
          setSvg(rendered);
        }
      },
      (renderError: unknown) => {
        if (!cancelled && generation === renderGeneration.current) {
          setError(getErrorMessage(renderError));
        }
      },
    );

    return () => {
      cancelled = true;
    };
  }, [code, isStreaming]);

  // Mermaid's documented integration pattern is to render after the source is
  // complete. Keeping the previous SVG in state prevents a blank intermediate
  // render when a completed diagram is refreshed, while streaming still shows
  // the authoritative source text.
  const showDiagram = Boolean(svg) && !isStreaming && !error;

  useEffect(() => {
    if (!isFullscreen) {
      return;
    }

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        setIsFullscreen(false);
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [isFullscreen]);

  useEffect(() => {
    if (!showDiagram) {
      setIsFullscreen(false);
      setZoom(1);
    }
  }, [showDiagram]);

  useEffect(() => {
    const container = diagramRef.current;
    if (!container) {
      return;
    }

    container.replaceChildren();
    if (!showDiagram || !svg) {
      return;
    }

    const parsed = new DOMParser().parseFromString(svg, "image/svg+xml");
    if (parsed.querySelector("parsererror")) {
      setError("Mermaid returned invalid SVG.");
      return;
    }

    const nonce = getStyleNonce();
    if (!nonce) {
      setError("The webview style nonce is unavailable.");
      return;
    }

    for (const style of Array.from(parsed.querySelectorAll("style"))) {
      style.setAttribute("nonce", nonce);
    }

    // Import the parsed SVG node directly. We deliberately do not serialize it
    // back to a string: Mermaid's foreignObject labels and scoped selectors
    // must retain their original SVG/XML namespaces and casing.
    container.replaceChildren(document.importNode(parsed.documentElement, true));
  }, [showDiagram, svg]);

  useEffect(() => {
    const renderedSvg = diagramRef.current?.querySelector("svg");
    if (renderedSvg) {
      renderedSvg.style.zoom = String(isFullscreen ? zoom : 1);
    }
  }, [isFullscreen, showDiagram, svg, zoom]);

  return (
    <div
      aria-modal={isFullscreen ? true : undefined}
      className={[
        "markdown-code-block",
        "markdown-mermaid-block",
        "copy-hover-target",
        isFullscreen ? "markdown-mermaid-block-fullscreen" : "",
      ].filter(Boolean).join(" ")}
      role={isFullscreen ? "dialog" : undefined}
    >
      <div className="markdown-code-block-header">
        <span>mermaid</span>
        <div className="markdown-mermaid-controls">
          <CopyButton className="markdown-code-copy-button" label="Copy Mermaid source" text={code} />
          {showDiagram && (
            <>
              {isFullscreen && (
                <div className="markdown-mermaid-zoom-controls" aria-label="Diagram zoom controls">
                  <Button
                    aria-label="Zoom out"
                    className="markdown-code-copy-button"
                    disabled={zoom <= MIN_ZOOM}
                    onClick={() => setZoom((current) => Math.max(MIN_ZOOM, current - ZOOM_STEP))}
                    title="Zoom out"
                    variant="ghost"
                  >
                    <Icon name="zoom-out" />
                  </Button>
                  <Button
                    aria-label={`Reset zoom (${Math.round(zoom * 100)}%)`}
                    className="markdown-mermaid-zoom-level"
                    onClick={() => setZoom(1)}
                    title="Reset zoom"
                    variant="ghost"
                  >
                    {Math.round(zoom * 100)}%
                  </Button>
                  <Button
                    aria-label="Zoom in"
                    className="markdown-code-copy-button"
                    disabled={zoom >= MAX_ZOOM}
                    onClick={() => setZoom((current) => Math.min(MAX_ZOOM, current + ZOOM_STEP))}
                    title="Zoom in"
                    variant="ghost"
                  >
                    <Icon name="zoom-in" />
                  </Button>
                </div>
              )}
              <Button
                aria-label={isFullscreen ? "Close diagram viewer" : "Open diagram in full screen"}
                className="markdown-code-copy-button"
                onClick={() => setIsFullscreen((current) => !current)}
                title={isFullscreen ? "Close diagram viewer" : "Open diagram in full screen"}
                variant="ghost"
              >
                <Icon name={isFullscreen ? "screen-normal" : "screen-full"} />
              </Button>
            </>
          )}
        </div>
      </div>
      {showDiagram ? (
        <div
          aria-label="Rendered Mermaid diagram"
          className="markdown-mermaid-diagram"
          ref={diagramRef}
          role="img"
        />
      ) : (
        <pre>
          <code>{code}</code>
        </pre>
      )}
      {error && <div className="markdown-mermaid-error">Mermaid could not be rendered: {error}</div>}
    </div>
  );
}

async function renderMermaid(source: string): Promise<string> {
  await waitForFonts();
  const mermaid = await loadMermaid();
  const result = await enqueueRender(async () => {
    await mermaid.parse(source);
    return mermaid.render(`sagent-mermaid-${renderId++}`, source);
  });

  const sanitized = DOMPurify.sanitize(result.svg, {
    USE_PROFILES: { html: true, svg: true, svgFilters: true },
    // Mermaid renders many labels as HTML inside SVG foreignObject nodes.
    // These are the same integration-point allowances Mermaid uses for its
    // own SVG sanitization; allowing only the tag drops the label subtree.
    ADD_TAGS: ["foreignobject"],
    ADD_ATTR: ["dominant-baseline"],
    HTML_INTEGRATION_POINTS: { foreignobject: true },
    FORBID_TAGS: ["script"],
    FORBID_CONTENTS: ["script"],
  });
  if (!sanitized) {
    throw new Error("Mermaid returned an empty diagram.");
  }
  return sanitized;
}

function loadMermaid(): Promise<MermaidRenderer> {
  mermaidModule ??= import("mermaid").then((module) => {
    const mermaid = module.default;
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      suppressErrorRendering: true,
      theme: "base",
      themeVariables: getThemeVariables(),
    });
    return mermaid;
  });
  return mermaidModule;
}

function enqueueRender<T>(render: () => Promise<T>): Promise<T> {
  const next = renderQueue.then(render, render);
  renderQueue = next.then(
    () => undefined,
    () => undefined,
  );
  return next;
}

function getThemeVariables(): Record<string, string | boolean> {
  const themeRoot = document.body;
  const styles = getComputedStyle(themeRoot);
  const background = readColor(styles, "--vscode-editor-background", "#1e1e1e");
  const panel = readColor(styles, "--vscode-textCodeBlock-background", background);
  const text = readColor(styles, "--vscode-editor-foreground", "#d4d4d4");
  const weakText = readColor(styles, "--vscode-descriptionForeground", text);
  const border = readColor(styles, "--vscode-panel-border", weakText);
  const accent = readColor(styles, "--vscode-textLink-foreground", "#3794ff");

  return {
    darkMode: document.body.classList.contains("vscode-dark"),
    background,
    textColor: text,
    mainBkg: panel,
    primaryColor: panel,
    primaryTextColor: text,
    primaryBorderColor: border,
    lineColor: weakText,
    secondaryColor: background,
    tertiaryColor: background,
    nodeBorder: border,
    labelColor: text,
    linkColor: accent,
    edgeLabelBackground: background,
    fontFamily: styles.getPropertyValue("--vscode-editor-font-family").trim() || "sans-serif",
  };
}

function readColor(styles: CSSStyleDeclaration, name: string, fallback: string): string {
  const value = styles.getPropertyValue(name).trim();
  if (!value) {
    return fallback;
  }

  const hex = value.match(/^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    return value;
  }

  const rgb = value.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
  if (rgb) {
    return `#${[rgb[1], rgb[2], rgb[3]]
      .map((channel) => Number(channel).toString(16).padStart(2, "0"))
      .join("")}`;
  }

  return fallback;
}

/**
 * Mermaid emits its diagram stylesheet inside the returned SVG. Sagent's
 * webview CSP intentionally disallows arbitrary inline styles, so authorize
 * only Mermaid's sanitized style elements with the nonce generated for this
 * webview document.
 */
function getStyleNonce(): string | undefined {
  return document
    .querySelector<HTMLMetaElement>('meta[name="sagent-style-nonce"]')
    ?.content.trim();
}

async function waitForFonts(): Promise<void> {
  if (document.fonts) {
    await document.fonts.ready;
  }
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Invalid Mermaid syntax.";
}
