import {
  Children,
  isValidElement,
  useMemo,
  type ReactElement,
  type ReactNode,
} from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { CopyButton } from "../ui/CopyButton";
import { MermaidCodeBlock } from "./MermaidCodeBlock";

export type MarkdownMessageProps = {
  content: string;
  isStreaming?: boolean;
};

function createMarkdownComponents(isStreaming: boolean): Components {
  return {
    a({ children, href, ...props }) {
      if (!isWebUrl(href)) {
        return <>{children}</>;
      }

      return (
        <a {...props} href={href}>
          {children}
        </a>
      );
    },
    pre({ children }) {
      return <MarkdownCodeBlock isStreaming={isStreaming}>{children}</MarkdownCodeBlock>;
    },
  };
}

/**
 * Render assistant-authored Markdown through the standard Markdown AST
 * pipeline used by mature agent webviews.
 *
 * Sagent owns the response text and its structured runtime events. This
 * component only turns response text into DOM; it deliberately does not
 * inspect fences, tool output, or agent status with string heuristics.
 *
 * Raw HTML stays disabled. Sagent responses can contain untrusted model output
 * and workspace-derived text, while the extension has no product requirement
 * for HTML in assistant messages. Enabling raw HTML would require a separate
 * sanitization policy and review.
 */
export function MarkdownMessage({ content, isStreaming = false }: MarkdownMessageProps): ReactElement {
  const markdownComponents = useMemo(
    () => createMarkdownComponents(isStreaming),
    [isStreaming],
  );

  return (
    <div className="markdown-message">
      <ReactMarkdown
        components={markdownComponents}
        skipHtml
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function MarkdownCodeBlock({ children, isStreaming }: { children?: ReactNode; isStreaming: boolean }): ReactElement {
  const codeElement = Children.toArray(children).find(isCodeElement);
  const className = codeElement?.props.className;
  const language = getLanguage(className);
  const code = codeElement ? getTextContent(codeElement.props.children) : "";

  if (language.toLowerCase() === "mermaid") {
    return <MermaidCodeBlock code={code} isStreaming={isStreaming} />;
  }

  return (
    <div className="markdown-code-block copy-hover-target">
      <div className="markdown-code-block-header">
        <span>{language}</span>
        <CopyButton className="markdown-code-copy-button" label="Copy code" text={code} />
      </div>
      <pre>{children}</pre>
    </div>
  );
}

function isCodeElement(value: ReactNode): value is ReactElement<{ className?: string; children?: ReactNode }> {
  return isValidElement(value) && value.type === "code";
}

function getLanguage(className: string | undefined): string {
  const match = className?.match(/(?:^|\s)language-([^\s]+)/);
  return match?.[1] ?? "text";
}

function getTextContent(value: ReactNode): string {
  return Children.toArray(value)
    .map((child) => {
      if (typeof child === "string" || typeof child === "number") {
        return String(child);
      }
      if (isValidElement<{ children?: ReactNode }>(child)) {
        return getTextContent(child.props.children);
      }
      return "";
    })
    .join("");
}

function isWebUrl(href: string | undefined): boolean {
  if (!href) {
    return false;
  }

  try {
    const url = new URL(href);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}
