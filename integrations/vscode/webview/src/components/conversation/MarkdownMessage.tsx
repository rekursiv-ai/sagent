import {
  Children,
  isValidElement,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { Button } from "../ui/Button";
import { Icon } from "../ui/Icon";

export type MarkdownMessageProps = {
  content: string;
};

const markdownComponents: Components = {
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
    return <MarkdownCodeBlock>{children}</MarkdownCodeBlock>;
  },
};

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
export function MarkdownMessage({ content }: MarkdownMessageProps): ReactElement {
  return (
    <div className="markdown-message">
      <ReactMarkdown
        components={markdownComponents}
        skipHtml
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function MarkdownCodeBlock({ children }: { children?: ReactNode }): ReactElement {
  const [copied, setCopied] = useState(false);
  const codeElement = Children.toArray(children).find(isCodeElement);
  const className = codeElement?.props.className;
  const language = getLanguage(className);
  const code = codeElement ? getTextContent(codeElement.props.children) : "";

  async function copyCode(): Promise<void> {
    try {
      await copyText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="markdown-code-block copy-hover-target">
      <div className="markdown-code-block-header">
        <span>{language}</span>
        <Button
          aria-label={copied ? "Code copied" : "Copy code"}
          className="markdown-code-copy-button copy-hover-action"
          onClick={() => void copyCode()}
          title={copied ? "Copied" : "Copy code"}
          variant="ghost"
        >
          <Icon name="copy" />
        </Button>
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

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
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
