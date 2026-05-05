# Security Policy

## Why this file exists

sagent can execute tools, read files, call network providers, and handle credentials. Security reports need a private path so exploit details are not published before review.

## Reporting a vulnerability

Please report suspected security vulnerabilities privately by emailing hello@rekursiv.ai.

Include:

- Affected version or commit.
- Steps to reproduce.
- Expected impact.
- Any suggested mitigation.

Please do not open public issues for vulnerabilities until we have investigated and coordinated disclosure.

## Scope

Security reports are especially useful for:

- Tool execution boundary issues.
- Filesystem access bugs.
- Prompt or provider credential leakage.
- SSRF or unsafe URL handling.
- Dependency or packaging issues that affect installed users.
