"""Linear integration: issue tracking via GraphQL.

Linear was picked as the default because it's the most scrum-friendly
of the popular OSS-typical trackers (cycles, issue states, parent/child
relationships, teams) with a clean GraphQL API and a personal API key
auth model - no OAuth dance needed for a CLI.

Auth: ``LINEAR_API_KEY`` env var; create at
https://linear.app/settings/api, key starts with ``lin_api_``).

Supported operations:

- ``list_issues`` - list issues, filterable by team key or assignee
- ``get_issue`` - fetch one issue by identifier (e.g. ``ENG-42``)
- ``create_issue`` - create a new issue
- ``update_issue`` - update title / description / state
- ``add_comment`` - add a comment to an issue
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast

import asyncio
import json
import os

from wesearch.errors import FetchError
from wesearch.fetch import Content, RequestParams, Retry, fetch

from sagent.lib.custom_json import JSON, MutableJSON, int_val, json_freeze
from sagent.tools.core import load_tool_description
from sagent.types.runtime import ToolResult


async def _gql(
    query: str,
    variables: MutableJSON,
    api_key: str,
    *,
    timeout_sec: float = 30.0,
) -> MutableJSON | ToolResult:
    """Execute a GraphQL request against Linear's API.

    Args:
      query: GraphQL query / mutation text.
      variables: Variable bindings for the query.
      api_key: Linear personal API key (``lin_api_...``).
      timeout_sec: Per-request HTTP timeout.

    Returns:
      data: Parsed ``data`` block on success, or a ``ToolResult`` error.

    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }
    try:
        raw = await asyncio.to_thread(
            fetch,
            url="https://api.linear.app/graphql",
            request=RequestParams(
                content=Content(
                    method="POST",
                    json={"query": query, "variables": variables},
                    headers=headers,
                ),
                retry=Retry(timeout_sec=timeout_sec),
            ),
        )
    except FetchError as e:
        return ToolResult(
            call_id="",
            content=(
                f"Linear API HTTP {e.status}: {e.body[:200].decode(errors='replace')}"
            ),
            is_error=True,
        )
    body = cast(MutableJSON, json.loads(raw[0]))
    if errors := body.get("errors"):
        return ToolResult(
            call_id="",
            content=f"Linear GraphQL errors: {errors}",
            is_error=True,
        )
    data = body.get("data")
    if not isinstance(data, dict):
        return ToolResult(
            call_id="",
            content="Linear GraphQL returned no data",
            is_error=True,
        )
    return cast(MutableJSON, data)


async def _team_id(team_key: str, api_key: str) -> str | ToolResult:
    """Look up the opaque team id for a Linear team key (e.g. ``ENG``)."""
    data = await _gql(
        """
query TeamByKey($key: String!) { teams(filter: { key: { eq: $key } }) { nodes { id } } }
""",
        variables={"key": team_key},
        api_key=api_key,
    )
    if isinstance(data, ToolResult):
        return data
    teams = cast(
        list[MutableJSON],
        cast(MutableJSON, data.get("teams") or {}).get("nodes") or [],
    )
    if not teams:
        return ToolResult(
            call_id="",
            content=f"No team with key {team_key!r}",
            is_error=True,
        )
    return str(teams[0]["id"])


_OPERATIONS: Final = (
    "list_issues",
    "get_issue",
    "create_issue",
    "update_issue",
    "add_comment",
)


class Linear:
    """Tool: Linear issue tracking via GraphQL.

    Requires ``LINEAR_API_KEY`` env var.
    """

    name: str = "Linear"
    tool_id: str = "application/x-tool-linear"
    clearable_results: bool = False
    description: str = load_tool_description("Linear")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": list(_OPERATIONS)},
                "id": {
                    "type": "string",
                    "description": "Issue identifier, e.g. ENG-42.",
                },
                "team": {"type": "string", "description": "Team key, e.g. ENG."},
                "assignee_email": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "body": {"type": "string"},
                "state_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["operation"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation.

        Args:
          args: Tool arguments.

        Returns:
          label: Human-readable summary string.

        """
        operation = str(args.get("operation", ""))
        ident = str(args.get("id", ""))
        suffix = f":{ident}" if ident else ""
        return f"Linear {operation}{suffix}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for Linear.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return supplemental system-prompt text.

        Returns:
          prompt: Empty string; this tool adds no prompt.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network call, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Dispatch the requested Linear operation.

        Args:
          args: Tool arguments containing ``operation`` and op-specific fields.

        Returns:
          result: Operation output or an error message.

        """
        operation = str(args.get("operation", ""))
        api_key = os.environ.get("LINEAR_API_KEY", "")
        if not api_key:
            return ToolResult(
                call_id="",
                content="Linear API key not configured.",
                is_error=True,
            )
        result = await self._dispatch(
            operation=operation,
            api_key=api_key,
            issue_id=str(args.get("id", "")),
            team=str(args.get("team", "")),
            assignee_email=str(args.get("assignee_email", "")),
            title=str(args.get("title", "")),
            description=str(args.get("description", "")),
            body=str(args.get("body", "")),
            state_id=str(args.get("state_id", "")),
            limit=int_val(args.get("limit"), 25),
        )
        if isinstance(result, ToolResult):
            return result
        return ToolResult(call_id="", content=result)

    async def _dispatch(
        self,
        *,
        operation: str,
        api_key: str,
        issue_id: str,
        team: str,
        assignee_email: str,
        title: str,
        description: str,
        body: str,
        state_id: str,
        limit: int,
    ) -> str | ToolResult:
        """Route ``operation`` to the matching private helper."""
        if operation == "list_issues":
            return await self._list(
                api_key=api_key,
                team=team,
                assignee_email=assignee_email,
                limit=limit,
            )
        if operation == "get_issue":
            return await self._get(api_key=api_key, issue_id=issue_id)
        if operation == "create_issue":
            return await self._create(
                api_key=api_key,
                team=team,
                title=title,
                description=description,
            )
        if operation == "update_issue":
            return await self._update(
                api_key=api_key,
                issue_id=issue_id,
                title=title,
                description=description,
                state_id=state_id,
            )
        if operation == "add_comment":
            return await self._comment(
                api_key=api_key,
                issue_id=issue_id,
                body=body,
            )
        return ToolResult(
            call_id="",
            content=f"Unknown operation: {operation}",
            is_error=True,
        )

    async def _list(
        self,
        *,
        api_key: str,
        team: str,
        assignee_email: str,
        limit: int,
    ) -> str | ToolResult:
        """Run ``ListIssues`` with optional team / assignee filters."""
        issue_filter: MutableJSON = {}
        if team:
            issue_filter["team"] = {"key": {"eq": team}}
        if assignee_email:
            issue_filter["assignee"] = {"email": {"eq": assignee_email}}
        data = await _gql(
            """
query ListIssues($filter: IssueFilter, $first: Int!) {
  issues(first: $first, filter: $filter) {
    nodes {
      identifier
      title
      state { name }
      assignee { name email }
      priority
      url
      updatedAt
    }
  }
}
""",
            variables={
                "filter": issue_filter or None,
                "first": max(1, min(100, limit)),
            },
            api_key=api_key,
        )
        if isinstance(data, ToolResult):
            return data
        issues = cast(
            list[MutableJSON],
            cast(MutableJSON, data.get("issues") or {}).get("nodes") or [],
        )
        if not issues:
            return "(no issues)"
        lines = [_summarize(i) for i in issues]
        return "\n".join(lines)

    async def _get(self, *, api_key: str, issue_id: str) -> str | ToolResult:
        """Fetch one issue via ``GetIssue`` and render it."""
        if not issue_id:
            return ToolResult(call_id="", content="'id' required.", is_error=True)
        data = await _gql(
            """
query GetIssue($id: String!) {
  issue(id: $id) {
    identifier
    title
    description
    state { name }
    assignee { name email }
    priority
    url
    createdAt
    updatedAt
    comments(first: 20) { nodes { body user { name } createdAt } }
  }
}
""",
            variables={"id": issue_id},
            api_key=api_key,
        )
        if isinstance(data, ToolResult):
            return data
        issue = cast(MutableJSON | None, data.get("issue"))
        if not issue:
            return f"No such issue: {issue_id}"
        return _render_issue(issue)

    async def _create(
        self,
        *,
        api_key: str,
        team: str,
        title: str,
        description: str,
    ) -> str | ToolResult:
        """Run ``CreateIssue`` against the team resolved from ``team`` key."""
        if not team or not title:
            return ToolResult(
                call_id="",
                content="'team' and 'title' required.",
                is_error=True,
            )
        team_id = await _team_id(team, api_key=api_key)
        if isinstance(team_id, ToolResult):
            return team_id
        data = await _gql(
            """
mutation CreateIssue($teamId: String!, $title: String!, $description: String) {
  issueCreate(input: { teamId: $teamId, title: $title, description: $description }) {
    success
    issue { identifier title url }
  }
}
""",
            variables={
                "teamId": team_id,
                "title": title,
                "description": description or None,
            },
            api_key=api_key,
        )
        if isinstance(data, ToolResult):
            return data
        res = cast(MutableJSON, data.get("issueCreate") or {})
        if not res.get("success"):
            return f"Create failed: {res}"
        issue = cast(MutableJSON, res.get("issue") or {})
        return (
            f"Created {issue.get('identifier')}: {issue.get('title')}"
            f" - {issue.get('url')}"
        )

    async def _update(
        self,
        *,
        api_key: str,
        issue_id: str,
        title: str,
        description: str,
        state_id: str,
    ) -> str | ToolResult:
        """Run ``UpdateIssue`` for the fields the caller supplied."""
        if not issue_id:
            return ToolResult(call_id="", content="'id' required.", is_error=True)
        update_input: MutableJSON = {}
        if title:
            update_input["title"] = title
        if description:
            update_input["description"] = description
        if state_id:
            update_input["stateId"] = state_id
        if not update_input:
            return ToolResult(
                call_id="",
                content="no fields to update.",
                is_error=True,
            )
        data = await _gql(
            """
mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
  issueUpdate(id: $id, input: $input) {
    success
    issue { identifier title url state { name } }
  }
}
""",
            variables={"id": issue_id, "input": update_input},
            api_key=api_key,
        )
        if isinstance(data, ToolResult):
            return data
        res = cast(MutableJSON, data.get("issueUpdate") or {})
        if not res.get("success"):
            return ToolResult(
                call_id="",
                content=f"Update failed: {res}",
                is_error=True,
            )
        issue = cast(MutableJSON, res.get("issue") or {})
        return f"Updated {issue.get('identifier')}: {issue.get('title')}"

    async def _comment(
        self,
        *,
        api_key: str,
        issue_id: str,
        body: str,
    ) -> str | ToolResult:
        """Run ``AddComment`` on the given issue."""
        if not issue_id or not body:
            return ToolResult(
                call_id="",
                content="'id' and 'body' required.",
                is_error=True,
            )
        data = await _gql(
            """
mutation AddComment($issueId: String!, $body: String!) {
  commentCreate(input: { issueId: $issueId, body: $body }) {
    success
    comment { id }
  }
}
""",
            variables={"issueId": issue_id, "body": body},
            api_key=api_key,
        )
        if isinstance(data, ToolResult):
            return data
        res = cast(MutableJSON, data.get("commentCreate") or {})
        if not res.get("success"):
            return ToolResult(
                call_id="",
                content=f"Comment failed: {res}",
                is_error=True,
            )
        return "Comment added."


def _summarize(issue: MutableJSON) -> str:
    """One-line summary of an issue: ``[ident] [state] title``."""
    ident = issue.get("identifier") or "?"
    state = cast(MutableJSON, issue.get("state") or {}).get("name") or "?"
    title = issue.get("title") or ""
    return f"[{ident}] [{state}] {title}"


def _render_issue(issue: MutableJSON) -> str:
    """Multi-line markdown rendering for a single ``GetIssue`` payload."""
    parts = [
        f"# [{issue.get('identifier')}] {issue.get('title')}",
        f"State: {cast('MutableJSON', issue.get('state') or {}).get('name') or '?'}",
        f"URL: {issue.get('url')}",
        f"Priority: {issue.get('priority')}",
    ]
    assignee = cast(MutableJSON | None, issue.get("assignee"))
    if assignee:
        parts.append(f"Assignee: {assignee.get('name')} ({assignee.get('email')})")
    desc = issue.get("description")
    if desc:
        parts.append("")
        parts.append(str(desc))
    comments = cast(
        list[MutableJSON],
        cast(MutableJSON, issue.get("comments") or {}).get("nodes") or [],
    )
    if comments:
        parts.append("\n## Comments")
        for c in comments:
            user = cast(MutableJSON, c.get("user") or {}).get("name") or "?"
            parts.append(f"- {user} @ {c.get('createdAt')}: {c.get('body')}")
    return "\n".join(parts)
