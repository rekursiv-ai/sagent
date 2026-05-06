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

from typing import cast

import asyncio
import json
import os

from sagent.custom_types import Message, TextMessage, is_message
from sagent.lib.json import JSON, MutableJSON, int_val, json_freeze
from sagent.lib.message import get_directive
from sagent.lib.web.fetch import FetchError, fetch


_API_URL = "https://api.linear.app/graphql"
_DEFAULT_TIMEOUT = 30.0


_LIST_ISSUES = """
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
"""

_GET_ISSUE = """
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
"""

_CREATE_ISSUE = """
mutation CreateIssue($teamId: String!, $title: String!, $description: String) {
  issueCreate(input: { teamId: $teamId, title: $title, description: $description }) {
    success
    issue { identifier title url }
  }
}
"""

_UPDATE_ISSUE = """
mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
  issueUpdate(id: $id, input: $input) {
    success
    issue { identifier title url state { name } }
  }
}
"""

_CREATE_COMMENT = """
mutation AddComment($issueId: String!, $body: String!) {
  commentCreate(input: { issueId: $issueId, body: $body }) {
    success
    comment { id }
  }
}
"""

_TEAM_BY_KEY = """
query TeamByKey($key: String!) { teams(filter: { key: { eq: $key } }) { nodes { id } } }
"""


async def _gql(
    query: str,
    variables: MutableJSON,
    api_key: str,
) -> MutableJSON | Message:
    """Execute a GraphQL request against Linear's API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }
    try:
        raw = cast(  # pyright: ignore[reportUnnecessaryCast] -- ty can't narrow to_thread through overloads
            bytes,
            await asyncio.to_thread(
                fetch,
                url=_API_URL,
                method="POST",
                json={"query": query, "variables": variables},
                headers=headers,
                timeout_sec=_DEFAULT_TIMEOUT,
            ),
        )
    except FetchError as e:
        return TextMessage(
            f"Linear API HTTP {e.status}: {e.body[:200].decode(errors='replace')}",
            "text/x-error",
        )
    body = cast(MutableJSON, json.loads(raw))
    if errors := body.get("errors"):
        return TextMessage(f"Linear GraphQL errors: {errors}", "text/x-error")
    data = body.get("data")
    if not isinstance(data, dict):
        return TextMessage("Linear GraphQL returned no data", "text/x-error")
    return cast(MutableJSON, data)


async def _team_id(team_key: str, api_key: str) -> str | Message:
    data = await _gql(_TEAM_BY_KEY, variables={"key": team_key}, api_key=api_key)
    if is_message(data):
        return data
    teams = cast(
        list[MutableJSON],
        cast(MutableJSON, data.get("teams") or {}).get("nodes") or [],
    )
    if not teams:
        return TextMessage(f"No team with key {team_key!r}", "text/x-error")
    return str(teams[0]["id"])


_OPERATIONS = (
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
    description: str = (
        "Interact with Linear (issue tracker) via GraphQL. Operations:\n"
        "  - 'list_issues' (team?, assignee_email?, limit?) → recent issues\n"
        "  - 'get_issue' (id=ENG-42) → full issue + comments\n"
        "  - 'create_issue' (team=ENG, title, description?) → new issue\n"
        "  - 'update_issue' (id, title?, description?, state_id?)\n"
        "  - 'add_comment' (id, body)\n"
        "Requires LINEAR_API_KEY env var."
    )
    supports_microcompaction: bool = False
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

    def summary(self, msg: Message) -> str:
        """Return a short label for this Linear operation.

        Args:
          msg: Tool call message.

        Returns:
          label: "Linear <operation>:<id>".

        """
        directive = get_directive(msg)
        operation = str(directive.get("operation", ""))
        ident = str(directive.get("id", ""))
        suffix = f":{ident}" if ident else ""
        return f"Linear {operation}{suffix}"

    def prompt(self) -> str:
        """Return per-request system prompt text.

        Returns:
          prompt: Always empty for this tool.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Dispatch the requested Linear operation.

        Args:
          msg: Tool call message with ``operation`` and operation-specific fields.

        Returns:
          result: Operation result or error message.

        """
        directive = get_directive(msg)
        operation = str(directive.get("operation", ""))
        api_key = os.environ.get("LINEAR_API_KEY", "")
        if not api_key:
            return TextMessage("Linear API key not configured.", "text/x-error")
        result = await self._dispatch(
            operation=operation,
            api_key=api_key,
            issue_id=str(directive.get("id", "")),
            team=str(directive.get("team", "")),
            assignee_email=str(directive.get("assignee_email", "")),
            title=str(directive.get("title", "")),
            description=str(directive.get("description", "")),
            body=str(directive.get("body", "")),
            state_id=str(directive.get("state_id", "")),
            limit=int_val(directive.get("limit"), 25),
        )
        if is_message(result):
            return result
        return TextMessage(result, "text/plain")

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
    ) -> str | Message:
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
        return TextMessage(f"Unknown operation: {operation}", "text/x-error")

    async def _list(
        self,
        *,
        api_key: str,
        team: str,
        assignee_email: str,
        limit: int,
    ) -> str | Message:
        issue_filter: MutableJSON = {}
        if team:
            issue_filter["team"] = {"key": {"eq": team}}
        if assignee_email:
            issue_filter["assignee"] = {"email": {"eq": assignee_email}}
        data = await _gql(
            _LIST_ISSUES,
            variables={
                "filter": issue_filter or None,
                "first": max(1, min(100, limit)),
            },
            api_key=api_key,
        )
        if is_message(data):
            return data
        issues = cast(
            list[MutableJSON],
            cast(MutableJSON, data.get("issues") or {}).get("nodes") or [],
        )
        if not issues:
            return "(no issues)"
        lines = [_summarize(i) for i in issues]
        return "\n".join(lines)

    async def _get(self, *, api_key: str, issue_id: str) -> str | Message:
        if not issue_id:
            return TextMessage("'id' required.", "text/x-error")
        data = await _gql(_GET_ISSUE, variables={"id": issue_id}, api_key=api_key)
        if is_message(data):
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
    ) -> str | Message:
        if not team or not title:
            return TextMessage("'team' and 'title' required.", "text/x-error")
        team_id = await _team_id(team, api_key=api_key)
        if is_message(team_id):
            return team_id
        data = await _gql(
            _CREATE_ISSUE,
            variables={
                "teamId": team_id,
                "title": title,
                "description": description or None,
            },
            api_key=api_key,
        )
        if is_message(data):
            return data
        res = cast(MutableJSON, data.get("issueCreate") or {})
        if not res.get("success"):
            return f"Create failed: {res}"
        issue = cast(MutableJSON, res.get("issue") or {})
        return f"Created {issue.get('identifier')}: {issue.get('title')} - {issue.get('url')}"

    async def _update(
        self,
        *,
        api_key: str,
        issue_id: str,
        title: str,
        description: str,
        state_id: str,
    ) -> str | Message:
        if not issue_id:
            return TextMessage("'id' required.", "text/x-error")
        update_input: MutableJSON = {}
        if title:
            update_input["title"] = title
        if description:
            update_input["description"] = description
        if state_id:
            update_input["stateId"] = state_id
        if not update_input:
            return TextMessage("no fields to update.", "text/x-error")
        data = await _gql(
            _UPDATE_ISSUE,
            variables={"id": issue_id, "input": update_input},
            api_key=api_key,
        )
        if is_message(data):
            return data
        res = cast(MutableJSON, data.get("issueUpdate") or {})
        if not res.get("success"):
            return TextMessage(f"Update failed: {res}", "text/x-error")
        issue = cast(MutableJSON, res.get("issue") or {})
        return f"Updated {issue.get('identifier')}: {issue.get('title')}"

    async def _comment(
        self,
        *,
        api_key: str,
        issue_id: str,
        body: str,
    ) -> str | Message:
        if not issue_id or not body:
            return TextMessage("'id' and 'body' required.", "text/x-error")
        data = await _gql(
            _CREATE_COMMENT,
            variables={"issueId": issue_id, "body": body},
            api_key=api_key,
        )
        if is_message(data):
            return data
        res = cast(MutableJSON, data.get("commentCreate") or {})
        if not res.get("success"):
            return TextMessage(f"Comment failed: {res}", "text/x-error")
        return "Comment added."


def _summarize(issue: MutableJSON) -> str:
    ident = issue.get("identifier") or "?"
    state = cast(MutableJSON, issue.get("state") or {}).get("name") or "?"
    title = issue.get("title") or ""
    return f"[{ident}] [{state}] {title}"


def _render_issue(issue: MutableJSON) -> str:
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
