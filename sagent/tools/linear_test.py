"""Tests for ``tools.linear``: Linear issue tracker via GraphQL."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import asyncio
import json

from wesearch.errors import FetchError
from wesearch.fetch import FetchSession

from sagent.tools.linear import Linear
from sagent.types.runtime import ToolResult


if TYPE_CHECKING:
    import pytest


def _gql_response(data: object) -> bytes:
    return json.dumps({"data": data}).encode()


def test_linear_metadata() -> None:
    t = Linear()
    assert t.name == "Linear"
    assert t.tool_id == "application/x-tool-linear"


def test_summary_with_id() -> None:
    t = Linear()
    assert t.summary({"operation": "get_issue", "id": "ENG-42"}) == (
        "Linear get_issue:ENG-42"
    )


def test_summary_without_id() -> None:
    t = Linear()
    assert t.summary({"operation": "list_issues"}) == "Linear list_issues"


def test_summary_result_returns_none() -> None:
    t = Linear()
    assert t.summary_result(ToolResult(call_id="", content="x")) is None


def test_prompt_empty() -> None:
    t = Linear()
    assert t.prompt() == ""


def test_run_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    t = Linear()
    result = asyncio.run(t.run({"operation": "list_issues"}))
    assert result.is_error
    assert "not configured" in result.content


def test_list_issues_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_x")
    payload = _gql_response({"issues": {"nodes": []}})
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(Linear().run({"operation": "list_issues"}))
    assert not result.is_error
    assert result.content == "(no issues)"


def test_list_issues_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_x")
    nodes = [
        {
            "identifier": "ENG-1",
            "title": "Foo",
            "state": {"name": "Todo"},
            "assignee": None,
            "priority": 2,
            "url": "https://linear.app/x",
            "updatedAt": "2024-01-01",
        },
    ]
    payload = _gql_response({"issues": {"nodes": nodes}})
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Linear().run(
                {
                    "operation": "list_issues",
                    "team": "ENG",
                    "assignee_email": "x@y",
                    "limit": 5,
                }
            )
        )
    assert "[ENG-1] [Todo] Foo" in result.content


def test_get_issue_requires_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    result = asyncio.run(Linear().run({"operation": "get_issue"}))
    assert result.is_error
    assert "'id' required" in result.content


def test_get_issue_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    payload = _gql_response({"issue": None})
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Linear().run({"operation": "get_issue", "id": "ENG-9999"}),
        )
    assert "No such issue" in result.content


def test_get_issue_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    issue = {
        "identifier": "ENG-1",
        "title": "Foo",
        "description": "Body text",
        "state": {"name": "Todo"},
        "assignee": {"name": "Alice", "email": "a@x"},
        "priority": 2,
        "url": "u",
        "createdAt": "t",
        "updatedAt": "t",
        "comments": {
            "nodes": [
                {
                    "body": "Hi",
                    "user": {"name": "Bob"},
                    "createdAt": "t2",
                },
            ]
        },
    }
    payload = _gql_response({"issue": issue})
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Linear().run({"operation": "get_issue", "id": "ENG-1"}),
        )
    assert "# [ENG-1] Foo" in result.content
    assert "Assignee: Alice (a@x)" in result.content
    assert "Body text" in result.content
    assert "Bob @ t2: Hi" in result.content


def test_create_issue_requires_team_and_title(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    result = asyncio.run(Linear().run({"operation": "create_issue"}))
    assert result.is_error
    assert "'team' and 'title' required" in result.content


def test_create_issue_team_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    payload = _gql_response({"teams": {"nodes": []}})
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Linear().run({"operation": "create_issue", "team": "NOPE", "title": "x"}),
        )
    assert result.is_error
    assert "No team with key" in result.content


def test_create_issue_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    responses = [
        (_gql_response({"teams": {"nodes": [{"id": "team-id"}]}}), FetchSession()),
        (
            _gql_response(
                {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "identifier": "ENG-5",
                            "title": "T",
                            "url": "u",
                        },
                    }
                }
            ),
            FetchSession(),
        ),
    ]
    with patch("sagent.tools.linear.fetch", side_effect=responses):
        result = asyncio.run(
            Linear().run(
                {
                    "operation": "create_issue",
                    "team": "ENG",
                    "title": "T",
                    "description": "D",
                }
            ),
        )
    assert result.content == "Created ENG-5: T - u"


def test_create_issue_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    responses = [
        (_gql_response({"teams": {"nodes": [{"id": "team-id"}]}}), FetchSession()),
        (_gql_response({"issueCreate": {"success": False}}), FetchSession()),
    ]
    with patch("sagent.tools.linear.fetch", side_effect=responses):
        result = asyncio.run(
            Linear().run({"operation": "create_issue", "team": "ENG", "title": "T"}),
        )
    assert "Create failed" in result.content


def test_update_issue_requires_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    result = asyncio.run(Linear().run({"operation": "update_issue"}))
    assert result.is_error


def test_update_issue_needs_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    result = asyncio.run(
        Linear().run({"operation": "update_issue", "id": "ENG-1"}),
    )
    assert result.is_error
    assert "no fields to update" in result.content


def test_update_issue_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    payload = _gql_response(
        {
            "issueUpdate": {
                "success": True,
                "issue": {
                    "identifier": "ENG-1",
                    "title": "NewT",
                    "url": "u",
                    "state": {"name": "Done"},
                },
            }
        }
    )
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Linear().run(
                {
                    "operation": "update_issue",
                    "id": "ENG-1",
                    "title": "NewT",
                    "description": "D",
                    "state_id": "s",
                }
            ),
        )
    assert "Updated ENG-1: NewT" in result.content


def test_update_issue_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    payload = _gql_response({"issueUpdate": {"success": False}})
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Linear().run({"operation": "update_issue", "id": "ENG-1", "title": "T"}),
        )
    assert result.is_error
    assert "Update failed" in result.content


def test_add_comment_requires_id_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    result = asyncio.run(
        Linear().run({"operation": "add_comment", "id": "ENG-1"}),
    )
    assert result.is_error


def test_add_comment_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    payload = _gql_response(
        {"commentCreate": {"success": True, "comment": {"id": "c1"}}}
    )
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Linear().run({"operation": "add_comment", "id": "ENG-1", "body": "Hi"}),
        )
    assert "Comment added" in result.content


def test_add_comment_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    payload = _gql_response({"commentCreate": {"success": False}})
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Linear().run({"operation": "add_comment", "id": "ENG-1", "body": "Hi"}),
        )
    assert result.is_error


def test_unknown_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    result = asyncio.run(Linear().run({"operation": "frobnicate"}))
    assert result.is_error
    assert "Unknown operation" in result.content


def test_http_error_returns_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    err = FetchError(
        url="https://api.linear.app/graphql",
        status=500,
        headers={},
        body=b"boom",
    )
    with patch("sagent.tools.linear.fetch", side_effect=err):
        result = asyncio.run(Linear().run({"operation": "list_issues"}))
    assert result.is_error
    assert "Linear API HTTP 500" in result.content


def test_graphql_errors_returns_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    payload = json.dumps({"errors": [{"message": "bad"}]}).encode()
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(Linear().run({"operation": "list_issues"}))
    assert result.is_error
    assert "GraphQL errors" in result.content


def test_response_missing_data_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "k")
    payload = json.dumps({}).encode()
    with patch(
        "sagent.tools.linear.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(Linear().run({"operation": "list_issues"}))
    assert result.is_error
    assert "no data" in result.content


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
