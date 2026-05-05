# pytest fixtures look unused to pyright; pytest wires them by name
"""Tests for tools.linear.

All tests mock the HTTP layer - no network calls.
"""

from __future__ import annotations

from typing import Any

import json as _json

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools import linear as linear_mod


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-linear"),),
        "multipart/x-tool-call",
    )


def _txt(msg: Message) -> str:
    if isinstance(msg, TextMessage):
        return msg.content
    if isinstance(msg, MultipartMessage):
        for p in msg.content:
            if isinstance(p, TextMessage):
                return p.content
    return ""


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    canned: dict[str, dict[str, Any]],
) -> None:
    """Mock ``fetch`` for Linear's GraphQL POST calls."""

    def _mock_fetch(
        url: str, *, method: str = "GET", json: Any = None, **kwargs: object
    ) -> bytes:
        del url, method, kwargs
        body: dict[str, Any] = json or {}
        query: str = body.get("query", "")
        for marker, data in canned.items():
            if marker in query:
                return _json.dumps({"data": data}).encode()
        return _json.dumps({"errors": [{"message": "no canned response"}]}).encode()

    monkeypatch.setattr("sagent.tools.linear.fetch", _mock_fetch)


@pytest.fixture
def _with_key(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction] -- pytest fixture used via decorator
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_test")


class TestAuth:
    @pytest.mark.anyio
    async def test_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LINEAR_API_KEY", raising=False)
        tool = linear_mod.Linear()
        resp = await tool.run(_msg(json_freeze({"operation": "list_issues"})))
        assert resp.descriptor == "text/x-error"
        assert "not configured" in _txt(resp)


@pytest.mark.usefixtures("_with_key")
class TestListIssues:
    @pytest.mark.anyio
    async def test_returns_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "ListIssues": {
                    "issues": {
                        "nodes": [
                            {
                                "identifier": "ENG-1",
                                "title": "First bug",
                                "state": {"name": "In Progress"},
                            },
                        ]
                    }
                }
            },
        )
        tool = linear_mod.Linear()
        resp = await tool.run(_msg(json_freeze({"operation": "list_issues"})))
        assert "ENG-1" in _txt(resp)
        assert "First bug" in _txt(resp)

    @pytest.mark.anyio
    async def test_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {"ListIssues": {"issues": {"nodes": []}}},
        )
        tool = linear_mod.Linear()
        resp = await tool.run(_msg(json_freeze({"operation": "list_issues"})))
        assert _txt(resp) == "(no issues)"


@pytest.mark.usefixtures("_with_key")
class TestGetIssue:
    @pytest.mark.anyio
    async def test_requires_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(monkeypatch, {})
        tool = linear_mod.Linear()
        resp = await tool.run(_msg(json_freeze({"operation": "get_issue"})))
        assert resp.descriptor == "text/x-error"
        assert "required" in _txt(resp)

    @pytest.mark.anyio
    async def test_renders(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "GetIssue": {
                    "issue": {
                        "identifier": "ENG-42",
                        "title": "A bug",
                        "description": "Reproduces on Tuesdays.",
                        "state": {"name": "Todo"},
                        "url": "https://linear.app/x/ENG-42",
                        "priority": 2,
                        "comments": {"nodes": []},
                    }
                }
            },
        )
        tool = linear_mod.Linear()
        resp = await tool.run(
            _msg(json_freeze({"operation": "get_issue", "id": "ENG-42"}))
        )
        assert "ENG-42" in _txt(resp)
        assert "Tuesdays" in _txt(resp)


@pytest.mark.usefixtures("_with_key")
class TestCreateUpdateComment:
    @pytest.mark.anyio
    async def test_create_requires_team_title(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(monkeypatch, {})
        tool = linear_mod.Linear()
        resp = await tool.run(_msg(json_freeze({"operation": "create_issue"})))
        assert resp.descriptor == "text/x-error"
        assert "required" in _txt(resp)

    @pytest.mark.anyio
    async def test_create_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "TeamByKey": {"teams": {"nodes": [{"id": "team-id"}]}},
                "CreateIssue": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "identifier": "ENG-5",
                            "title": "New",
                            "url": "u",
                        },
                    }
                },
            },
        )
        tool = linear_mod.Linear()
        resp = await tool.run(
            _msg(
                json_freeze(
                    {"operation": "create_issue", "team": "ENG", "title": "New"}
                )
            )
        )
        assert "ENG-5" in _txt(resp)

    @pytest.mark.anyio
    async def test_update_requires_fields(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(monkeypatch, {})
        tool = linear_mod.Linear()
        resp = await tool.run(
            _msg(json_freeze({"operation": "update_issue", "id": "ENG-1"}))
        )
        assert resp.descriptor == "text/x-error"
        assert "no fields" in _txt(resp)

    @pytest.mark.anyio
    async def test_update_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "UpdateIssue": {
                    "issueUpdate": {
                        "success": True,
                        "issue": {
                            "identifier": "ENG-1",
                            "title": "Updated",
                            "url": "u",
                            "state": {"name": "Done"},
                        },
                    }
                },
            },
        )
        tool = linear_mod.Linear()
        resp = await tool.run(
            _msg(
                json_freeze(
                    {"operation": "update_issue", "id": "ENG-1", "title": "Updated"}
                )
            )
        )
        assert "Updated" in _txt(resp)

    @pytest.mark.anyio
    async def test_comment_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "AddComment": {
                    "commentCreate": {
                        "success": True,
                        "comment": {"id": "c-1"},
                    }
                }
            },
        )
        tool = linear_mod.Linear()
        resp = await tool.run(
            _msg(
                json_freeze(
                    {"operation": "add_comment", "id": "ENG-1", "body": "Looks fine."}
                )
            )
        )
        assert "added" in _txt(resp).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
