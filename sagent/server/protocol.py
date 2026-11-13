"""Names and boundaries for Sagent's native daemon protocol.

This module is the protocol vocabulary, not the dispatcher. Keeping the
method names here makes the native contract reviewable before the server and
frontends are migrated to every method in the contract.
"""

from __future__ import annotations


NATIVE_PROTOCOL = "sagent/v2"
"""Protocol selected by the native event handshake."""

NATIVE_EVENT_METHOD = "session.event"
"""Canonical native runtime event notification method."""

NATIVE_NOTIFICATION_METHOD = "session.notification"
"""Canonical native daemon/session notification method."""

SYSTEM_METHODS = frozenset({"system.initialize", "system.shutdown"})
"""Process-level methods; these do not address a live Sagent session."""

SESSION_METHODS = frozenset(
    {
        "session.open",
        "session.list",
        "session.state",
        "session.records",
        "session.team.state",
        "session.agent.state",
        "session.submit",
        "session.halt",
        "session.close",
    },
)
"""Canonical native methods for a live session.

These methods define the live session surface exposed by the daemon.
"""

CATALOG_METHODS = frozenset(
    {
        "catalog.skills",
        "catalog.modelSettings",
        "catalog.modelSettings.set",
        "catalog.usage",
    },
)
"""Model/provider/catalog methods, separate from session lifecycle."""

COMMAND_METHODS = frozenset({"commands.list", "commands.execute"})
"""Command discovery and execution methods."""
