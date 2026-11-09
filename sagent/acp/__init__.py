"""Client-side support for Agent Client Protocol (ACP) agents.

ACP is an agent protocol, not a model transport. The client in this package
keeps that boundary explicit: Sagent owns the surrounding orchestration while
an ACP server owns one delegated agent session.
"""

from sagent.acp.agy import AgyAcp, find_agy_binary
from sagent.acp.client import (
    AcpAgentCapabilities,
    AcpClient,
    AcpError,
    AcpPromptResult,
    AcpSession,
    AcpUpdate,
    PermissionDecision,
    PermissionHandler,
    RequestHandler,
    UpdateHandler,
    image_content,
    text_content,
)


__all__ = [
    "AcpAgentCapabilities",
    "AcpClient",
    "AcpError",
    "AcpPromptResult",
    "AcpSession",
    "AcpUpdate",
    "AgyAcp",
    "PermissionDecision",
    "PermissionHandler",
    "RequestHandler",
    "UpdateHandler",
    "find_agy_binary",
    "image_content",
    "text_content",
]
