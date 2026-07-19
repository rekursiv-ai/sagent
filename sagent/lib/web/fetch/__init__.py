"""Unified HTTP fetch with selectable transport backends."""

from sagent.lib.web.fetch.common import ValidatedHost
from sagent.lib.web.fetch.fetch import (
    FetchSession,
    RequestParams,
    Transport,
    egress_ip,
    fetch,
    last_known_egress_ip,
    on_egress_rotation,
    resolve_transport,
    set_last_egress_ip,
)


__all__ = [
    "FetchSession",
    "RequestParams",
    "Transport",
    "ValidatedHost",
    "egress_ip",
    "fetch",
    "last_known_egress_ip",
    "on_egress_rotation",
    "resolve_transport",
    "set_last_egress_ip",
]
