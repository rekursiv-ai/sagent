"""Cross-cutting type definitions for sagent.

This package is the root of the project's type system: every other
module depends on these types, and these types depend only on standard
library leaves plus ``sagent.lib`` leaves. Nothing in this tree imports
from ``agent``, ``providers``, ``tools``, or ``repl``.

This ``__init__`` does not flatten symbols into the package namespace.
Callers must reach into a specific submodule:

- ``sagent.types.runtime`` -- session/runtime event
  vocabulary, including provider-visible messages and tape events.
- ``sagent.types.tape`` -- tape mechanics:
  refs, referrable events, context splices, and splice validators.
- ``sagent.catalog.cost`` -- token count / price / cost
  calculus and the price catalog.
- ``sagent.catalog.capability`` -- ``ModelLimits``, ``ModelCapability``,
  and the thinking vocabulary a catalog row declares.
- ``sagent.types.model`` -- the narrowed ``ModelSpec``,
  the ``Model`` Protocol, and request/response types.
- ``sagent.types.providers`` -- provider Protocols.
- ``sagent.types.tools`` -- ``Tool`` Protocol.
- ``sagent.types.compactor`` -- compaction Protocols.
- ``sagent.types.exceptions`` -- user-facing errors.
"""

from sagent.types import (
    compactor,
    exceptions,
    model,
    providers,
    runtime,
    tape,
    tools,
)


__all__ = [
    "compactor",
    "exceptions",
    "model",
    "providers",
    "runtime",
    "tape",
    "tools",
]
