"""Lazy module importing — zero work at construction time.

Returns a thin proxy whose ``__getattr__`` triggers the real import on
first attribute access.  Unlike ``importlib.util.LazyLoader`` (which
overrides ``__getattribute__`` and fires on *any* access including
``__file__``), this proxy only intercepts attributes not already in
``__dict__``, so ``inspect.getmodule`` / ``hasattr(m, '__file__')``
during heavy library init (e.g. torch) won't trigger a premature
import.

Crucially, ``find_spec`` is never called at construction time.
``find_spec`` eagerly resolves parent packages, which can trigger
``__init__.py`` chains (``sagent.lib.model`` → ``attention`` → ``torch``)
and cause circular-import failures for deeply nested submodules.
"""

from __future__ import annotations

from typing import override

import importlib
import sys
import types


def lazy_import(name: str) -> types.ModuleType:
    """Import a module lazily; loading deferred to first attribute access.

    Args:
      name: Fully qualified module name (e.g. ``"anthropic"``).

    Returns:
      module: A proxy that imports the real module on first attribute
        access.  Returns the already-imported module if present in
        ``sys.modules``.

    """
    if name in sys.modules:
        return sys.modules[name]
    return _DeferredModule(name)


class _DeferredModule(types.ModuleType):
    @override
    def __getattr__(self, name: str) -> object:
        real = importlib.import_module(self.__name__)
        object.__setattr__(self, "__class__", type(real))
        self.__dict__.update(real.__dict__)
        return getattr(self, name)
