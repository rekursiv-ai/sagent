"""Generate the patched typeshed tree while the wheel is being built.

Runs inside hatchling's isolated build environment, where ``build-system.requires``
has installed the exact ``ty`` and ``basedpyright`` this package pins; the tree
therefore matches the checkers a consumer receives as dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Final, cast, override

import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.builders.wheel import WheelBuilderConfig


_CWD: Final = Path(__file__).resolve().parent


class TypeshedBuildHook(BuildHookInterface[WheelBuilderConfig]):
    """Generate the tree into a temp dir and register it as ``shared-data``."""

    PLUGIN_NAME = "custom"

    @override
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        del version
        # Loaded by path: the package under construction is not importable from
        # the isolated build environment.
        spec = spec_from_file_location("rekursiv_ai_typeshed.build", _CWD / "build.py")
        assert spec is not None
        assert spec.loader is not None
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        build = cast("Callable[[Path], None]", module.build)
        # Outside the source tree: a checker walking the checkout must never see
        # a second, unpatched-looking stdlib.
        target = Path(tempfile.mkdtemp(prefix="rekursiv-ai-typeshed-")) / "typeshed.d"
        build(target)
        shared = cast("dict[str, str]", build_data.setdefault("shared_data", {}))
        shared[str(target)] = "share/rekursiv-ai-typeshed"
