"""Tests for the static import graph the import-purity tests read."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import gc

import pytest

from sagent.lib.testing.import_graph import (
    Counted,
    _parsed,
    import_graph,
    imported_names,
    module_source,
    reached,
)


if TYPE_CHECKING:
    from pathlib import Path


def _package(root: Path, files: dict[str, str]) -> list[str]:
    """Write ``files`` (relative path to text) under ``root``; return ``[str(root)]``."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return [str(root)]


def _every(name: str) -> bool:
    del name
    return True


# A module spelling an import in every place one can go.
_EVERY_PLACE: Final = (
    "from typing import TYPE_CHECKING\n"
    "import top\n"
    "if TYPE_CHECKING:\n    import typed\nelse:\n    import untyped\n"
    "if top.x:\n    import branch\n"
    "try:\n    import tried\nexcept ImportError:\n    import fallback\n"
    "else:\n    import tried_else\nfinally:\n    import tried_finally\n"
    "class C:\n    import in_class\n"
    "    def method(self):\n        import in_method\n"
    "def later(x=__import__('default')):\n    import in_function\n"
    "    return lazy_import('later_lazily', 'name')\n"
    "@decorate(import_module('decorator'))\ndef decorated():\n    pass\n"
    "loaded = import_module('by_call')\n"
    "qualified = importlib.import_module('by_attribute')\n"
    # A callee that is itself a call names no function, so is not read as one.
    "fetched = getattr(importlib, 'import_module')('by_returned_callee')\n"
    "deferred = lazy_import('lazily')\n"
    "made = lambda: import_module('in_lambda')\n"
    "keyed = lambda *, k=import_module('lambda_default'): k\n"
    "match top.y:\n    case 1:\n        import in_case\n"
)

# What importing the module runs: no handler, function or lambda body, lazy proxy, or
# typing-only import.
_AT_IMPORT: Final = frozenset(
    {
        "typing",
        "typing.TYPE_CHECKING",
        "top",
        "untyped",
        "branch",
        "tried",
        "tried_else",
        "tried_finally",
        "in_class",
        "default",
        "decorator",
        "by_call",
        "by_attribute",
        "lambda_default",
        "in_case",
    },
)
_LATER: Final = frozenset(
    {"in_method", "in_function", "later_lazily", "lazily", "in_lambda"},
)


@pytest.mark.parametrize(
    ("counted", "expected"),
    [
        (Counted.IMPORT_TIME, _AT_IMPORT),
        (Counted.IMPORT_TIME_AND_HANDLERS, {*_AT_IMPORT, "fallback"}),
        (Counted.RUNTIME, {*_AT_IMPORT, "fallback", *_LATER}),
        (Counted.SOURCE, {*_AT_IMPORT, "fallback", *_LATER, "typed"}),
    ],
    ids=["import_time", "and_handlers", "runtime", "source"],
)
def test_each_mode_counts_the_imports_it_names(
    counted: Counted,
    expected: frozenset[str],
    tmp_path: Path,
) -> None:
    path = _package(tmp_path, {"mod.py": _EVERY_PLACE})

    graph = import_graph(["mod"], path=path, follow=_every, counted=counted)

    assert graph == {"mod": expected}


def test_a_typing_import_is_skipped_however_type_checking_is_spelled(
    tmp_path: Path,
) -> None:
    path = _package(
        tmp_path,
        {"mod.py": "import typing\nif typing.TYPE_CHECKING:\n    import typed\n"},
    )

    graph = import_graph(["mod"], path=path, follow=_every, counted=Counted.RUNTIME)

    assert graph == {"mod": {"typing"}}


def test_names_each_package_on_the_way_and_resolves_relative_imports(
    tmp_path: Path,
) -> None:
    path = _package(
        tmp_path,
        {
            "pkg/__init__.py": "from . import sibling\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/mod.py": (
                "import a.b.c\n"
                "from .. import up\n"
                "from ..other import thing\n"
                "from .near import *\n"
                "x = import_module(name)\n"
            ),
        },
    )

    graph = import_graph(
        ["pkg.sub.mod"],
        path=path,
        follow=_every,
        counted=Counted.IMPORT_TIME,
    )

    assert graph["pkg.sub.mod"] == {
        "a",
        "a.b",
        "a.b.c",
        "pkg",
        "pkg.up",
        "pkg.other",
        "pkg.other.thing",
        "pkg.sub",
        "pkg.sub.near",
    }
    # Importing a module runs each package above it; the package imports ``sibling``.
    assert graph["pkg"] == {"pkg", "pkg.sibling"}
    assert set(graph) == {"pkg", "pkg.sub", "pkg.sub.mod"}


def test_follows_only_what_follow_allows_and_has_a_source(tmp_path: Path) -> None:
    path = _package(
        tmp_path,
        {
            "start.py": "import kept, dropped, compiled, missing\n",
            "kept.py": "import deeper\n",
            "deeper.py": "import json\n",
            "dropped.py": "import never_read\n",
            "compiled.so": "",
        },
    )

    graph = import_graph(
        ["start"],
        path=path,
        follow=lambda name: name != "dropped",
        counted=Counted.IMPORT_TIME,
    )

    assert graph == {
        "start": {"kept", "dropped", "compiled", "missing"},
        "kept": {"deeper"},
        "deeper": {"json"},
    }


def test_a_start_without_a_source_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ModuleNotFoundError, match="absent"):
        import_graph(
            ["absent"],
            path=[str(tmp_path)],
            follow=_every,
            counted=Counted.IMPORT_TIME,
        )


@pytest.mark.parametrize("counted", [Counted.IMPORT_TIME, Counted.RUNTIME])
def test_an_import_after_the_first_definition_counts(
    counted: Counted,
    tmp_path: Path,
) -> None:
    """Only a module's head is parsed when the rest spells no import; this one does."""
    path = _package(
        tmp_path,
        {"mod.py": "import head\ndef run():\n    pass\nimport tail\n"},
    )

    graph = import_graph(["mod"], path=path, follow=_every, counted=counted)

    assert graph == {"mod": {"head", "tail"}}


def test_a_definition_spelled_in_a_string_does_not_end_the_head(
    tmp_path: Path,
) -> None:
    """A head cut inside a string fails to parse, so the whole module is read."""
    path = _package(
        tmp_path,
        {
            "mod.py": (
                'import real\nWORDS = """\nclass of bugs\n"""\ndef run():\n    pass\n'
            ),
        },
    )

    graph = import_graph(["mod"], path=path, follow=_every, counted=Counted.SOURCE)

    assert graph == {"mod": {"real"}}


def test_reads_each_file_once_per_process(tmp_path: Path) -> None:
    path = _package(tmp_path, {"mod.py": "import first\n"})
    import_graph(["mod"], path=path, follow=_every, counted=Counted.IMPORT_TIME)
    (tmp_path / "mod.py").write_text("import second\n")

    graph = import_graph(["mod"], path=path, follow=_every, counted=Counted.IMPORT_TIME)

    assert graph == {"mod": {"first"}}


@pytest.mark.parametrize("collecting", [True, False])
def test_leaves_the_collector_as_it_found_it(
    collecting: bool,
    tmp_path: Path,
) -> None:
    path = _package(tmp_path, {"mod.py": "import json\n"})
    was = gc.isenabled()
    (gc.enable if collecting else gc.disable)()
    try:
        import_graph(["mod"], path=path, follow=_every, counted=Counted.IMPORT_TIME)
        assert gc.isenabled() is collecting
    finally:
        (gc.enable if was else gc.disable)()


def test_module_source_finds_a_module_a_package_or_nothing(tmp_path: Path) -> None:
    path = _package(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "",
            "plain.py": "",
            "native.so": "",
            "space/inner.py": "",
            "pkg/space/inner.py": "",
        },
    )

    assert module_source("pkg", path=path) == tmp_path / "pkg" / "__init__.py"
    assert module_source("pkg.mod", path=path) == tmp_path / "pkg" / "mod.py"
    assert module_source("space.inner", path=path) == tmp_path / "space" / "inner.py"
    assert module_source("space", path=path) is None
    # A namespace package inside a regular one: its parent is not imported, so
    # `PathFinder` alone raises `KeyError` building the namespace path.
    assert module_source("pkg.space.inner", path=path) == (
        tmp_path / "pkg" / "space" / "inner.py"
    )
    assert module_source("pkg.space", path=path) is None
    assert module_source("native", path=path) is None
    assert module_source("pkg.absent", path=path) is None
    assert module_source("plain.attribute", path=path) is None


def test_reached_walks_any_source_of_names() -> None:
    """A caller holding sources in memory, such as a wheel's, walks them the same way."""
    imports = {
        "start": frozenset({"kept", "unread", "skipped", "deeper"}),
        "kept": frozenset({"start", "deeper"}),
        "deeper": frozenset[str](),
        "skipped": frozenset({"never"}),
    }
    read: list[str] = []
    assert reached(
        ["start"],
        names=lambda name: read.append(name) or imports.get(name),
        follow=lambda name: name != "skipped",
    ) == {
        "start": imports["start"],
        "kept": imports["kept"],
        "deeper": imports["deeper"],
    }
    assert sorted(read) == ["deeper", "kept", "start", "unread"]


def test_imported_names_reads_source_text_relative_to_its_package() -> None:
    assert imported_names(
        "from . import sibling\nfrom ..up import x\n",
        package="a.b",
        counted=Counted.IMPORT_TIME,
    ) == {"a", "a.b", "a.b.sibling", "a.up", "a.up.x"}


def test_each_dot_past_the_first_climbs_one_package() -> None:
    assert imported_names(
        "from .. import x\n",
        package="a.b.c",
        counted=Counted.IMPORT_TIME,
    ) == {"a", "a.b", "a.b.x"}


def test_an_import_after_a_blank_first_line_counts() -> None:
    assert imported_names(
        "\nimport os\n",
        package="",
        counted=Counted.IMPORT_TIME,
    ) == {"os"}


def test_a_callee_that_names_no_function_does_not_end_the_statement() -> None:
    assert imported_names(
        "pair = (import_module('json'), factory()('x'))\n",
        package="",
        counted=Counted.IMPORT_TIME,
    ) == {"json"}


def test_a_start_without_a_source_names_itself_in_the_error(tmp_path: Path) -> None:
    with pytest.raises(ModuleNotFoundError) as raised:
        import_graph(
            ["absent"],
            path=[str(tmp_path)],
            follow=_every,
            counted=Counted.IMPORT_TIME,
        )
    assert (str(raised.value), raised.value.name) == (
        "no source for 'absent'",
        "absent",
    )


@pytest.mark.parametrize(
    ("text", "head"),
    [
        (
            "import os\nX = 1\n\ndef f():\n    pass\n\nclass C:\n    pass\n",
            "import os\nX = 1\n\n",
        ),
        ("def f():\n    pass\n", ""),
        ("import os\n\ndef f():\n    pass\n\ndef g():\n    pass\n", "import os\n\n"),
        (
            "import os\n@importer\ndef f():\n    pass\n",
            "import os\n@importer\ndef f():\n    pass\n",
        ),
        ('import os\nS = """\ndef x\n"""\n', 'import os\nS = """\ndef x\n"""\n'),
    ],
    ids=[
        "cut-at-the-first-definition",
        "no-import",
        "first-of-two-definitions",
        "decorator",
        "no-definition",
    ],
)
def test_only_the_head_up_to_the_first_definition_after_the_last_import_is_parsed(
    text: str,
    head: str,
) -> None:
    """A head cut inside a decorator's line fails to parse, so the whole is read."""
    assert _parsed(text)[0] == head


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
