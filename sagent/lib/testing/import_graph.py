"""Read which modules a module imports from source, for the tests that pin it.

A test that keeps a module from importing something used to import the module in a
fresh interpreter and read ``sys.modules``: a process per test. ``import_graph``
reads the same graph from source, without importing anything. ``Counted`` says which
of a module's imports count, from what runs when it is imported to every import it
spells.
"""

from __future__ import annotations

from importlib.machinery import PathFinder
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import ast
import bisect
import enum
import functools
import gc
import re


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence


class Counted(enum.Enum):
    """Which imports of a module count."""

    IMPORT_TIME = enum.auto()
    """What runs when the module is imported, which is what ``sys.modules`` shows.

    An import at module level or in a class body, in either branch of an ``if``, in a
    loop, ``with`` or ``match``, in a ``try`` but not its handlers; and the module an
    ``import_module`` or ``__import__`` call names in an expression that runs then,
    such as a decorator or a default. Not a function's or a lambda's body, and not a
    ``lazy_import``, which imports on first use.
    """

    IMPORT_TIME_AND_HANDLERS = enum.auto()
    """``IMPORT_TIME``, and the imports in a ``try``'s handlers, which run when it fails."""

    RUNTIME = enum.auto()
    """Every import that can run: in a handler, a function or a lambda too, and the
    module a ``lazy_import`` names."""

    SOURCE = enum.auto()
    """``RUNTIME``, and the body of an ``if TYPE_CHECKING``, which never runs."""


# Calls that import the module their first argument names, by their last name.
_EAGER_CALLS: Final = frozenset({"import_module", "__import__"})
_LAZY_CALLS: Final = frozenset({*_EAGER_CALLS, "lazy_import"})

# The modes that count only what importing the module runs.
_AT_IMPORT: Final = frozenset({Counted.IMPORT_TIME, Counted.IMPORT_TIME_AND_HANDLERS})

# The fields of a statement that hold statements; the others hold expressions.
_BLOCKS: Final = frozenset({"body", "orelse", "finalbody", "handlers", "cases"})

# A line starting one of these at column 0 starts a definition, which ends a module's
# head of imports and constants.
_DEFINITIONS: Final = ("\n@", "\nclass ", "\ndef ", "\nasync def ")


def import_graph(
    modules: Iterable[str],
    *,
    path: Sequence[str],
    follow: Callable[[str], bool],
    counted: Counted,
) -> dict[str, frozenset[str]]:
    """Return what each module reachable from ``modules`` imports.

    Args:
      modules: The dotted names to start from, each with a source file. Importing
        one runs its packages first, so they are read too.
      path: The directories to find a module's source in, as ``sys.path`` holds them.
      follow: Whether to read a module that a read module imports. A module
        without a Python source file (built in, compiled) is never read.
      counted: Which of a module's imports count.

    Returns:
      graph: For each module read, the dotted names it imports. Importing ``a.b``
        imports ``a`` too, so both are named; ``from a import b`` names ``a`` and
        ``a.b``, which may be an attribute rather than a module.

    Raises:
      ModuleNotFoundError: A module of ``modules`` has no source file.

    """
    starts = frozenset(modules)
    return reached(
        {package for module in starts for package in _packages(module)},
        names=functools.partial(
            _source_names,
            path=tuple(path),
            counted=counted,
            starts=starts,
        ),
        follow=follow,
    )


def reached(
    modules: Iterable[str],
    *,
    names: Callable[[str], frozenset[str] | None],
    follow: Callable[[str], bool],
) -> dict[str, frozenset[str]]:
    """Return what each module reachable from ``modules`` imports, as ``names`` reads it.

    ``import_graph`` reads modules from files; a caller holding sources some other
    way, such as a wheel's, walks them with this.

    Args:
      modules: The dotted names to start from.
      names: The names a module imports, or None for a module it cannot read.
      follow: Whether to read a module that a read module imports.

    Returns:
      graph: For each module read, the names it imports.

    """
    pending = sorted(modules)
    seen = set(pending)
    graph: dict[str, frozenset[str]] = {}
    # Parsing makes many objects, and a collection pass over each tree in turn is a
    # third of the cost; the cache keeps only the names, so nothing cyclic is left.
    collecting = gc.isenabled()
    gc.disable()
    try:
        while pending:
            module = pending.pop()
            imported = names(module)
            if imported is None:
                continue
            graph[module] = imported
            for name in sorted(imported - seen):
                if follow(name):
                    seen.add(name)
                    pending.append(name)
    finally:
        if collecting:
            gc.enable()
    return graph


def module_source(module: str, *, path: Sequence[str]) -> Path | None:
    """Return the Python source file of ``module``, found without importing anything.

    Args:
      module: A dotted module name.
      path: The directories to search, as ``sys.path`` holds them.

    Returns:
      source: The ``.py`` file, or None for a module that is missing, built in,
        compiled, or a namespace package.

    """
    search: tuple[str, ...] | None = tuple(path)
    source = None  # pragma: no mutate -- the loop runs at least once and sets it.
    for package in _packages(module):
        if search is None:
            return None
        found = _found(package, search=search)
        if found is None:
            return None
        search, source = found
    return source


def imported_names(text: str, *, package: str, counted: Counted) -> frozenset[str]:
    """Return every dotted name ``text`` imports, with each package on the way.

    Args:
      text: The module's source.
      package: The package its relative imports start from: the module itself for an
        ``__init__``, else the package holding it.
      counted: Which of its imports count.

    Returns:
      names: Importing ``a.b`` imports ``a`` too, so both are named; ``from a import
        b`` names ``a`` and ``a.b``, which may be an attribute rather than a module.

    """
    head, tree = _parsed(text)
    calls = _EAGER_CALLS if counted in _AT_IMPORT else _LAZY_CALLS
    # Only a statement on a line that spells an importing call has its expressions
    # read: reading every expression is most of the cost of a module.
    lines = _lines_matching(head, re.compile("|".join(sorted(calls))))
    # Every import statement and importing call spells ``import``: a statement whose
    # lines do not is skipped, with every statement inside it.
    spelled = _lines_matching(head, re.compile("import"))
    names: set[str] = set()
    for statement in _counted_statements(tree.body, counted=counted, spelled=spelled):
        if isinstance(statement, ast.Import):
            names.update(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            if statement.level:
                # ``from . import x`` is the package; each further dot is its parent.
                base = package.rsplit(".", statement.level - 1)[0]
                origin = (
                    base if statement.module is None else f"{base}.{statement.module}"
                )
            else:
                origin = statement.module or ""  # pragma: no mutate -- never None here.
            names.add(origin)
            names.update(
                f"{origin}.{alias.name}"
                for alias in statement.names
                if alias.name != "*"
            )
        if _spans_any(lines, statement):
            names.update(_named_by_calls(statement, calls=calls, counted=counted))
    return frozenset(package for name in names for package in _packages(name))


# Cached: a walk asks for each package once per module beneath it, and every ask
# stats the files that could hold it.
@functools.cache
def _found(
    package: str,
    *,
    search: tuple[str, ...],
) -> tuple[tuple[str, ...] | None, Path | None] | None:
    """Return where ``package``'s submodules live and its ``.py`` file, or None."""
    try:
        spec = PathFinder.find_spec(package, list(search))
    except KeyError:
        # A namespace package inside a regular one: `PathFinder` looks its parent
        # up in `sys.modules`, which holds nothing here. It raises only once no
        # entry held a module, so the namespace is that directory in each entry; a
        # missing one holds nothing, as `PathFinder` treats it.
        leaf = package.rpartition(".")[2]
        return tuple(str(Path(entry, leaf)) for entry in search), None
    if spec is None:
        return None
    locations = spec.submodule_search_locations
    origin = spec.origin
    return (
        None if locations is None else tuple(locations),
        Path(origin) if origin is not None and origin.endswith(".py") else None,
    )


# Cached: every module a walk reads names most of the same packages.
@functools.cache
def _packages(module: str) -> tuple[str, ...]:
    """Return ``module`` and each package above it: ``a``, ``a.b``, ``a.b.c``."""
    parts = module.split(".")
    return tuple(".".join(parts[:end]) for end in range(1, len(parts) + 1))


def _source_names(
    module: str,
    *,
    path: tuple[str, ...],
    counted: Counted,
    starts: frozenset[str],
) -> frozenset[str] | None:
    """Return what ``module``'s source file imports, or None when it has none."""
    source = module_source(module, path=path)
    if source is None:
        if module in starts:
            raise ModuleNotFoundError(f"no source for {module!r}", name=module)
        return None
    package = module if source.name == "__init__.py" else module.rpartition(".")[0]
    return _imported_names(source, package=package, counted=counted)


# Cached by file: the tests that walk one module's graph share what they read.
@functools.cache
def _imported_names(
    source: Path,
    *,
    package: str,
    counted: Counted,
) -> frozenset[str]:
    """Return every dotted name ``source`` imports, with each package on the way."""
    return imported_names(
        source.read_text(encoding="utf-8"),
        package=package,
        counted=counted,
    )


# Parsing is most of the cost of a module, and most of a module is definitions. Every
# import statement and importing call spells ``import``, so the definitions after the
# last ``import`` add nothing: the module is parsed up to the first one. That part is
# parsed alone only when it parses: a cut inside a string or brackets leaves one open,
# which fails, and then the whole module is parsed.
def _parsed(text: str) -> tuple[str, ast.Module]:
    """Return the part of ``text`` that can import, and its tree."""
    lined = "\n" + text
    # ``lined`` is one longer, so this starts the search at the ``i`` of ``import``.
    last = text.rfind("import") + 1
    cut = min(
        (at for start in _DEFINITIONS if (at := lined.find(start, last)) >= 0),
        default=None,
    )
    head = text[:cut]
    try:
        return head, ast.parse(head)
    except SyntaxError:
        return text, ast.parse(text)


def _lines_matching(text: str, pattern: re.Pattern[str]) -> list[int]:
    """Return the sorted 1-based numbers of the lines of ``text`` ``pattern`` matches."""
    lines: list[int] = []
    line, counted_to = 1, 0
    for found in pattern.finditer(text):
        line += text.count("\n", counted_to, found.start())
        counted_to = found.start()
        if not lines or lines[-1] != line:
            lines.append(line)
    return lines


# A statement none of whose lines is in ``spelled`` is skipped, with its body.
def _counted_statements(
    body: list[ast.stmt],
    *,
    counted: Counted,
    spelled: list[int],
) -> Iterator[ast.stmt | ast.excepthandler]:
    """Yield each statement of ``body``, nested ones too, whose imports count."""
    pending: list[ast.stmt | ast.excepthandler] = list(reversed(body))
    while pending:
        node = pending.pop()
        if not _spans_any(spelled, node):
            continue
        yield node
        children: list[ast.stmt | ast.excepthandler]
        if (
            isinstance(node, ast.If)
            and counted is not Counted.SOURCE
            and _is_type_checking(node.test)
        ):
            children = [*node.orelse]
        elif counted in _AT_IMPORT and isinstance(
            node,
            ast.FunctionDef | ast.AsyncFunctionDef,
        ):
            # Defining a function runs its decorators and defaults, not its body.
            children = []
        elif isinstance(node, ast.Try | ast.TryStar):
            # A handler runs only when its ``try`` fails, which importing here did
            # not: ``hashlib`` imports ``logging`` only to report a missing hash.
            handlers = [] if counted is Counted.IMPORT_TIME else node.handlers
            children = [*node.body, *handlers, *node.orelse, *node.finalbody]
        elif isinstance(node, ast.Match):
            children = [statement for case in node.cases for statement in case.body]
        else:
            # A ``match``'s cases are read above, so these fields hold statements.
            children = [
                child
                for field, value in cast(
                    "Iterator[tuple[str, object]]",
                    ast.iter_fields(node),
                )
                if field in _BLOCKS
                for child in cast("list[ast.stmt | ast.excepthandler]", value)
            ]
        pending.extend(reversed(children))


def _named_by_calls(
    statement: ast.AST,
    *,
    calls: frozenset[str],
    counted: Counted,
) -> Iterator[str]:
    """Yield the module each importing call in ``statement``'s expressions names."""
    pending: list[ast.AST] = [
        child
        for field, value in cast(
            "Iterator[tuple[str, object]]",
            ast.iter_fields(statement),
        )
        if field not in _BLOCKS
        for child in _items(value)
        if isinstance(child, ast.AST)
    ]
    while pending:
        node = pending.pop()
        if counted in _AT_IMPORT and isinstance(node, ast.Lambda):
            # A lambda's body runs when it is called, its defaults when it is made.
            pending.extend(node.args.defaults)
            pending.extend(default for default in node.args.kw_defaults if default)
            continue
        pending.extend(ast.iter_child_nodes(node))
        if not isinstance(node, ast.Call) or not node.args:
            continue
        match node.func:
            case ast.Attribute(attr=called) | ast.Name(id=called):
                pass
            case _:
                continue
        first = node.args[0]
        if (
            called in calls
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
        ):
            yield first.value


def _items(value: object) -> list[object]:
    """Return a field's nodes: each of a list's, or the field alone."""
    return list(cast("list[object]", value)) if isinstance(value, list) else [value]


def _spans_any(lines: list[int], node: ast.stmt | ast.excepthandler) -> bool:
    """Return whether one of the sorted ``lines`` lies within ``node``."""
    # A decorated definition starts at its first decorator, not at ``def``.
    decorators = (
        node.decorator_list
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        else []
    )
    at = bisect.bisect_left(
        lines,
        min([node.lineno, *(decorator.lineno for decorator in decorators)]),
    )
    return at < len(lines) and lines[at] <= (node.end_lineno or node.lineno)


def _is_type_checking(test: ast.expr) -> bool:
    """Return whether ``test`` is ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )
