"""The package's built wheel installs alone and ships what it was built from.

Ships to the root of every exported package and tests that package's own
wheel, reading what it should hold from ``pyproject.toml``. In a checkout whose
project is not a published package, as the monorepo's is not, it skips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import os
import re
import sys
import tempfile
import zipfile

import pytest

from sagent.lib.testing.wheel import (
    Wheel,
    built_wheel,
    console_scripts,
    declared_extras,
    missing_artifacts,
    project_root,
    top_level_modules,
)


_THIS: Final = Path(__file__).resolve()

_PYPROJECT = """\
[project]
name = "demo-pkg"
version = "0.1.0"
dependencies = ["helper>=1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["demo"]

[tool.uv.sources]
helper = { git = "https://github.com/example/helper", branch = "main" }
pinned = { git = "https://github.com/example/pinned" }
local = { path = "../local" }
"""

_BUILDABLE = """\
[project]
name = "demo-pkg"
version = "0.1.0"

[project.scripts]
demo = "demo.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["demo"]
"""


@pytest.mark.cli_uv
def test_the_package_wheel_imports_and_serves_its_scripts_alone() -> None:
    """Every module the wheel ships imports, and every script's target resolves.

    Installed alone, outside the checkout: a module that needs an unshipped
    file, a test-only dependency, or an extra fails here, not for a user.
    """
    project = project_root(anchor=_THIS)
    if project is None:
        pytest.skip("this checkout's project is not a published package")
    wheel = built_wheel(project)
    wheel.run(_import_everything(wheel))


@pytest.mark.cli_uv
def test_the_package_wheel_ships_every_artifact_it_declares() -> None:
    """An ``artifacts`` entry ships a file the build makes; missing, it ships nothing.

    Hatch builds the wheel without complaint when an artifact was never made,
    as trackinizer's web app is when ``npm run build`` did not run first.
    """
    project = project_root(anchor=_THIS)
    if project is None:
        pytest.skip("this checkout's project is not a published package")

    assert missing_artifacts(built_wheel(project)) == ()


def test_an_artifact_the_build_never_made_is_missing(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        _BUILDABLE.replace(
            'packages = ["demo"]',
            'packages = ["demo"]\nartifacts = ["demo/web/dist/**", "demo/py.typed"]',
        ),
    )
    path = tmp_path / "demo_pkg-0.1.0-py3-none-any.whl"
    _zip(path, "demo/__init__.py", "demo/py.typed")

    assert missing_artifacts(Wheel(path=path, project=root)) == ("demo/web/dist/**",)


def test_a_wheel_with_no_artifacts_declared_misses_none(tmp_path: Path) -> None:
    path = tmp_path / "demo_pkg-0.1.0-py3-none-any.whl"
    _zip(path, "demo/__init__.py")

    wheel = Wheel(path=path, project=_project(tmp_path, _BUILDABLE))

    assert missing_artifacts(wheel) == ()


# A script may need an extra (``wesearch-mcp`` needs ``mcp``), and the base install
# lacks it; it must then fail saying which, as ``pip install <pkg>[<extra>]``. Any
# other import error, or one naming an extra the wheel does not declare, fails.
_SCRIPT_CHECK: Final = """\
def _script(module, attr):
    try:
        getattr(importlib.import_module(module), attr)
    except ImportError as err:
        if not any(f"[{extra}]" in str(err) for extra in EXTRAS):
            raise
"""


def _import_everything(wheel: Wheel) -> str:
    """Return code importing every top-level module and console-script target."""
    targets = [target.partition(":") for target in console_scripts(wheel).values()]
    return "\n".join(
        (
            "import importlib",
            *(
                f"importlib.import_module({name!r})"
                for name in top_level_modules(wheel)
            ),
            *(
                (f"EXTRAS = {declared_extras(wheel)!r}", _SCRIPT_CHECK)
                if targets
                else ()
            ),
            *(f"_script({module!r}, {attr!r})" for module, _, attr in targets),
        ),
    )


def _project(tmp_path: Path, pyproject: str = _PYPROJECT) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / "demo" / "deep").mkdir(parents=True)
    (tmp_path / "demo" / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    (tmp_path / "demo" / "cli.py").write_text(
        "def main() -> int:\n    return 0\n",
        encoding="utf-8",
    )
    return tmp_path


def test_project_root_is_the_nearest_project(tmp_path: Path) -> None:
    root = _project(tmp_path)

    assert project_root(anchor=root / "demo" / "deep" / "x_test.py") == root


def test_project_root_is_none_for_a_project_with_no_build_system(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path, '[project]\nname = "demo-pkg"\n')

    assert project_root(anchor=root / "demo" / "x_test.py") is None


def test_project_root_is_none_for_the_monorepo(tmp_path: Path) -> None:
    """The monorepo builds a wheel that no one installs, in over a minute."""
    root = _project(tmp_path, _BUILDABLE.replace('"demo-pkg"', '"loop"'))

    assert project_root(anchor=root / "demo" / "x_test.py") is None


def test_project_root_is_none_with_no_project_above(tmp_path: Path) -> None:
    assert project_root(anchor=tmp_path / "x_test.py") is None


def test_built_wheel_skips_where_no_project_publishes() -> None:
    with pytest.raises(pytest.skip.Exception, match="not a published package"):
        built_wheel(None)


def test_run_installs_only_the_wheel_and_its_git_sources(tmp_path: Path) -> None:
    """The project's git sources stand in for siblings not yet on PyPI."""
    root = _project(tmp_path)
    wheel = Wheel(path=tmp_path / "demo_pkg-0.1.0-py3-none-any.whl", project=root)

    argv = wheel.argv("import demo")

    assert argv[:4] == ["uv", "run", "--isolated", "--no-project"]
    assert [argv[i + 1] for i, part in enumerate(argv) if part == "--with"] == [
        str(wheel.path),
        "helper @ git+https://github.com/example/helper@main",
        "pinned @ git+https://github.com/example/pinned",
    ]
    assert argv[-4:] == ["python", "-P", "-c", "import demo"]


def test_run_uses_the_project_indexes(tmp_path: Path) -> None:
    """A project that pins torch to a CPU index resolves it there, as its lock does."""
    root = _project(
        tmp_path,
        _BUILDABLE
        + '\n[[tool.uv.index]]\nname = "cpu"\n'
        + 'url = "https://download.pytorch.org/whl/cpu"\nexplicit = true\n',
    )

    argv = Wheel(path=tmp_path / "demo.whl", project=root).argv("pass")

    assert argv[argv.index("--index") + 1] == "https://download.pytorch.org/whl/cpu"
    assert argv[argv.index("--index-strategy") + 1] == "unsafe-best-match"


def test_run_adds_no_index_a_project_does_not_name(tmp_path: Path) -> None:
    root = _project(tmp_path, _BUILDABLE)

    argv = Wheel(path=tmp_path / "demo.whl", project=root).argv("pass")

    assert "--index" not in argv
    assert "--index-strategy" not in argv


# Stands in for uv: ``build`` writes a wheel of the project's ``demo`` package with
# one console script; ``run`` prints its cwd and argv. The build and run paths run
# without a network or a real build backend.
_FAKE_UV = """\
#!/bin/sh
if [ "$1" = build ]; then
  python3 - "$4" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1] + "/demo_pkg-0.1.0-py3-none-any.whl", "w") as w:
    w.writestr("demo/__init__.py", "VALUE = 7\\n")
    w.writestr("demo/cli.py", "def main():\\n    return 0\\n")
    w.writestr(
        "demo_pkg-0.1.0.dist-info/entry_points.txt",
        "[console_scripts]\\ndemo = demo.cli:main\\n",
    )
    w.writestr("demo_pkg-0.1.0.dist-info/METADATA", "Name: demo-pkg\\n")
EOF
else
  pwd
  printf '%s\\n' "$@"
fi
"""


def _fake_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(_FAKE_UV, encoding="utf-8")
    (bin_dir / "uv").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_built_wheel_builds_once_and_runs_outside_the_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_uv(tmp_path, monkeypatch)
    root = _project(tmp_path / "project", _BUILDABLE)

    wheel = built_wheel(root)

    assert built_wheel(root) is wheel
    assert wheel.read("demo/__init__.py") == "VALUE = 7\n"
    cwd, *argv = wheel.run("import demo").splitlines()
    # ``pwd`` prints the physical path; macOS's temp dir is under ``/var``, a symlink.
    assert Path(cwd) == Path(tempfile.gettempdir()).resolve()
    assert argv == wheel.argv("import demo")[1:]


def test_a_failed_build_says_what_the_build_said(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CI log holds only the error, so the build's own words must be in it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(
        "#!/bin/sh\necho 'npm error ERESOLVE' >&2\nexit 2\n",
        encoding="utf-8",
    )
    (bin_dir / "uv").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    root = _project(tmp_path / "broken", _BUILDABLE)

    with pytest.raises(RuntimeError, match=r"(?s)exit 2.*npm error ERESOLVE"):
        built_wheel(root)


def test_the_wheel_names_its_top_level_modules_and_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_uv(tmp_path, monkeypatch)
    wheel = built_wheel(_project(tmp_path / "project", _BUILDABLE))

    assert top_level_modules(wheel) == ("demo",)
    assert console_scripts(wheel) == {"demo": "demo.cli:main"}
    code = _import_everything(wheel)
    assert code.startswith("import importlib\nimportlib.import_module('demo')\n")
    assert code.endswith("_script('demo.cli', 'main')")


def _wheel_with_script(tmp_path: Path, cli: str, *, extras: tuple[str, ...]) -> Wheel:
    """Return a wheel whose one script runs ``demo.cli:main``, ``cli`` its body."""
    (tmp_path / "demo").mkdir(parents=True)
    (tmp_path / "demo" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "demo" / "cli.py").write_text(cli, encoding="utf-8")
    path = tmp_path / "demo_pkg-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("demo/__init__.py", "")
        archive.writestr(
            "demo_pkg-0.1.0.dist-info/entry_points.txt",
            "[console_scripts]\ndemo = demo.cli:main\n",
        )
        archive.writestr(
            "demo_pkg-0.1.0.dist-info/METADATA",
            "Name: demo-pkg\n" + "".join(f"Provides-Extra: {e}\n" for e in extras),
        )
    return Wheel(path=path, project=tmp_path)


def _exec(tmp_path: Path, code: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``code`` with ``tmp_path``'s ``demo`` package, as the installed wheel's."""
    # Not ``syspath_prepend``: pytest leaves its parameter untyped.
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    # Another test's ``demo`` must not answer for this one's.
    for name in [name for name in sys.modules if name.split(".")[0] == "demo"]:
        monkeypatch.delitem(sys.modules, name)
    exec(compile(code, "<wheel-check>", "exec"), {})  # noqa: S102 -- The check's own generated code.


def test_a_script_that_names_the_extra_it_needs_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wesearch-mcp`` needs the ``mcp`` extra and says so; that is correct."""
    wheel = _wheel_with_script(
        tmp_path,
        'raise ImportError("demo needs the mcp extra: pip install demo-pkg[mcp]")\n',
        extras=("mcp",),
    )

    _exec(tmp_path, _import_everything(wheel), monkeypatch)


@pytest.mark.parametrize(
    ("message", "extras"),
    [
        ("No module named 'requests'", ("mcp",)),
        ("demo needs: pip install demo-pkg[gpu]", ("mcp",)),
    ],
    ids=["unexplained", "undeclared extra"],
)
def test_a_script_that_fails_otherwise_fails_the_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    extras: tuple[str, ...],
) -> None:
    wheel = _wheel_with_script(
        tmp_path,
        f"raise ImportError({message!r})\n",
        extras=extras,
    )

    with pytest.raises(ImportError, match=re.escape(message)):
        _exec(tmp_path, _import_everything(wheel), monkeypatch)


def test_a_wheel_without_metadata_declares_no_extras(tmp_path: Path) -> None:
    path = tmp_path / "demo_pkg-0.1.0-py3-none-any.whl"
    _zip(path, "demo/__init__.py")

    assert declared_extras(Wheel(path=path, project=tmp_path)) == ()


def test_a_wheel_without_scripts_names_none(tmp_path: Path) -> None:
    path = tmp_path / "demo_pkg-0.1.0-py3-none-any.whl"
    _zip(path, "demo/__init__.py", "demo_pkg-0.1.0.dist-info/METADATA")

    assert console_scripts(Wheel(path=path, project=tmp_path)) == {}


def test_only_console_scripts_count_as_scripts(tmp_path: Path) -> None:
    """A plugin entry point is not a command a user runs."""
    path = tmp_path / "demo_pkg-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "demo_pkg-0.1.0.dist-info/entry_points.txt",
            "[pytest11]\ndemo = demo.plugin\n",
        )

    assert console_scripts(Wheel(path=path, project=tmp_path)) == {}


def test_a_stubs_only_wheel_names_no_module_to_import(tmp_path: Path) -> None:
    """``*-stubs`` packages and ``.pyi`` files are for checkers, not import."""
    path = tmp_path / "rekursiv_ai_typings-0.1.0-py3-none-any.whl"
    _zip(
        path,
        "yaml-stubs/__init__.pyi",
        "lib/testing/wheel.py",
        "top.pyi",
        "rekursiv_ai_typings-0.1.0.dist-info/METADATA",
    )

    assert top_level_modules(Wheel(path=path, project=tmp_path)) == ("lib",)


@pytest.mark.cli_uv
def test_a_real_wheel_runs_code_outside_the_checkout(tmp_path: Path) -> None:
    root = _project(tmp_path, _BUILDABLE)

    wheel = built_wheel(root)

    assert top_level_modules(wheel) == ("demo",)
    out = wheel.run(f"{_import_everything(wheel)}\nimport demo; print(demo.__file__)")
    assert "site-packages" in out
    assert str(root) not in out


def _zip(path: Path, *names: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
