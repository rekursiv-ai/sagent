"""Tests for building a package's wheel and running code against it alone."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import tempfile

import pytest

from sagent.lib.testing.wheel import Wheel, built_wheel, project_root


if TYPE_CHECKING:
    from pathlib import Path


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


def _project(tmp_path: Path, pyproject: str = _PYPROJECT) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / "demo" / "deep").mkdir(parents=True)
    (tmp_path / "demo" / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    return tmp_path


def test_project_root_is_the_nearest_project_of_that_name(tmp_path: Path) -> None:
    root = _project(tmp_path)

    assert project_root("demo-pkg", anchor=root / "demo" / "deep" / "x_test.py") == root
    # Distribution names compare normalized, as packaging does.
    assert project_root("Demo_Pkg", anchor=root / "demo" / "x_test.py") == root


def test_project_root_is_none_in_a_checkout_that_does_not_build_it(
    tmp_path: Path,
) -> None:
    """The monorepo's root project is not the package, so it builds no wheel."""
    root = _project(tmp_path, '[project]\nname = "loop"\n')

    assert project_root("demo-pkg", anchor=root / "demo" / "x_test.py") is None


def test_built_wheel_skips_where_no_project_builds_it(tmp_path: Path) -> None:
    root = _project(tmp_path, '[project]\nname = "loop"\n')

    with pytest.raises(pytest.skip.Exception, match="demo-pkg"):
        built_wheel("demo-pkg", anchor=root / "demo" / "x_test.py")


def test_run_installs_only_the_wheel_and_its_git_sources(tmp_path: Path) -> None:
    """The project's git sources stand in for siblings not yet on PyPI."""
    root = _project(tmp_path)
    wheel = Wheel(path=tmp_path / "demo_pkg-0.1.0-py3-none-any.whl", project=root)

    argv = wheel.argv("import demo", extras=("server",))

    assert argv[:3] == ["uv", "run", "--isolated"]
    assert "--no-project" in argv
    assert argv[argv.index("--with") + 1] == f"{wheel.path}[server]"
    assert [argv[i + 1] for i, part in enumerate(argv) if part == "--with"] == [
        f"{wheel.path}[server]",
        "helper @ git+https://github.com/example/helper@main",
        "pinned @ git+https://github.com/example/pinned",
    ]
    assert argv[-4:] == ["python", "-P", "-c", "import demo"]


def test_run_without_extras_installs_the_bare_wheel(tmp_path: Path) -> None:
    root = _project(tmp_path, _BUILDABLE)
    wheel = Wheel(path=tmp_path / "demo.whl", project=root)

    argv = wheel.argv("pass")

    assert [argv[i + 1] for i, part in enumerate(argv) if part == "--with"] == [
        str(wheel.path),
    ]


def test_project_root_is_none_with_no_project_above(tmp_path: Path) -> None:
    assert project_root("demo-pkg", anchor=tmp_path / "x_test.py") is None


# Stands in for uv: ``build`` writes a one-file wheel, ``run`` prints its cwd and
# argv, so the build and run paths run without a network or a real build.
_FAKE_UV = """\
#!/bin/sh
if [ "$1" = build ]; then
  out="$4"
  python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1], 'w').writestr('demo/__init__.py', 'VALUE = 7\\n')" "$out/demo_pkg-0.1.0-py3-none-any.whl"
else
  pwd
  printf '%s\\n' "$@"
fi
"""


def test_built_wheel_builds_once_and_runs_outside_the_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(_FAKE_UV, encoding="utf-8")
    (bin_dir / "uv").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    root = _project(tmp_path / "project", _BUILDABLE)

    wheel = built_wheel("demo-pkg", anchor=root / "demo" / "x_test.py")

    assert built_wheel("demo-pkg", anchor=root / "demo" / "x_test.py") is wheel
    assert wheel.names() == frozenset({"demo/__init__.py"})
    assert wheel.read("demo/__init__.py") == "VALUE = 7\n"
    cwd, *argv = wheel.run("import demo").splitlines()
    assert cwd == tempfile.gettempdir()
    assert argv == wheel.argv("import demo")[1:]


_BUILDABLE = """\
[project]
name = "demo-pkg"
version = "0.1.0"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["demo"]
"""


@pytest.mark.cli_uv
def test_a_built_wheel_runs_code_outside_the_checkout(tmp_path: Path) -> None:
    root = _project(tmp_path, _BUILDABLE)

    wheel = built_wheel("demo-pkg", anchor=root / "demo" / "x_test.py")

    assert "demo/__init__.py" in wheel.names()
    assert wheel.read("demo/__init__.py") == "VALUE = 7\n"
    assert wheel.run("import demo; print(demo.VALUE, demo.__file__)").startswith("7 ")
    assert str(root) not in wheel.run("import demo; print(demo.__file__)")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
