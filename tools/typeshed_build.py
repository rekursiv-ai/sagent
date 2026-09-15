#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Build the house typeshed at ``typings/typeshed.d`` from the installed checkers.

The only house change to the standard-library stubs is ``typings/typeshed.patch``
(``int``/``float`` ``__pow__`` return ``float``, not ``Any``). Neither checker can
take that as a partial override -- basedpyright drops the ``BuiltIn`` flag unless
the file sits at ``stdlib/builtins.pyi`` inside a full typeshed tree, and ty binds
``builtins`` to its bundled stdlib -- so the patch is applied to a COPY of the
bundle and both checkers are pointed at the copy. The copy is generated, not
tracked: the artifact under review is the patch.

Sources, each taken from the installed package so the tree always matches the
checker that reads it:

* ``stdlib/`` from the typeshed zip embedded in the ``ty`` binary. It is the
  newer of the two bundles (basedpyright's lags it by months, so ``TypedDict``
  ``closed=``/``extra_items=`` are rejected there) and it already carries
  ``ty_extensions/``, which basedpyright's bundle lacks; a stdlib without it
  degrades configgle's ``Intersection`` return to a plain alias under ty.
* ``stubs/`` from basedpyright's ``dist/typeshed-fallback``. ty does not bundle
  third-party stubs; basedpyright resolves them from this directory.
* ``basedpyright/`` -- a stub root for basedpyright alone: every hand-written
  stub under ``typings/`` symlinked in, plus ``ty_extensions/`` holding
  configgle's polyfill (``Intersection[A, B] = A``). ``stubPath`` is
  single-valued and wins over typeshed, so pointing it here is the only way
  basedpyright sees the polyfill while ty (whose ``extra-paths`` would also
  beat its stdlib) keeps the real ``_SpecialForm``.

A stamp file records the inputs (bundle commit, ``ty`` binary digest, patch
digest); a rerun with an unchanged stamp is a no-op, so this is safe to call from
every gate. ``--check`` reports staleness without writing.

Examples:
  ./typeshed_build.py            # build or refresh
  ./typeshed_build.py --check    # exit 1 if the tree is missing or stale

'''
# fmt: on

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import hashlib
import io
import re
import shutil
import subprocess
import sys
import zipfile


_CWD: Final = Path(__file__).resolve().parent


def main() -> int:
    """Run the program.

    Returns:
      exit_code: 0 when the tree is fresh (or was refreshed); 1 under ``--check``
        when it is missing or stale.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    root = project_root()
    house = root / "typings" / "typeshed.d"
    patch = root / "typings" / "typeshed.patch"
    stamp = compute_stamp(patch)
    if is_fresh(house, stamp):
        return 0
    if flags.check:
        sys.stderr.write(f"{house} is stale; run tools/typeshed_build.py\n")
        return 1
    build(house, patch=patch, stamp=stamp)
    sys.stderr.write(f"Built {house}\n")
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    check: bool


def is_fresh(house: Path, stamp: str) -> bool:
    """Report whether ``house`` was built from exactly these inputs.

    Args:
      house: The generated tree.
      stamp: Expected digest from :func:`compute_stamp`.

    Returns:
      fresh: True iff the stamp matches and the patched file is present.

    """
    marker = house / ".stamp"
    # Both the stamp and the patched file: a wiped tree that kept its stamp
    # otherwise reads as fresh and both checkers run against nothing.
    return (
        marker.is_file()
        and marker.read_text() == stamp
        and (house / "stdlib" / "builtins.pyi").is_file()
        and (
            polyfill_source() is None
            or (house / "basedpyright" / "ty_extensions" / "__init__.pyi").is_file()
        )
    )


def compute_stamp(patch: Path) -> str:
    """Digest every input the tree is derived from.

    Args:
      patch: The house patch file.

    Returns:
      stamp: Hex digest; equal iff a rebuild would reproduce the current tree.

    """
    commit = (bundle_dir() / "commit.txt").read_bytes()
    # The stub-root listing is an input too: a new `typings/<pkg>/` must gain
    # its symlink or basedpyright silently loses that package's stubs.
    siblings = ",".join(sorted(stub_dirs(patch.parent))).encode()
    return hashlib.sha256(
        commit
        + ty_binary().read_bytes()
        + patch.read_bytes()
        + (b"" if (polyfill := polyfill_source()) is None else polyfill.read_bytes())
        + siblings,
    ).hexdigest()


def build(house: Path, *, patch: Path, stamp: str) -> None:
    """Regenerate ``house`` from the bundles and ``patch``.

    Args:
      house: Output directory; replaced wholesale.
      patch: Unified diff applied at ``house`` with ``-p1``.
      stamp: Digest written last, so an interrupted build reads as stale.

    """
    bundle = bundle_dir()
    staging = house.with_name(house.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    extract_ty_stdlib(staging)
    shutil.copytree(bundle / "stubs", staging / "stubs")
    shutil.copy2(bundle / "commit.txt", staging / "stubs.commit.txt")
    stub_root = staging / "basedpyright"
    stub_root.mkdir()
    for name in stub_dirs(house.parent):
        (stub_root / name).symlink_to(
            Path("..") / ".." / name, target_is_directory=True
        )
    source = polyfill_source()
    if source is not None:
        polyfill = stub_root / "ty_extensions" / "__init__.pyi"
        polyfill.parent.mkdir()
        shutil.copy2(source, polyfill)
    subprocess.run(  # noqa: S603 -- no shell; fixed argv.
        ["patch", "-p1", "--silent", "--input", str(patch)],  # noqa: S607 -- ``patch`` on PATH.
        cwd=staging,
        check=True,
    )
    shutil.rmtree(house, ignore_errors=True)
    staging.rename(house)
    (house / ".stamp").write_text(stamp)


def stub_dirs(typings: Path) -> list[str]:
    """Name the hand-written stub packages under ``typings``.

    Args:
      typings: The ``typings/`` directory.

    Returns:
      names: Sorted directory names, excluding generated trees and dotfiles.

    """
    return sorted(
        e.name
        for e in typings.iterdir()
        if e.is_dir() and not e.name.startswith(".") and not e.name.endswith(".d")
    )


def bundle_dir() -> Path:
    """Locate basedpyright's bundled typeshed.

    Returns:
      bundle: Directory holding ``stdlib/``, ``stubs/``, ``commit.txt``.

    """
    spec = find_spec("basedpyright")
    assert spec is not None
    assert spec.origin is not None
    return Path(spec.origin).resolve().parent / "dist" / "typeshed-fallback"


def project_root() -> Path:
    """Locate the checkout this script serves: the nearest ``pyproject.toml`` above it.

    The monorepo and each exported package place this file at a different
    depth; walking up to the project file serves both without a copy.

    Returns:
      root: Directory holding ``pyproject.toml`` and ``typings/``.

    """
    for candidate in _CWD.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"No pyproject.toml above {_CWD}.")


def polyfill_source() -> Path | None:
    """Locate configgle's ``Intersection`` polyfill, the basedpyright-side stub.

    Resolved from the installed ``ty_extensions`` package: the monorepo and every
    export that depends on configgle install it. An export without it gets no
    polyfill, and its ``stubPath`` root holds only the hand-written stubs.

    Returns:
      source: The polyfill module, or None when the package is not installed.

    """
    spec = find_spec("ty_extensions")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def ty_binary() -> Path:
    """Locate the ``ty`` executable in the active environment.

    Returns:
      binary: Path beside the running interpreter.

    """
    return Path(sys.executable).parent / "ty"


# The zip has no leading signature to seek to, so each end-of-central-directory
# record is tried against the local-file header its offsets point back to.
def extract_ty_stdlib(target: Path) -> None:
    """Unpack the typeshed embedded in ``ty`` (``stdlib/``, licence, commit).

    Args:
      target: Staging root; gains ``stdlib/``, ``LICENSE``, ``commit.txt``.

    """
    data = ty_binary().read_bytes()
    for match in re.finditer(rb"PK\x05\x06", data):
        eocd = match.start()
        cd_size = int.from_bytes(data[eocd + 12 : eocd + 16], "little")
        cd_offset = int.from_bytes(data[eocd + 16 : eocd + 20], "little")
        start = eocd - cd_size - cd_offset
        if start < 0 or data[start : start + 4] != b"PK\x03\x04":
            continue
        archive = zipfile.ZipFile(io.BytesIO(data[start : eocd + 22]))
        names = archive.namelist()
        if "stdlib/ty_extensions/__init__.pyi" not in names:
            continue
        for name in names:
            if name.endswith("/") or not name.startswith("stdlib/"):
                continue
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_read_member(archive, name))
        (target / "LICENSE").write_bytes(_read_member(archive, "LICENSE"))
        (target / "commit.txt").write_bytes(_read_member(archive, "source_commit.txt"))
        return
    raise RuntimeError(f"{ty_binary()} carries no embedded typeshed.")


# Ty compresses its archive with Zstandard (method 93), which ``zipfile`` only
# decodes from Python 3.14. The public packages validate on 3.12, so the raw
# member bytes are handed to the ``zstd`` CLI there (present on every GitHub
# ``ubuntu-*`` runner and in Debian/Ubuntu/Homebrew).
def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    """Return one member's decompressed bytes."""
    info = archive.getinfo(name)
    # 93 is ZIP_ZSTANDARD, a name ``zipfile`` only gained in 3.14.
    if info.compress_type != 93 or sys.version_info >= (3, 14):
        return archive.read(name)
    assert archive.fp is not None
    archive.fp.seek(info.header_offset)
    header = archive.fp.read(30)
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    archive.fp.seek(info.header_offset + 30 + name_len + extra_len)
    raw = archive.fp.read(info.compress_size)
    return subprocess.run(
        ["zstd", "-d", "-c"],  # noqa: S607 -- ``zstd`` on PATH.
        input=raw,
        capture_output=True,
        check=True,
    ).stdout


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the tree is missing or stale; never write.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
