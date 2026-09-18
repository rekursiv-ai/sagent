"""Build the patched typeshed trees from the installed ``ty`` and ``basedpyright``.

The only change to the standard-library stubs is ``typeshed.patch`` beside this
module (``int``/``float`` ``__pow__`` return ``float``, not ``Any``). Neither
checker can take that as a partial override -- basedpyright drops the ``BuiltIn``
flag unless the file sits at ``stdlib/builtins.pyi`` inside a full typeshed tree,
and ty binds ``builtins`` to its bundled stdlib -- so the patch is applied to a
COPY of the bundle and both checkers are pointed at the copy.

Each checker gets its own tree, because ``ty_extensions`` must differ: ty reads
the real ``Intersection: _SpecialForm``; basedpyright has no intersection type
and needs the alias in ``typings/basedpyright/``. A consumer's ``stubPath``
therefore stays free for its own hand-written stubs (``stubPath`` is the one
root that shadows an installed package's inline types; ``extraPaths`` does not).

Sources, each taken from the installed package so the tree always matches the
checker that reads it:

* ``stdlib/`` from the typeshed zip embedded in the ``ty`` binary. It is the
  newer of the two bundles (basedpyright's lags it by months, so ``TypedDict``
  ``closed=``/``extra_items=`` are rejected there) and it already carries
  ``ty_extensions/``.
* ``stubs/`` from basedpyright's ``dist/typeshed-fallback``. ty does not bundle
  third-party stubs; basedpyright resolves them from this directory.
* ``typings/`` beside this module: ``stdlib/`` holds whole-file replacements
  applied to both trees; ``basedpyright/`` holds replacements applied to
  basedpyright's tree only.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Final

import io
import re
import shutil
import subprocess
import sys
import zipfile


_CWD: Final = Path(__file__).resolve().parent


def build(target: Path) -> None:
    """Write the patched trees to ``target``, replacing whatever was there.

    Args:
      target: Output directory; gains ``ty/`` (``stdlib/``, ``LICENSE``,
        ``commit.txt``) and ``basedpyright/`` (the same plus ``stubs/`` and
        ``stubs.commit.txt``).

    """
    staging = target.with_name(target.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    ty_tree = staging / "ty"
    ty_tree.mkdir(parents=True)
    extract_ty_stdlib(ty_tree)
    # Read beside this file, not via ``importlib.resources``: the wheel build
    # loads this module by path, before the package is importable.
    patch = (_CWD / "typeshed.patch").read_bytes()
    subprocess.run(
        ["patch", "-p1", "--silent"],  # noqa: S607 -- ``patch`` on PATH.
        input=patch,
        cwd=ty_tree,
        check=True,
    )
    shutil.copytree(_CWD / "typings" / "stdlib", ty_tree / "stdlib", dirs_exist_ok=True)
    bpr_tree = staging / "basedpyright"
    shutil.copytree(ty_tree, bpr_tree)
    bundle = bundle_dir()
    shutil.copytree(bundle / "stubs", bpr_tree / "stubs")
    shutil.copy2(bundle / "commit.txt", bpr_tree / "stubs.commit.txt")
    shutil.copytree(
        _CWD / "typings" / "basedpyright",
        bpr_tree / "stdlib",
        dirs_exist_ok=True,
    )
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)


def bundle_dir() -> Path:
    """Locate basedpyright's bundled typeshed.

    Returns:
      bundle: Directory holding ``stdlib/``, ``stubs/``, ``commit.txt``.

    """
    spec = find_spec("basedpyright")
    if spec is None:
        raise ValueError("Expected spec is not None.")
    if spec.origin is None:
        raise ValueError("Expected spec.origin is not None.")
    return Path(spec.origin).resolve().parent / "dist" / "typeshed-fallback"


def ty_binary() -> Path:
    """Locate the ``ty`` executable beside the running interpreter.

    Returns:
      binary: The executable.

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
# decodes from Python 3.14. On older interpreters the raw member bytes go to the
# ``zstd`` CLI (present on every GitHub ``ubuntu-*`` runner and in
# Debian/Ubuntu/Homebrew).
def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    """Return one member's decompressed bytes."""
    info = archive.getinfo(name)
    # 93 is ZIP_ZSTANDARD, a name ``zipfile`` only gained in 3.14.
    if info.compress_type != 93 or sys.version_info >= (3, 14):
        return archive.read(name)
    if archive.fp is None:
        raise ValueError("Expected archive.fp is not None.")
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
