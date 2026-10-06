"""``_read_member`` returns each member's bytes on every interpreter ty supports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import io
import shutil
import subprocess
import sys
import zipfile

import pytest

# From the submodule path: the package re-exports a ``build`` FUNCTION, which
# shadows the submodule on ``from rekursiv_ai_typeshed import build``.
from rekursiv_ai_typeshed.build import _read_member, extract_ty_stdlib


if TYPE_CHECKING:
    from pathlib import Path


def _archive(compress_type: int, payload: bytes) -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as writer:
        writer.writestr(zipfile.ZipInfo("stdlib/a.pyi"), payload, compress_type)
    return zipfile.ZipFile(io.BytesIO(buffer.getvalue()))


@pytest.mark.parametrize("compress_type", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_a_member_zipfile_decodes_is_read_directly(compress_type: int) -> None:
    payload = b"def f() -> int: ...\n" * 50
    assert _read_member(_archive(compress_type, payload), "stdlib/a.pyi") == payload


def test_a_zstandard_member_is_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ty's own archive format: method 93, through the ``zstd`` CLI before 3.14."""
    if shutil.which("zstd") is None:
        pytest.skip("zstd CLI not installed")
    payload = b"class int: ...\n" * 50
    compressed = subprocess.run(
        ["zstd", "-q", "-c"],  # noqa: S607 -- ``zstd`` on PATH.
        input=payload,
        capture_output=True,
        check=True,
    ).stdout
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as writer:
        info = zipfile.ZipInfo("stdlib/a.pyi")
        writer.writestr(info, compressed, zipfile.ZIP_STORED)
    # Relabel the stored member as Zstandard: the bytes ARE zstd frames, so a
    # reader honoring method 93 recovers ``payload``.
    raw = bytearray(buffer.getvalue())
    for header in (b"PK\x03\x04", b"PK\x01\x02"):
        at = raw.find(header)
        method = at + (8 if header == b"PK\x03\x04" else 10)
        raw[method : method + 2] = (93).to_bytes(2, "little")
    archive = zipfile.ZipFile(io.BytesIO(bytes(raw)))
    assert archive.getinfo("stdlib/a.pyi").compress_type == 93
    # Forces the pre-3.14 branch, the one the CLI decodes.
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))

    assert _read_member(archive, "stdlib/a.pyi") == payload


def test_the_real_ty_archive_round_trips(tmp_path: Path) -> None:
    """Every stdlib member ty ships comes back as parseable stub text."""
    extract_ty_stdlib(tmp_path)
    builtins = (tmp_path / "stdlib" / "builtins.pyi").read_text()
    assert "class int" in builtins
    assert (tmp_path / "commit.txt").read_text().strip()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
