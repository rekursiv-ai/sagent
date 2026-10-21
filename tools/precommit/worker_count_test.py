"""Tests for ``worker_count`` and ``default_workers``."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import os
import subprocess
import sys

import pytest

from tools.precommit import worker_count as worker_count_module
from tools.precommit.worker_count import capped_workers, default_workers, worker_count


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent

_REPO = next(
    parent for parent in _THIS.parents if (parent / ".pre-commit-config.yaml").is_file()
)
"""The checkout root: in the monorepo and in every export that vendors this module."""


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("", 1),
        ("   ", 1),
        ("0", 1),
        ("-4", 1),
        ("abc", 1),
        ("5", 5),
        (" 5 ", 5),
        # `str.isdigit` is not an `int()` guard. Superscripts and other Unicode
        # digit characters satisfy it and `int()` still refuses them, so the
        # predicate admits exactly the values the conversion rejects.
        ("\N{SUPERSCRIPT TWO}", 1),
        ("\N{CIRCLED DIGIT SEVEN}", 1),
        # `lstrip("-")` strips EVERY leading dash, so a doubled one passes the
        # digit test and reaches `int()` with the dashes still attached.
        ("--4", 1),
        ("----", 1),
    ],
)
def test_a_worker_count_never_resolves_below_serial(given: str, expected: int) -> None:
    """`max_workers=0` raises ValueError, and a negative one is nonsense.

    The floor lives here rather than only at the pool because
    `affected.harvest` is a library entry point too: `max(1, ...)` there
    would mask a `--workers -4` the operator meant as an error, while
    absorbing it at the boundary keeps one rule -- an unusable count is
    serial.

    "Unusable" means EVERY unusable spelling, not the ones a `str.isdigit`
    guard happens to catch: this function exists precisely so that no
    `--workers` value can take the harvester down, and a `ValueError` escaping
    it is argparse exit 2 -- the silent downgrade to the sidecar probe its own
    docstring names as the thing worse than running serial.
    """
    assert worker_count(given) == expected


def _topology(
    root: Path,
    cores: list[tuple[int, int, int]],
) -> Path:
    """Write a fake ``/sys/devices/system/cpu``: one (package, core, max kHz) per logical CPU."""
    for i, (package, core, freq) in enumerate(cores):
        cpu = root / f"cpu{i}"
        (cpu / "topology").mkdir(parents=True)
        (cpu / "cpufreq").mkdir()
        (cpu / "topology" / "physical_package_id").write_text(f"{package}\n")
        (cpu / "topology" / "core_id").write_text(f"{core}\n")
        (cpu / "cpufreq" / "cpuinfo_max_freq").write_text(f"{freq}\n")
    return root


def _meminfo(path: Path, mib: int) -> Path:
    path.write_text(f"MemTotal: 99999999 kB\nMemAvailable: {mib * 1024} kB\n")
    return path


@pytest.fixture(autouse=True)
def no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a caller's ``WORKERS_DEFAULT`` (pre-commit sets it) out of the formula tests."""
    monkeypatch.delenv("WORKERS_DEFAULT", raising=False)


@pytest.mark.parametrize(
    ("physical", "expected"),
    [(2, 2), (4, 2), (8, 3), (12, 5), (16, 6), (24, 9), (32, 12)],
)
def test_default_workers_take_three_eighths_of_the_physical_cores(
    tmp_path: Path,
    physical: int,
    expected: int,
) -> None:
    # Two hyperthreads per core, every core the same speed.
    cores = [(0, c, 4_000_000) for c in range(physical) for _ in range(2)]
    cpu = _topology(tmp_path / "cpu", cores)
    meminfo = _meminfo(tmp_path / "meminfo", 1_000_000)
    assert default_workers(cpu=cpu, meminfo=meminfo) == expected


def test_default_workers_leave_a_performance_core_free(tmp_path: Path) -> None:
    # 8 performance cores and 8 efficiency cores: 3/8 of 16 is 6, but only 7
    # performance cores may run workers.
    cores = [(0, c, 5_000_000) for c in range(8)] + [
        (0, c, 3_000_000) for c in range(8, 16)
    ]
    cpu = _topology(tmp_path / "cpu", cores)
    meminfo = _meminfo(tmp_path / "meminfo", 1_000_000)
    assert default_workers(cpu=cpu, meminfo=meminfo) == 6
    few = _topology(
        tmp_path / "few",
        [(0, 0, 5), (0, 1, 5), (0, 2, 5), *[(0, c, 1) for c in range(3, 16)]],
    )
    assert default_workers(cpu=few, meminfo=meminfo) == 2


def test_default_workers_are_capped_by_available_memory(tmp_path: Path) -> None:
    cpu = _topology(tmp_path / "cpu", [(0, c, 1) for c in range(32)])
    assert default_workers(cpu=cpu, meminfo=_meminfo(tmp_path / "a", 6 * 1024)) == 3
    assert default_workers(cpu=cpu, meminfo=_meminfo(tmp_path / "b", 100)) == 1


def test_workers_default_overrides_the_formula(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu = _topology(tmp_path / "cpu", [(0, c, 1) for c in range(32)])
    meminfo = _meminfo(tmp_path / "m", 1_000_000)
    monkeypatch.setenv("WORKERS_DEFAULT", "3")
    assert default_workers(cpu=cpu, meminfo=meminfo) == 3
    monkeypatch.setenv("WORKERS_DEFAULT", "junk")
    assert default_workers(cpu=cpu, meminfo=meminfo) == 1


def test_the_override_is_read_from_the_environment_passed_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gate resolves workers from its own environment, never the process's."""
    cpu = _topology(tmp_path / "cpu", [(0, c, 1) for c in range(32)])
    meminfo = _meminfo(tmp_path / "m", 1_000_000)
    monkeypatch.setenv("WORKERS_DEFAULT", "3")

    assert (
        default_workers(environ={"WORKERS_DEFAULT": "7"}, cpu=cpu, meminfo=meminfo) == 7
    )
    assert default_workers(environ={}, cpu=cpu, meminfo=meminfo) == 12


@pytest.mark.parametrize(
    ("units", "per_worker", "expected"),
    [
        (None, 1, 12),
        (100, 1, 12),
        (12, 1, 12),
        (3, 1, 3),
        (1, 1, 1),
        (0, 1, 1),
        (15, 16, 1),
        (32, 16, 2),
        (1_000, 16, 12),
    ],
)
def test_a_job_size_caps_the_machine_count(
    tmp_path: Path,
    units: int | None,
    per_worker: int,
    expected: int,
) -> None:
    # 32 uniform cores: the machine alone gives 12.
    cpu = _topology(tmp_path / "cpu", [(0, c, 1) for c in range(32)])
    meminfo = _meminfo(tmp_path / "m", 1_000_000)
    got = default_workers(units=units, per_worker=per_worker, cpu=cpu, meminfo=meminfo)
    assert got == expected


def test_workers_default_overrides_a_job_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu = _topology(tmp_path / "cpu", [(0, c, 1) for c in range(32)])
    meminfo = _meminfo(tmp_path / "m", 1_000_000)
    monkeypatch.setenv("WORKERS_DEFAULT", "5")
    assert default_workers(units=1, cpu=cpu, meminfo=meminfo) == 5


def test_a_per_worker_below_one_is_refused(tmp_path: Path) -> None:
    cpu = _topology(tmp_path / "cpu", [(0, c, 1) for c in range(32)])
    with pytest.raises(ValueError, match="per_worker"):
        default_workers(units=4, per_worker=0, cpu=cpu)
    with pytest.raises(ValueError, match="per_worker"):
        capped_workers(8, units=4, per_worker=0)


@pytest.mark.parametrize(
    ("workers", "units", "per_worker", "expected"),
    [
        (8, 100, 1, 8),
        (8, 3, 1, 3),
        (8, 0, 1, 1),
        (8, 9, 10, 1),
        (8, 50, 10, 5),
        (8, 1_000, 16, 8),
        (0, 5, 1, 1),
    ],
)
def test_capped_workers_is_one_per_per_worker_units_within_the_budget(
    workers: int,
    units: int,
    per_worker: int,
    expected: int,
) -> None:
    assert capped_workers(workers, units=units, per_worker=per_worker) == expected


def _fake_tool(bin_dir: Path, name: str, body: str) -> None:
    """Write an executable ``name`` into ``bin_dir`` that runs ``body`` under sh."""
    bin_dir.mkdir(exist_ok=True)
    tool = bin_dir / name
    tool.write_text(f"#!/bin/sh\n{body}\n")
    tool.chmod(0o755)


def _vm_stat(bin_dir: Path, *, free: int, file_backed: int, inactive: int) -> None:
    """Put a fake macOS ``vm_stat`` on ``bin_dir`` reporting 16 KiB pages."""
    _fake_tool(
        bin_dir,
        "vm_stat",
        'echo "Mach Virtual Memory Statistics: (page size of 16384 bytes)"\n'
        f'echo "Pages free:                     {free}."\n'
        f'echo "Pages inactive:                 {inactive}."\n'
        f'echo "File-backed pages:              {file_backed}."',
    )


def test_vm_stat_wins_and_counts_free_and_file_backed_but_not_inactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    # 16 KiB pages: 131072 free (2 GiB) + 131072 file-backed (2 GiB) = 4 GiB;
    # 10 GiB inactive must not count.
    _vm_stat(bin_dir, free=131_072, file_backed=131_072, inactive=655_360)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cpu = _topology(tmp_path / "cpu", [(0, c, 1) for c in range(32)])
    meminfo = _meminfo(tmp_path / "meminfo", 1_000_000)
    assert default_workers(cpu=cpu, meminfo=meminfo) == 2


def test_an_unusable_vm_stat_leaves_the_formula_uncapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    _fake_tool(bin_dir, "vm_stat", "exit 1")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cpu = _topology(tmp_path / "cpu", [(0, c, 1) for c in range(32)])
    assert default_workers(cpu=cpu, meminfo=tmp_path / "absent") == 12
    _fake_tool(bin_dir, "vm_stat", 'echo "Pages free: 5."')
    assert default_workers(cpu=cpu, meminfo=tmp_path / "absent") == 12


def test_available_memory_is_whole_mebibytes(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1 kB\nMemAvailable: 4097 kB\n")
    assert worker_count_module._available_mib(meminfo) == 4
    meminfo.write_text("MemTotal: 1 kB\n")
    assert worker_count_module._available_mib(meminfo) is None


def test_macos_cores_come_from_sysctl_skipping_efficiency_cores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    # An M-series layout: 12 physical cores, 8 performance and 4 efficiency.
    _fake_tool(
        bin_dir,
        "sysctl",
        'case "$2" in\n'
        "  hw.physicalcpu) echo 12 ;;\n"
        "  hw.nperflevels) echo 2 ;;\n"
        "  hw.perflevel0.name) echo Performance ;;\n"
        "  hw.perflevel0.physicalcpu) echo 8 ;;\n"
        "  hw.perflevel1.name) echo Efficiency ;;\n"
        "  hw.perflevel1.physicalcpu) echo 4 ;;\n"
        "  *) exit 1 ;;\n"
        "esac",
    )
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    meminfo = _meminfo(tmp_path / "meminfo", 1_000_000)
    # round(12 * 3/8) = 5 (4.5 rounds up), under 8 - 1 performance cores.
    assert default_workers(cpu=tmp_path / "no-sysfs", meminfo=meminfo) == 5


def test_without_sysctl_or_sysfs_the_cpu_count_stands_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    _fake_tool(bin_dir, "sysctl", "exit 1")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    meminfo = _meminfo(tmp_path / "meminfo", 1_000_000)
    assert default_workers(cpu=tmp_path / "no-sysfs", meminfo=meminfo) == 6


def test_cores_without_frequencies_are_all_performance_cores(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu"
    for core in range(16):
        topology = cpu / f"cpu{core}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text("0\n")
        (topology / "core_id").write_text(f"{core}\n")
    meminfo = _meminfo(tmp_path / "meminfo", 1_000_000)
    assert default_workers(cpu=cpu, meminfo=meminfo) == 6


def test_running_the_module_prints_only_the_count() -> None:
    run = subprocess.run(  # noqa: S603 -- No shell; the module's own executable.
        [sys.executable, "-m", worker_count_module.__name__],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ | {"WORKERS_DEFAULT": "4"},
        cwd=_REPO,
    )
    assert run.stdout == "4\n"


def test_main_prints_the_job_capped_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["worker_count", "--units", "15", "--per-worker", "16"],
    )
    assert worker_count_module.main() == 0
    assert capsys.readouterr().out == "1\n"


def test_running_the_module_with_a_job_size_prints_the_capped_count() -> None:
    env = {k: v for k, v in os.environ.items() if k != "WORKERS_DEFAULT"}
    run = subprocess.run(  # noqa: S603 -- No shell; the module's own executable.
        [sys.executable, "-m", worker_count_module.__name__, "--units", "1"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        cwd=_REPO,
    )
    assert run.stdout == "1\n"
    run = subprocess.run(  # noqa: S603 -- No shell; the module's own executable.
        [
            sys.executable,
            "-m",
            worker_count_module.__name__,
            "--units",
            "15",
            "--per-worker",
            "16",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        cwd=_REPO,
    )
    assert run.stdout == "1\n"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
