#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Size a worker pool: parse a ``--workers`` argument, or compute the machine's default.

Run (``python -m tools.gate.worker_count``), it prints the machine's default
count and nothing else. ``--units N [--per-worker K]`` caps that count by the
job's size.

- :func:`worker_count` coerces a ``--workers`` argument without letting argparse
  exit 2. Every pre-commit gate shares it: ``affected.py`` sizes a thread pool
  over a SQLite index, ``houselint.py`` and ``unnecessary_assert_isinstance.py``
  a process pool over files, ``check_export_leaks.py`` a Copybara subprocess
  fan-out, and ``mutated.py`` a mutmut run.
- :func:`default_workers` is the count to use when none is given, and the one
  place it is computed: the pre-commit gates call it, and so does any tool that
  sizes a pool, so every job on a machine takes the same share of it.
  ``WORKERS_DEFAULT`` overrides it.

The formula::

  WORKERS = max(2, min(perf - 1, round(phys * 3/8)))
  WORKERS = max(1, min(WORKERS, available_RAM / 2 GiB))
  WORKERS = max(1, min(WORKERS, units // per_worker))     # only if units given

The job term is a CEILING too. ``units`` is the job's own count (test files,
mutants, packages); ``per_worker`` is how many units one worker needs to pay
its startup. A job too small to share runs serial instead of paying every
worker's startup for nothing. With no ``units`` the count is the machine's.

``phys`` counts physical cores (SMT deduped), ``perf`` those at the top clock.
Deduped on ``(physical_package_id, core_id)``, not ``core_id`` alone: core ids
restart at 0 per socket, so a dual-socket host collapsed to one socket's worth
of cores and ran half the workers it should. Both terms were FITTED to measured
optima, not derived.

Do NOT "simplify" this to a core count: physical cores, top frequency domain,
and nproc/2 each overshot a box. The EPYC curve is a sawtooth (24, 32 fast; 20,
28 slow -- likely CCD alignment), so a nearby value is not interchangeable.

The memory term is a CEILING, not a ratio. "Available" is RAM free at launch,
not installed RAM, so it binds only when memory is already short: a host keeps
N workers unless under 2N GiB is free. It exists for a laptop whose browser and
sibling agents already hold most of RAM: there the formula's 7 workers (~1-1.8
GB each once torch loads) pushed into swap and froze the desktop. It reads
``vm_stat`` (macOS: free + file-backed + purgeable, i.e. Activity Monitor's
free plus "Cached Files") or ``MemAvailable`` (Linux): memory reclaimable
without paging out an application. Do NOT add macOS's inactive pages. On a
laptop near its limit they are mostly idle application memory, freed only by
compressing or swapping it -- the stall this ceiling exists to prevent.
File-backed already includes speculative read-ahead; purgeable is anonymous
memory an app marked disposable.
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import os
import shutil
import subprocess


if TYPE_CHECKING:
    from collections.abc import Mapping


def worker_count(value: str) -> int:
    """Return ``value`` as a worker count, treating anything unusable as serial.

    NOT `type=int`. A caller that interpolates an unset count passes
    `--workers ''` -- argparse exits 2, and a gate reads that nonzero exit as an
    empty selection and falls back to its floor. A silent downgrade is worse
    than running serial, which is what an unsized caller asked for. Every
    override variable (`WORKERS_DEFAULT`, `WORKERS_<HOOK>`) is parsed here for
    the same reason.

    Which makes EVERY escaping exception the same bug, so the conversion is
    guarded by `try`, not by a predicate. `str.isdigit()` was one, and it
    admits precisely what `int()` refuses: `"\N{SUPERSCRIPT TWO}".isdigit()` is
    True. `lstrip("-")` was the other, stripping every leading dash, so
    `--workers=--4` passed the digit test and raised at the conversion. Both
    reached argparse as exit 2 from inside the callable written to prevent it.

    Args:
      value: The raw ``--workers`` argument, as argparse received it.

    Returns:
      workers: The requested count, never below 1.

    """
    try:
        return max(1, int(value))
    except ValueError:
        return 1


def default_workers(
    *,
    units: int | None = None,
    per_worker: int = 1,
    environ: Mapping[str, str] = os.environ,
    cpu: Path = Path("/sys/devices/system/cpu"),
    meminfo: Path = Path("/proc/meminfo"),
) -> int:
    """Return the worker count for a job on this machine, as the pre-commit gates size theirs.

    ``WORKERS_DEFAULT``, when set, wins (parsed by :func:`worker_count`).
    Otherwise the module docstring's formula: 3/8 of the physical cores,
    leaving a performance core free, capped at one worker per 2 GiB of
    available memory, then at one worker per ``per_worker`` units of work.

    Args:
      units: The job's size, in whatever it parallelizes over; None is unknown,
        and leaves the machine's count.
      per_worker: Units one worker needs to pay for its startup.
      environ: Environment ``WORKERS_DEFAULT`` is read from.
      cpu: Linux's CPU topology directory; absent, macOS ``sysctl`` is read.
      meminfo: Linux's memory report; absent, macOS ``vm_stat`` is read.

    Returns:
      workers: The count, at least 1.

    Raises:
      ValueError: ``per_worker`` is below 1.

    """
    if per_worker < 1:
        raise ValueError(f"per_worker={per_worker} must be at least 1.")
    override = environ.get("WORKERS_DEFAULT", "")
    if override:
        return worker_count(override)
    phys, perf = _cores(cpu)
    workers = max(2, min(perf - 1, (phys * 3 + 4) // 8))
    available = _available_mib(meminfo)
    if available is not None:
        workers = min(workers, max(1, available // 2048))
    if units is None:
        return workers
    return capped_workers(workers, units=units, per_worker=per_worker)


def capped_workers(workers: int, *, units: int, per_worker: int = 1) -> int:
    """Return ``workers`` capped at one per ``per_worker`` units of work, never below 1.

    The job-size term of the module docstring's formula, for a caller whose
    budget is already resolved (a ``--workers`` flag) and whose job size is
    known only once its inputs are read.

    Args:
      workers: The budget, from :func:`default_workers` or a ``--workers`` flag.
      units: The job's size, in whatever it parallelizes over.
      per_worker: Units one worker needs to pay for its startup.

    Returns:
      workers: The count, at least 1.

    Raises:
      ValueError: ``per_worker`` is below 1.

    """
    if per_worker < 1:
        raise ValueError(f"per_worker={per_worker} must be at least 1.")
    return max(1, min(workers, units // per_worker))


def main() -> int:
    """Print the worker count.

    Returns:
      exit_code: Always 0; argparse exits 2 on its own for a bad flag.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    print(default_workers(units=flags.units, per_worker=flags.per_worker))  # noqa: T201 -- The count is the whole output; a shell caller reads it.
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    units: int | None
    per_worker: int


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--units",
        type=int,
        default=None,
        help="Job size; caps the count at one worker per --per-worker units.",
    )
    parser.add_argument(
        "--per-worker",
        type=int,
        default=1,
        help="Units one worker needs to pay for its startup (default: 1).",
    )


def _cores(cpu: Path) -> tuple[int, int]:
    """Return physical cores and physical performance cores."""
    if (cpu / "cpu0" / "topology").is_dir():
        dirs = [d for d in cpu.glob("cpu[0-9]*") if (d / "topology").is_dir()]
        phys = len({_core_key(d) for d in dirs})
        freqs = {d: _read_int(d / "cpufreq" / "cpuinfo_max_freq") for d in dirs}
        top = max((f for f in freqs.values() if f is not None), default=None)
        if top is None:
            return phys, phys
        return phys, len({_core_key(d) for d, f in freqs.items() if f == top})
    phys = _sysctl("hw.physicalcpu") or os.cpu_count() or 1
    perf = 0
    for level in range(_sysctl("hw.nperflevels") or 1):
        if _sysctl_text(f"hw.perflevel{level}.name") == "Efficiency":
            continue
        perf += _sysctl(f"hw.perflevel{level}.physicalcpu") or 0
    return phys, perf if perf >= 1 else phys


def _core_key(cpu: Path) -> tuple[int | None, int | None]:
    """Return a logical CPU's (package, core): its physical core's identity."""
    topology = cpu / "topology"
    return (
        _read_int(topology / "physical_package_id"),
        _read_int(topology / "core_id"),
    )


# ``vm_stat`` wins wherever it is on PATH, so one memory figure drives the cap on
# macOS and on a Linux test host that fakes it.
def _available_mib(meminfo: Path) -> int | None:
    """Return memory available without paging, in MiB, or None if unknown."""
    if shutil.which("vm_stat") is not None:
        return _vm_stat_mib()
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return None


def _vm_stat_mib() -> int | None:
    """Return macOS free, purgeable and file-backed memory in MiB."""
    try:
        text = subprocess.run(
            ["vm_stat"],  # noqa: S607 -- A system tool found on PATH.
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    page = 0
    pages = 0
    for line in text.splitlines():
        if "page size of" in line:
            page = int(line.split()[7])
        elif line.startswith(("Pages free:", "Pages purgeable:", "File-backed pages:")):
            pages += int(line.split()[-1].rstrip("."))
    return pages * page // 1_048_576 if page else None


def _read_int(path: Path) -> int | None:
    """Return the integer in ``path``, or None if it is absent or unreadable."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _sysctl_text(name: str) -> str:
    """Return a macOS ``sysctl`` value, or '' where there is none."""
    try:
        return subprocess.run(  # noqa: S603 -- No shell; the argv is sysctl and a name this module spells.
            ["sysctl", "-n", name],  # noqa: S607 -- A system tool found on PATH.
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _sysctl(name: str) -> int | None:
    """Return a macOS ``sysctl`` integer, or None where there is none."""
    try:
        return int(_sysctl_text(name))
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
