from __future__ import annotations

import sys

import pytest

from sagent.lib.testing import runner


def test_test_main_exits_with_pytest_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_main(args: list[str]) -> int:
        captured.append(args)
        return 7

    monkeypatch.setattr(pytest, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["runner", "-q"])

    with pytest.raises(SystemExit) as excinfo:
        runner.test_main("sample_test.py")

    assert excinfo.value.code == 7
    assert captured == [
        [
            "sample_test.py",
            "-v",
            "-s",
            "-W",
            "ignore::pytest.PytestAssertRewriteWarning",
            "-q",
        ],
    ]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
