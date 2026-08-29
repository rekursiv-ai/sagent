from __future__ import annotations

import copy
import pickle

from sagent.lib.absent import ABSENT, Absent


def test_absent_is_singleton():
    assert Absent() is ABSENT


def test_absent_copy_preserves_singleton():
    assert copy.copy(ABSENT) is ABSENT
    assert copy.deepcopy(ABSENT) is ABSENT


def test_absent_pickle_preserves_singleton():
    assert pickle.loads(pickle.dumps(ABSENT)) is ABSENT


def test_missing_type_repr():
    assert repr(ABSENT) == "ABSENT"


def test_missing_type_bool():
    assert bool(ABSENT) is False
    assert not ABSENT


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
