from __future__ import annotations

from sagent.providers.lib.id_remap import IdRemapper


class TestIdRemapper:
    def test_first_call_returns_prefix_zero(self) -> None:
        r = IdRemapper("msg_")
        assert r.map("original") == "msg_0"

    def test_second_distinct_id_returns_prefix_one(self) -> None:
        r = IdRemapper("msg_")
        r.map("a")
        assert r.map("b") == "msg_1"

    def test_same_id_is_idempotent(self) -> None:
        r = IdRemapper("msg_")
        first = r.map("x")
        second = r.map("x")
        assert first == second == "msg_0"

    def test_different_prefixes(self) -> None:
        a = IdRemapper("a_")
        b = IdRemapper("b_")
        assert a.map("id") == "a_0"
        assert b.map("id") == "b_0"

    def test_counter_increments_across_ids(self) -> None:
        r = IdRemapper("n")
        results = [r.map(f"id_{i}") for i in range(5)]
        assert results == ["n0", "n1", "n2", "n3", "n4"]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
