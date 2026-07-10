"""Tests for sagent.lib.web.paper.fuse (reciprocal-rank fusion + dedup)."""

from __future__ import annotations

from dataclasses import fields

from sagent.lib.web.paper.custom_types import PaperRecord
from sagent.lib.web.paper.fuse import fuse


def _rec(title: str, *, doi: str | None = None, source: str = "s2") -> PaperRecord:
    return PaperRecord(title=title, doi=doi, sources=(source,))


class TestMergeCompleteness:
    def test_merge_preserves_every_field_on_dedup(self) -> None:
        # A-WEB-005: _merge enumerated fields by hand and forgot several optional
        # ones (e.g. is_influential), zeroing them when two records dedup. Any
        # field populated only on the first record must survive the merge --
        # assert per-field so a NEWLY added field can't be silently dropped.
        first = PaperRecord(
            title="X",
            doi="10.1/a",
            is_influential=True,
            sources=("s2",),
        )
        second = PaperRecord(title="X", doi="10.1/a", sources=("openalex",))
        (merged,) = fuse([first], [second])
        for f in fields(PaperRecord):
            if f.name == "sources":
                continue  # sources are unioned, asserted elsewhere
            assert getattr(merged, f.name) == getattr(first, f.name), (
                f"_merge dropped field {f.name!r}"
            )


class TestFuse:
    def test_agreement_outranks_lone_top(self) -> None:
        # A paper both backends rank (#2 S2, #1 OpenAlex) must beat S2's lone #1.
        s2 = [_rec("solo", doi="10.1/solo"), _rec("shared", doi="10.1/shared")]
        oa = [_rec("shared", doi="10.1/shared", source="openalex")]
        out = fuse(s2, oa)
        assert out[0].doi == "10.1/shared"
        assert set(out[0].sources) == {"s2", "openalex"}

    def test_dedup_by_doi_merges_sources(self) -> None:
        s2 = [_rec("t", doi="10.1/x")]
        oa = [_rec("t", doi="10.1/x", source="openalex")]
        out = fuse(s2, oa)
        assert len(out) == 1
        assert set(out[0].sources) == {"s2", "openalex"}

    def test_dedup_by_title_when_no_doi(self) -> None:
        s2 = [_rec("Deep Learning!")]
        oa = [_rec("deep  learning", source="openalex")]
        out = fuse(s2, oa)
        assert len(out) == 1

    def test_openalex_only_still_ranked(self) -> None:
        # A throttled S2 (empty) degrades to OpenAlex-ranked results, not nothing.
        out = fuse([], [_rec("a", source="openalex"), _rec("b", source="openalex")])
        assert [r.title for r in out] == ["a", "b"]

    def test_s2_wins_equal_rank_tie(self) -> None:
        # Same-rank single-backend papers break in S2's favor (higher weight).
        out = fuse(
            [_rec("s2top", doi="10.1/s")],
            [_rec("oatop", doi="10.1/o", source="openalex")],
        )
        assert out[0].doi == "10.1/s"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
