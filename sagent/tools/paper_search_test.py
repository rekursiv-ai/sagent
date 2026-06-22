"""Tests for ``tools.paper_search``: S2 + OpenAlex search + fusion."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import asyncio
import json


if TYPE_CHECKING:
    import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.fetch import FetchError
from sagent.lib.web.search import PaperResult
from sagent.tools.paper_common import PaperRecord
from sagent.tools.paper_search import (
    PaperSearch,
    _dedup_key,
    _fuse,
    _merge,
    _normalize_title,
    _openalex_filter,
    _openalex_work_to_record,
    _s2_year_param,
    _searxng_paper_to_record,
)


def test_paper_search_metadata() -> None:
    t = PaperSearch()
    assert t.name == "PaperSearch"
    assert t.tool_id == "application/x-tool-papersearch"


def test_summary_default() -> None:
    t = PaperSearch()
    out = t.summary({"query": "transformers"})
    assert "PaperSearch" in out
    assert "transformers" in out


def test_summary_with_source() -> None:
    out = PaperSearch().summary({"query": "x", "source": "openalex"})
    assert "(openalex)" in out


def test_summary_truncates_long_query() -> None:
    out = PaperSearch().summary({"query": "q" * 80})
    assert "..." in out


def test_summary_empty_query() -> None:
    assert PaperSearch().summary({"query": ""}) == "PaperSearch"


def test_prompt_empty() -> None:
    assert PaperSearch().prompt() == ""


def test_s2_year_param_both() -> None:
    assert _s2_year_param(2020, 2022) == "2020-2022"


def test_s2_year_param_from_only() -> None:
    assert _s2_year_param(2020, None) == "2020-"


def test_s2_year_param_to_only() -> None:
    assert _s2_year_param(None, 2022) == "-2022"


def test_s2_year_param_neither() -> None:
    assert _s2_year_param(None, None) is None


def test_openalex_filter_no_bounds() -> None:
    assert (
        _openalex_filter(year_from=None, year_to=None, open_access_only=False) is None
    )


def test_openalex_filter_year_from() -> None:
    out = _openalex_filter(year_from=2020, year_to=None, open_access_only=False)
    assert out == "from_publication_date:2020-01-01"


def test_openalex_filter_open_access() -> None:
    out = _openalex_filter(year_from=None, year_to=None, open_access_only=True)
    assert out == "open_access.is_oa:true"


def test_openalex_filter_combined() -> None:
    out = _openalex_filter(year_from=2020, year_to=2022, open_access_only=True)
    assert out is not None
    assert "from_publication_date:2020-01-01" in out
    assert "to_publication_date:2022-12-31" in out
    assert "open_access.is_oa:true" in out


def test_openalex_work_to_record_full() -> None:
    work: MutableJSON = {
        "title": "Attention",
        "authorships": [
            {"author": {"display_name": "A"}},
            {"author": {"display_name": "B"}},
            {"author": {}},
        ],
        "publication_year": 2017,
        "primary_location": {"source": {"display_name": "NeurIPS"}},
        "doi": "https://doi.org/10.0/abc",
        "ids": {"arxiv": "https://arxiv.org/abs/1706.03762"},
        "abstract_inverted_index": {"hello": [0], "world": [1]},
        "cited_by_count": 100_000,
        "referenced_works_count": 50,
        "open_access": {"oa_url": "https://oa/x"},
    }
    rec = _openalex_work_to_record(work)
    assert rec.title == "Attention"
    assert rec.authors == ("A", "B")
    assert rec.year == 2017
    assert rec.venue == "NeurIPS"
    assert rec.doi == "10.0/abc"
    assert rec.arxiv_id == "1706.03762"
    assert rec.abstract == "hello world"
    assert rec.citation_count == 100_000
    assert rec.open_access_pdf == "https://oa/x"
    assert rec.sources == ("openalex",)


def test_openalex_work_to_record_sparse() -> None:
    rec = _openalex_work_to_record({"display_name": "Bare"})
    assert rec.title == "Bare"
    assert rec.authors == ()
    assert rec.doi is None


def test_openalex_work_to_record_doi_no_prefix() -> None:
    rec = _openalex_work_to_record({"title": "X", "doi": "10.1/y"})
    assert rec.doi == "10.1/y"


def test_normalize_title_strips_punct_lowercase() -> None:
    assert _normalize_title("Hello, World!") == "hello world"


def test_normalize_title_collapses_whitespace() -> None:
    assert _normalize_title("a   b\t\nc") == "a b c"


def test_dedup_key_prefers_doi() -> None:
    rec = PaperRecord(title="T", doi="10.1/X")
    assert _dedup_key(rec) == "doi:10.1/x"


def test_dedup_key_falls_back_to_title() -> None:
    rec = PaperRecord(title="Some Paper!")
    assert _dedup_key(rec) == "title:some paper"


def test_merge_prefers_first() -> None:
    a = PaperRecord(title="A", year=2020, sources=("s2",))
    b = PaperRecord(title="A", year=2019, doi="10.1/x", sources=("openalex",))
    out = _merge(a, b)
    assert out.year == 2020  # first wins
    assert out.doi == "10.1/x"  # filled from second
    assert out.sources == ("s2", "openalex")


def test_fuse_rewards_agreement_across_backends() -> None:
    # RRF: a paper both backends rank highly must outrank a paper only one
    # backend has, even when the single-backend paper is that backend's #1.
    # AGREE is S2 #2 and OA #1; SOLO is S2 #1 but absent from OA.
    s2 = [
        PaperRecord(title="Solo", doi="10.1/solo", sources=("s2",)),
        PaperRecord(title="Agree", doi="10.1/agree", sources=("s2",)),
    ]
    oa = [PaperRecord(title="Agree", doi="10.1/agree", sources=("openalex",))]
    fused = _fuse(s2, oa)
    assert fused[0].doi == "10.1/agree"  # cross-backend agreement wins
    assert {r.doi for r in fused} == {"10.1/agree", "10.1/solo"}


def test_fuse_upweights_s2_on_single_backend_tie() -> None:
    # When two papers each appear in exactly one backend at the same rank
    # (S2 #1 vs OpenAlex #1), the S2 paper ranks first: S2's relevance is more
    # precise, so it carries a higher per-engine weight.
    s2 = [PaperRecord(title="FromS2", doi="10.1/s2", sources=("s2",))]
    oa = [PaperRecord(title="FromOA", doi="10.1/oa", sources=("openalex",))]
    fused = _fuse(s2, oa)
    assert fused[0].doi == "10.1/s2"
    assert fused[1].doi == "10.1/oa"


def test_fuse_strong_openalex_hit_interleaves_not_buried() -> None:
    # A strong OpenAlex-only hit (its #1) must interleave into S2's top, not
    # sink below the entire S2 list. With k=60 the RRF curve is so flat that
    # all of S2's top ~27 outrank OpenAlex #1, burying cross-pollinated hits;
    # a smaller k keeps the per-engine weights meaningful. Contract: OpenAlex #1
    # outranks the tail of a 10-deep S2 list.
    s2 = [
        PaperRecord(title=f"S{i}", doi=f"10.1/s{i}", sources=("s2",)) for i in range(10)
    ]
    oa = [PaperRecord(title="OAtop", doi="10.1/oatop", sources=("openalex",))]
    fused = _fuse(s2, oa)
    pos = {r.doi: i for i, r in enumerate(fused)}
    # OpenAlex #1 must rank above S2's #10 (last) -- it is NOT buried at the end.
    assert pos["10.1/oatop"] < pos["10.1/s9"]


def test_fuse_includes_openalex_only_hits() -> None:
    # Something every time: an OpenAlex-only paper still appears (ranked by its
    # single-backend RRF score), so a fused result is never just S2's list.
    s2 = [PaperRecord(title="S", doi="10.1/s", sources=("s2",))]
    oa = [PaperRecord(title="O", doi="10.1/o", sources=("openalex",))]
    fused = _fuse(s2, oa)
    assert {r.doi for r in fused} == {"10.1/s", "10.1/o"}


def test_fuse_empty_s2_returns_openalex_ranked() -> None:
    # S2 down/throttled -> fused degrades to OpenAlex alone, in OpenAlex rank
    # order, not an empty or error result.
    oa = [
        PaperRecord(title="First", doi="10.1/1", sources=("openalex",)),
        PaperRecord(title="Second", doi="10.1/2", sources=("openalex",)),
    ]
    fused = _fuse([], oa)
    assert [r.doi for r in fused] == ["10.1/1", "10.1/2"]


def test_fuse_dedups_by_doi() -> None:
    s2 = [PaperRecord(title="X", doi="10.1/a", citation_count=100, sources=("s2",))]
    oa = [PaperRecord(title="Y", doi="10.1/a", abstract="abs", sources=("openalex",))]
    fused = _fuse(s2, oa)
    assert len(fused) == 1
    assert fused[0].title == "X"
    assert fused[0].abstract == "abs"
    assert "s2" in fused[0].sources
    assert "openalex" in fused[0].sources


def test_run_empty_query() -> None:
    result = asyncio.run(PaperSearch().run({"query": "  "}))
    assert result.is_error
    assert "required" in result.content


def test_run_invalid_source() -> None:
    result = asyncio.run(PaperSearch().run({"query": "x", "source": "bing"}))
    assert result.is_error
    assert "Invalid source" in result.content


def _s2_search_payload() -> bytes:
    return json.dumps(
        {
            "total": 1,
            "data": [
                {
                    "title": "Hit",
                    "year": 2020,
                    "externalIds": {"DOI": "10.0/x"},
                    "authors": [{"name": "A"}],
                }
            ],
        }
    ).encode()


def test_run_s2_success() -> None:
    # Pin source="s2": default fused would also hit the (unpatched) OpenAlex.
    with patch(
        "sagent.tools.paper_common.fetch",
        return_value=_s2_search_payload(),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "transformers", "source": "s2"})
        )
    assert not result.is_error
    assert "Hit" in result.content


def test_run_no_results() -> None:
    # Pin source="s2" so only the (patched, empty) S2 path runs; default fused
    # would also query the real OpenAlex.
    payload = json.dumps({"total": 0, "data": []}).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(PaperSearch().run({"query": "nothing", "source": "s2"}))
    assert result.content.startswith("(no results)")


def test_run_no_results_multiterm_explains_and_narrowing() -> None:
    # A multi-term query that returns nothing is most often AND-narrowing: both
    # backends require EVERY term in title/abstract, so one rare term zeroes the
    # result. The empty must explain this and tell the agent to drop terms --
    # the only cross-backend broadening lever (operators aren't portable).
    s2_empty = json.dumps({"total": 0, "data": []}).encode()
    oa_empty = json.dumps({"meta": {"count": 0}, "results": []}).encode()

    def fake(**kw: object) -> bytes:
        url = str(kw.get("url", ""))
        return oa_empty if "openalex" in url else s2_empty

    with (
        patch("sagent.tools.paper_common.fetch", side_effect=fake),
        patch("sagent.tools.paper_search.fetch", side_effect=fake),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "object-centric slot attention ARC"})
        )
    assert "(no results)" in result.content
    assert "every query term" in result.content.lower()
    assert "drop" in result.content.lower()


def test_run_fused_one_errors_other_empty_is_not_error() -> None:
    # S2 throttled (error) + OpenAlex clean-empty must NOT surface as an error:
    # one backend genuinely answered (empty), so it's a real AND-narrowing empty
    # with guidance, not a failure hidden behind the sibling's throttle.
    err = FetchError(url="u", status=429, headers={}, body=b"")
    oa_empty = json.dumps({"meta": {"count": 0}, "results": []}).encode()

    def fake(**kw: object) -> bytes:
        if "openalex" in str(kw.get("url", "")):
            return oa_empty
        raise err

    with (
        patch("sagent.tools.paper_common.fetch", side_effect=fake),
        patch("sagent.tools.paper_search.fetch", side_effect=fake),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "object-centric slot attention ARC"})
        )
    assert not result.is_error
    assert "(no results)" in result.content
    assert "drop" in result.content.lower()  # AND-narrowing guidance present


def test_run_no_results_single_term_no_and_hint() -> None:
    # A single-term empty isn't an AND-narrowing problem; no drop-terms advice.
    s2_empty = json.dumps({"total": 0, "data": []}).encode()
    oa_empty = json.dumps({"meta": {"count": 0}, "results": []}).encode()

    def fake(**kw: object) -> bytes:
        url = str(kw.get("url", ""))
        return oa_empty if "openalex" in url else s2_empty

    with (
        patch("sagent.tools.paper_common.fetch", side_effect=fake),
        patch("sagent.tools.paper_search.fetch", side_effect=fake),
    ):
        result = asyncio.run(PaperSearch().run({"query": "xyzzynonsense"}))
    assert "drop" not in result.content.lower()


def test_run_no_results_s2_appends_author_hint() -> None:
    # An EXPLICIT source="s2" returning nothing must nudge toward fused: an
    # author-surname query silently zero-hits on S2 but resolves via OpenAlex
    # (live 2026-06-19 ARC-AGI survey reference walk). The hint fires only for
    # explicit s2 now, since fused is the default and already covers it.
    payload = json.dumps({"total": 0, "data": []}).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperSearch().run({"query": "Andrews Sparks", "source": "s2"})
        )
    assert "(no results)" in result.content
    assert 'source="fused"' in result.content
    assert "author" in result.content.lower()


def test_run_no_results_fused_no_author_hint() -> None:
    # Fused already includes OpenAlex, so the s2-only author-name hint would be
    # circular and must NOT appear. (The generic AND-narrowing note may appear
    # for a multi-term query -- that is correct and separate.)
    s2_empty = json.dumps({"total": 0, "data": []}).encode()
    oa_empty = json.dumps({"meta": {"count": 0}, "results": []}).encode()

    def fake(**kw: object) -> bytes:
        url = str(kw.get("url", ""))
        return oa_empty if "openalex" in url else s2_empty

    with (
        patch("sagent.tools.paper_common.fetch", side_effect=fake),
        patch("sagent.tools.paper_search.fetch", side_effect=fake),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "nothing here", "source": "fused"})
        )
    assert result.content.startswith("(no results)")
    assert "author names" not in result.content  # no circular fused-author hint


def test_run_hits_no_author_hint() -> None:
    # The hint is for empty results only; a populated result stays clean.
    # Pin source="s2" (the hint path is s2-only) so fused doesn't hit OpenAlex.
    with patch(
        "sagent.tools.paper_common.fetch",
        return_value=_s2_search_payload(),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "transformers", "source": "s2"})
        )
    assert 'source="fused"' not in result.content


def test_openalex_uses_interactive_timeout() -> None:
    # The fused backend queries OpenAlex alongside S2; its timeout must be
    # bounded too, or one leg can silently hang an interactive turn even after
    # the other returned (live 2026-06-19 idle-hang regression guard).
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return b'{"meta": {"count": 0}, "results": []}'

    with patch("sagent.tools.paper_search.fetch", side_effect=fake_fetch):
        _ = asyncio.run(
            PaperSearch().run({"query": "timeout_probe", "source": "openalex"})
        )
    timeout = captured["timeout_sec"]
    assert isinstance(timeout, float)
    assert timeout <= 15.0


def test_openalex_uses_precise_title_abstract_filter() -> None:
    # OpenAlex's broad ``search=`` param is noisy (returns thousands of loosely
    # related works). Use the ``title_and_abstract.search`` filter for precision
    # so the OpenAlex-only (S2-throttled) path returns relevant hits.
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return b'{"meta": {"count": 3}, "results": [{"title": "Hit", "doi": "10/x"}]}'

    with patch("sagent.tools.paper_search.fetch", side_effect=fake_fetch):
        _ = asyncio.run(
            PaperSearch().run({"query": "arc reasoning", "source": "openalex"})
        )
    params = cast(dict[str, object], captured["params"])
    assert "search" not in params  # not the broad param
    flt = str(params.get("filter", ""))
    # Unquoted: OpenAlex treats unquoted terms as AND-of-terms (the recall we
    # want). Quoting would force exact-PHRASE match -> near-zero hits.
    assert "title_and_abstract.search:arc reasoning" in flt
    assert '"' not in flt


def test_openalex_query_with_comma_does_not_break_filter() -> None:
    # A comma in the query is OpenAlex's filter separator -> bare comma yields
    # HTTP 400. Replace it with a space (commas aren't a search operator), which
    # keeps AND-of-terms semantics. Quoting would avoid the 400 too but at the
    # cost of phrase-match (recall collapse), so it is NOT used.
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return b'{"meta": {"count": 0}, "results": []}'

    with patch("sagent.tools.paper_search.fetch", side_effect=fake_fetch):
        _ = asyncio.run(
            PaperSearch().run(
                {"query": "deep learning, attention", "source": "openalex"}
            )
        )
    flt = str(cast(dict[str, object], captured["params"])["filter"])
    # Comma gone (replaced by space), value unquoted, AND-semantics intact.
    assert "title_and_abstract.search:deep learning  attention" in flt
    assert "," not in flt.split("title_and_abstract.search:")[1]
    assert '"' not in flt


def test_openalex_sends_api_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # A configured OPENALEX_API_KEY must be sent (raises the daily credit
    # budget); otherwise the key in the user's env is silently ignored.
    monkeypatch.setenv("OPENALEX_API_KEY", "testkey123")
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return b'{"meta": {"count": 0}, "results": []}'

    with patch("sagent.tools.paper_search.fetch", side_effect=fake_fetch):
        _ = asyncio.run(
            PaperSearch().run({"query": "apikey_set_probe", "source": "openalex"})
        )
    params = cast(dict[str, object], captured["params"])
    assert params.get("api_key") == "testkey123"


def test_openalex_omits_api_key_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return b'{"meta": {"count": 0}, "results": []}'

    with patch("sagent.tools.paper_search.fetch", side_effect=fake_fetch):
        _ = asyncio.run(
            PaperSearch().run({"query": "apikey_unset_probe", "source": "openalex"})
        )
    assert "api_key" not in cast(dict[str, object], captured["params"])


def test_openalex_does_not_fall_back_to_fulltext() -> None:
    # An empty title_and_abstract result must NOT fall back to the broad
    # fulltext ``search`` param: its relevance_score is citation-dominated and
    # surfaces high-cite off-topic reviews. One request, never the broad param;
    # a clean empty is correct (the fused default still covers it via S2).
    calls: list[dict[str, object]] = []

    def fake_fetch(**kw: object) -> bytes:
        calls.append(dict(cast(dict[str, object], kw.get("params") or {})))
        return b'{"meta": {"count": 0}, "results": []}'

    with patch("sagent.tools.paper_search.fetch", side_effect=fake_fetch):
        result = asyncio.run(
            PaperSearch().run(
                {"query": "very specific long query", "source": "openalex"}
            )
        )
    assert len(calls) == 1  # single request, no fulltext fallback
    assert "title_and_abstract" in str(calls[0].get("filter", ""))
    assert "search" not in calls[0]
    assert result.content.startswith("(no results)")


def test_default_source_is_fused() -> None:
    # Default now queries BOTH backends and fuses, so a throttled/missing S2
    # still yields OpenAlex results -- "something every time".
    oa_payload = json.dumps(
        {"meta": {"count": 1}, "results": [{"title": "OA", "doi": "10.0/oa"}]}
    ).encode()
    with (
        patch(
            "sagent.tools.paper_common.fetch",
            return_value=_s2_search_payload(),
        ) as s2_fetch,
        patch(
            "sagent.tools.paper_search.fetch",
            return_value=oa_payload,
        ) as oa_fetch,
    ):
        result = asyncio.run(PaperSearch().run({"query": "default_source_probe"}))
    assert not result.is_error
    assert s2_fetch.call_count == 1
    assert oa_fetch.call_count == 1  # OpenAlex queried too, by default


def test_default_source_survives_s2_throttle() -> None:
    # The whole point of fused-default: a 429 from S2 must not blank the result;
    # OpenAlex carries it.
    err = FetchError(url="u", status=429, headers={}, body=b"")
    oa_payload = json.dumps(
        {"meta": {"count": 1}, "results": [{"title": "Rescued", "doi": "10.0/r"}]}
    ).encode()
    with (
        patch("sagent.tools.paper_common.fetch", side_effect=err),
        patch(
            "sagent.tools.paper_search.fetch",
            return_value=oa_payload,
        ),
    ):
        result = asyncio.run(PaperSearch().run({"query": "throttle_probe"}))
    assert not result.is_error
    assert "Rescued" in result.content


def test_run_rejects_zero_abstract_chars() -> None:
    # Schema declares minimum 1; 0 must be rejected, not silently treated as
    # "no truncation" (matches validate_limit's reject-0 contract).
    result = asyncio.run(PaperSearch().run({"query": "x", "abstract_chars": 0}))
    assert result.is_error
    assert "abstract_chars" in result.content


def test_run_openalex_path() -> None:
    payload = json.dumps(
        {
            "meta": {"count": 1},
            "results": [
                {"title": "OA Hit", "doi": "10.0/oa", "publication_year": 2021}
            ],
        }
    ).encode()
    with patch("sagent.tools.paper_search.fetch", return_value=payload):
        result = asyncio.run(
            PaperSearch().run({"query": "openalex_path_query", "source": "openalex"}),
        )
    assert "OA Hit" in result.content


def test_run_openalex_timeout_returns_error() -> None:
    # OpenAlex shares the same fetch path as S2: a socket timeout raises
    # TimeoutError/OSError (not FetchError). With the lowered timeout it must
    # render as a ToolResult, not escape (parity with the S2 fix).
    with patch(
        "sagent.tools.paper_search.fetch",
        side_effect=TimeoutError("slow"),
    ):
        result = asyncio.run(PaperSearch().run({"query": "x", "source": "openalex"}))
    assert result.is_error
    assert "OpenAlex" in result.content


def test_run_openalex_connection_error_returns_error() -> None:
    with patch(
        "sagent.tools.paper_search.fetch",
        side_effect=OSError("refused"),
    ):
        result = asyncio.run(PaperSearch().run({"query": "x", "source": "openalex"}))
    assert result.is_error


def test_run_openalex_rate_limit_returns_error() -> None:
    err = FetchError(
        url="https://api.openalex.org/works",
        status=429,
        headers={},
        body=b"slow down",
    )
    with patch("sagent.tools.paper_search.fetch", side_effect=err):
        result = asyncio.run(
            PaperSearch().run(
                {"query": "openalex_rate_limit_query", "source": "openalex"}
            ),
        )
    assert result.is_error
    assert "rate limit" in result.content.lower()


def test_run_openalex_other_http_error() -> None:
    err = FetchError(
        url="https://api.openalex.org/works",
        status=500,
        headers={},
        body=b"boom",
    )
    with patch("sagent.tools.paper_search.fetch", side_effect=err):
        result = asyncio.run(
            PaperSearch().run(
                {"query": "openalex_other_http_query", "source": "openalex"}
            ),
        )
    assert result.is_error
    assert "500" in result.content


def test_run_openalex_invalid_json_returns_error() -> None:
    # A non-JSON 200 body must surface a ToolResult error, not crash the tool.
    with patch("sagent.tools.paper_search.fetch", return_value=b"<html>"):
        result = asyncio.run(
            PaperSearch().run(
                {"query": "openalex_bad_json_query", "source": "openalex"}
            ),
        )
    assert result.is_error
    assert "invalid JSON" in result.content


def test_run_fused_combines_backends() -> None:
    s2_payload = _s2_search_payload()
    openalex_payload = json.dumps(
        {
            "meta": {"count": 1},
            "results": [{"title": "Other", "doi": "10.0/other"}],
        }
    ).encode()

    def s2_fetch(**_kw: object) -> bytes:
        return s2_payload

    def oa_fetch(**_kw: object) -> bytes:
        return openalex_payload

    with (
        patch("sagent.tools.paper_common.fetch", side_effect=s2_fetch),
        patch("sagent.tools.paper_search.fetch", side_effect=oa_fetch),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "fused_combines_query", "source": "fused"}),
        )
    assert "Hit" in result.content
    assert "Other" in result.content


def test_run_fused_all_fail_returns_error() -> None:
    err = FetchError(url="u", status=500, headers={}, body=b"")
    with (
        patch("sagent.tools.paper_common.fetch", side_effect=err),
        patch("sagent.tools.paper_search.fetch", side_effect=err),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "fused_all_fail_query", "source": "fused"}),
        )
    assert result.is_error


def test_run_caches_results() -> None:
    # Pin source="s2" so the cache test is hermetic (fused would also hit the
    # unpatched OpenAlex) and the call-count assertion targets one backend.
    payload = _s2_search_payload()
    with patch("sagent.tools.paper_common.fetch", return_value=payload) as mock_fetch:
        args = {"query": "uniquecachetestkey1", "source": "s2"}
        _ = asyncio.run(PaperSearch().run(args))
        _ = asyncio.run(PaperSearch().run(args))
    assert mock_fetch.call_count == 1


def test_fused_partial_failure_is_not_cached() -> None:
    # A fused result where one backend ERRORED (S2 429) and the other answered
    # must NOT be cached -- otherwise a later query returns the degraded
    # one-backend result and never retries the recovered backend.
    oa_payload = json.dumps(
        {"meta": {"count": 1}, "results": [{"title": "OnlyOA", "doi": "10.0/oa"}]}
    ).encode()
    s2_payload = json.dumps(
        {
            "total": 1,
            "data": [
                {"title": "NowS2", "externalIds": {"DOI": "10.0/s2"}, "authors": []}
            ],
        }
    ).encode()
    q = "fused_partial_cache_probe_xyzzy"

    # First call: S2 throttled, OpenAlex answers -> partial result, must not cache.
    with (
        patch(
            "sagent.tools.paper_common.fetch",
            side_effect=FetchError(url="u", status=429, headers={}, body=b""),
        ),
        patch(
            "sagent.tools.paper_search.fetch",
            return_value=oa_payload,
        ),
    ):
        first = asyncio.run(PaperSearch().run({"query": q}))
    assert "OnlyOA" in first.content

    # Second call: S2 recovered -> must re-query S2 (cache miss), not serve stale.
    with (
        patch(
            "sagent.tools.paper_common.fetch",
            return_value=s2_payload,
        ) as s2_fetch,
        patch(
            "sagent.tools.paper_search.fetch",
            return_value=oa_payload,
        ),
    ):
        second = asyncio.run(PaperSearch().run({"query": q}))
    assert s2_fetch.call_count == 1
    assert "NowS2" in second.content


def test_searxng_paper_to_record_maps_fields() -> None:
    rec = _searxng_paper_to_record(
        PaperResult(
            url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            snippet="We propose the Transformer.",
            authors=("Vaswani", "Shazeer"),
            journal="NeurIPS",
            doi="10.5555/3295222",
            pdf_url="https://arxiv.org/pdf/1706.03762",
            published=datetime(2017, 6, 12),  # noqa: DTZ001 -- naive ok in test
            citations=100,
        )
    )
    assert rec.title == "Attention Is All You Need"
    assert rec.authors == ("Vaswani", "Shazeer")
    assert rec.year == 2017
    assert rec.venue == "NeurIPS"
    assert rec.doi == "10.5555/3295222"
    assert rec.arxiv_id == "1706.03762"  # recovered from URL
    assert rec.citation_count == 100
    assert rec.open_access_pdf == "https://arxiv.org/pdf/1706.03762"
    assert rec.sources == ("searxng",)


def test_searxng_paper_to_record_drops_unparseable_doi() -> None:
    rec = _searxng_paper_to_record(
        PaperResult(url="https://x", title="T", snippet="", doi="not-a-doi")
    )
    assert rec.doi is None


def test_run_searxng_source() -> None:
    hits = [
        PaperResult(
            url="https://doi.org/10.1/x",
            title="Hit One",
            snippet="abstract",
            doi="10.1/x",
        )
    ]
    with patch("sagent.tools.paper_search.searxng", return_value=hits) as mock:
        result = asyncio.run(
            PaperSearch().run({"query": "transformers", "source": "searxng"})
        )
    assert not result.is_error
    assert "Hit One" in result.content
    assert mock.call_args.kwargs["categories"] == "science"


def test_run_searxng_source_year_filter() -> None:
    hits = [
        PaperResult(
            url="https://a",
            title="Old",
            snippet="",
            published=datetime(2010, 1, 1),  # noqa: DTZ001 -- naive ok in test
        ),
        PaperResult(
            url="https://b",
            title="New",
            snippet="",
            published=datetime(2023, 1, 1),  # noqa: DTZ001 -- naive ok in test
        ),
    ]
    with patch("sagent.tools.paper_search.searxng", return_value=hits):
        result = asyncio.run(
            PaperSearch().run({"query": "x", "source": "searxng", "year_from": 2020})
        )
    assert "New" in result.content
    assert "Old" not in result.content


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
