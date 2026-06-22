Text search over the scholarly literature.

Default backend is Semantic Scholar (~200M papers, strong citation-graph
metadata). Set env var `SEMANTIC_SCHOLAR_API_KEY` for higher rate limits; omit
it to use unauthenticated API limits.
Alternative backends available for comparison when S2 coverage seems
thin: OpenAlex (~240M works, broader non-CS coverage), a fused mode that
merges both, and SearXNG science metasearch (adds PubMed, Crossref,
arXiv, OpenAIRE breadth beyond S2/OpenAlex).

Parameters:
  - `query` (required) — free-form text. Title words, author names,
    or venue fragments all work.
  - `source` — `"fused"` (default), `"s2"`, `"openalex"`, or
    `"searxng"`. Use `"openalex"` to sanity-check S2 results or reach
    beyond S2's coverage; use `"fused"` to dedup-merge S2 + OpenAlex (S2
    ordering preserved, OpenAlex-only hits appended at their OpenAlex
    rank); use `"searxng"` to widen to PubMed/Crossref/arXiv via the
    self-hosted SearXNG instance when S2 + OpenAlex miss a paper (e.g.
    biomedical or very recent work). SearXNG has no citation graph, so
    its hits carry no reference counts and `year_from`/`year_to`/
    `open_access_only` are applied best-effort client-side.
  - `limit` — cap on returned hits. Omit to let the backend decide its
    default page; no cap is imposed by the tool.
  - `year_from` / `year_to` — publication-year bounds, inclusive.
  - `open_access_only` (bool) — restrict to papers with a known OA PDF.
  - `abstract_chars` (int) — truncate abstracts. Omit for full text.

Output: one paper per line, in the same shape `PaperDetails` uses, with a
`sources:` tag indicating which backend(s) found the record. Results
where both backends agree (in `fused` mode) are labelled
`sources: s2,openalex`.

Filters deliberately omitted for v1: `fields_of_study` and `venue`.
S2 and OpenAlex disagree on taxonomies/shapes for these, and the
translation would be lossy. Use query text instead
(e.g. `"neural network cardiology"` in place of a field filter).

Workflow guidance:
  - Start here when you have a query but no identifier.
  - Pick the most promising hit and hand its id to `PaperDetails`
    (for metadata or citation-graph walk) or `PaperFetch` (for bytes).
  - If search is noisy, tighten with `year_from=` and `open_access_only=`
    before widening `limit`.
  - If S2 returns nothing for a query you expect to have hits (often
    for non-CS, older, or non-English work), retry with
    `source="openalex"` or `source="fused"`; for biomedical or very
    recent work that S2 + OpenAlex both miss, try `source="searxng"`.
  - SearXNG hits have no citation graph: to walk references/citations
    of a SearXNG result, take its DOI/arXiv id to `PaperDetails`.

No Google Scholar backend is provided. GS requires scraping with
captcha/proxy handling, which is out of scope — OpenAlex covers the
"broad academic search" niche without the fragility.

Sources:
  - Semantic Scholar: https://api.semanticscholar.org (`SEMANTIC_SCHOLAR_API_KEY`)
  - OpenAlex: https://api.openalex.org (`OPENALEX_API_KEY`)
  - SearXNG science metasearch (`SEARXNG_URL`)
