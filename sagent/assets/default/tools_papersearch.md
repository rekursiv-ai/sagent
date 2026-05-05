Text search over the scholarly literature.

Default backend is Semantic Scholar (~200M papers, strong citation-graph
metadata). Set env var `SEMANTIC_SCHOLAR_API_KEY` for higher rate limits; omit
it to use unauthenticated API limits.
Alternative backends available for comparison when S2 coverage seems
thin: OpenAlex (~240M works, broader non-CS coverage) and a fused mode
that merges both.

Parameters:
  - `query` (required) — free-form text. Title words, author names,
    or venue fragments all work.
  - `source` — `"s2"` (default), `"openalex"`, or `"fused"`. Use
    `"openalex"` to sanity-check S2 results or reach beyond S2's
    coverage; use `"fused"` to dedup-merge both indexes (S2 ordering
    preserved, OpenAlex-only hits appended at their OpenAlex rank).
  - `limit` (default 20, max 1000) — cap on returned hits.
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
    `source="openalex"` or `source="fused"`.

No Google Scholar backend is provided. GS requires scraping with
captcha/proxy handling, which is out of scope — OpenAlex covers the
"broad academic search" niche without the fragility.

Sources:
  - Semantic Scholar: https://api.semanticscholar.org
  - OpenAlex: https://api.openalex.org
