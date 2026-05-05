Bibliographic lookup and citation-graph traversal for scholarly papers.
Backed by the Semantic Scholar Graph API. Set env var `SEMANTIC_SCHOLAR_API_KEY`
for higher rate limits; omit it to use unauthenticated API limits.

Operations (dispatched by which fields are set):
  - Plain metadata lookup — pass `id` alone, omit `operation`. Returns
    a block with title, authors, year, venue, abstract, citation and
    reference counts, and an open-access PDF URL when available.
  - References — `id` + `operation="references"`. Returns the papers
    this one cites (backward edges of the citation graph).
  - Citations — `id` + `operation="citations"`. Returns the papers
    that cite this one (forward edges).

`id` accepts either a DOI (`10.xxxx/yyy`, optionally prefixed with
`doi:` or `https://doi.org/`) or an arXiv id (`2106.15928`,
`arXiv:2106.15928`, or legacy `hep-th/9901001`). PMIDs and raw S2 ids
are not accepted — they are niche enough that overloading `id` with
them would hurt more than help.

Citation-only filters:
  - `influential_only` (bool) — restrict to citations S2 flags as
    substantively building on the paper (its unique quality signal).
    Default false.
  - `year_from` (int) — drop citations published before this year.
    Useful for "who built on this since 2022" queries.

Shared:
  - `limit` (default 100, max 1000) — only applies to references and
    citations; ignored for plain metadata lookup.
  - `abstract_chars` (int) — trim abstracts to this many characters
    across every record in the response. Omit for full abstracts.

Output:
  - Metadata lookup returns a multi-line block (one field per line).
  - References / citations return one paper per line; when the total
    exceeds `limit`, a "showing N of M" footer tells the agent to
    tighten filters rather than paginate blindly.

Workflow guidance:
  - If you have a query but no id, call `PaperSearch` first to find
    the seed paper, then feed the returned id here.
  - To trace a paper's intellectual lineage, start with
    `operation="references"` on the seed; pick promising entries and
    recurse on their ids.
  - To find descendants, use `operation="citations"` with
    `influential_only=true` first (narrow, high-signal), then broaden
    if you need more.
  - To pull the PDF bytes of any paper, hand the id to `PaperFetch`.
