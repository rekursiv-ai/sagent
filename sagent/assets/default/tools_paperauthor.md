Author lookup on the Semantic Scholar Graph API.

Three modes dispatched by which fields are set:

  - `query` — search authors by name. Returns one line per candidate
    with id, name, h-index, citation count, paper count, and primary
    affiliation. Sorted by h-index descending so the most-prolific
    match surfaces first (names are often ambiguous).
  - `ids` (metadata) — full metadata for one or more authors: aliases,
    affiliations, homepage, h-index, citation / paper counts. Pass every
    id at once: they are resolved in ONE batched request (up to 500),
    far more efficient against the 1 request/second rate limit than one
    call per id. Results come back in input order.
  - `ids` (one id) + `operation="papers"` — that author's publications,
    in the same one-line-per-paper format `PaperSearch` and
    `PaperDetails` use. Supports `year_from` / `year_to` filters and
    `abstract_chars` truncation.

`ids` holds Semantic Scholar's opaque integer AUTHOR ids as strings
(e.g. `"1741101"`) — NOT paper ids. These are a different namespace
from the DOI / arXiv ids used by `PaperDetails` and `PaperFetch`; a
DOI or arXiv id will not resolve here. Agents typically obtain an
author id from `query` results (or from the author list on a paper)
and feed it back in.

Exactly one of `query` / `ids` is required. `operation`, `year_from`,
`year_to`, and `abstract_chars` all require a single id in `ids` plus
`operation="papers"`.

Other parameters:
  - `limit` (int) — max results. Omit to let Semantic Scholar return its
    default page; the tool imposes no cap.

Typical workflow:

```
PaperAuthor(query="Yoshua Bengio")               # → candidate list w/ ids
PaperAuthor(ids=["1741101", "2064160"])          # → batched author records
PaperAuthor(ids=["1741101"], operation="papers", # → recent work
            year_from=2022, limit=30)
```

Set env var `SEMANTIC_SCHOLAR_API_KEY` for higher Semantic Scholar rate limits;
omit it to use unauthenticated API limits.
