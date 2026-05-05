Author lookup on the Semantic Scholar Graph API.

Three modes dispatched by which fields are set:

  - `query` — search authors by name. Returns one line per candidate
    with id, name, h-index, citation count, paper count, and primary
    affiliation. Sorted by h-index descending so the most-prolific
    match surfaces first (names are often ambiguous).
  - `id` alone — full metadata for one author: all aliases, all
    affiliations, homepage, h-index, citation / paper counts.
  - `id` + `operation="papers"` — that author's publications, in the
    same one-line-per-paper format `PaperSearch` and `PaperDetails`
    use. Supports `year_from` / `year_to` filters and `abstract_chars`
    truncation.

`id` is Semantic Scholar's opaque integer author id as a string
(e.g. `"1741101"`). Agents typically get one from `query` results
and feed it back in.

Exactly one of `query` / `id` is required. `operation`, `year_from`,
`year_to`, and `abstract_chars` all require `id` + `operation="papers"`.

Other parameters:
  - `limit` (int) — max results. Defaults to 20 for search, 100 for
    a papers list. Capped at 1000.

Typical workflow:

```
PaperAuthor(query="Yoshua Bengio")              # → candidate list w/ ids
PaperAuthor(id="1741101")                       # → full author record
PaperAuthor(id="1741101", operation="papers",   # → recent work
            year_from=2022, limit=30)
```

Set env var `SEMANTIC_SCHOLAR_API_KEY` for higher Semantic Scholar rate limits;
omit it to use unauthenticated API limits.
