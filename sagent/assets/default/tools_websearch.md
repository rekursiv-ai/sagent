Web search with results inline. Single round-trip.

Use for time-sensitive facts or anything outside training data.

MANDATORY: every response using search results ends with a `Sources:` block of markdown links.

```
[Response body]

Sources:
- [First Source](https://example.com/a)
```

- `allowed_domains` / `blocked_domains` scope results.
- `backend` selects the search engine. Only set it to compare engines or recover from failure.
- `transport` selects retrieval: `auto` (default), `curl`, `curl-then-zendriver`, `zendriver`, or `stdlib`. `auto` tries curl and escalates to Zendriver when a site bot-blocks it, routing straight to Zendriver for domains already learned to require it. Set it explicitly to stress a path or isolate transport failures.
- `categories` -- SearXNG result tab; each returns results structured for that domain. Omit for general web. A non-default value selects the SearXNG backend when you name none, and is REJECTED alongside an explicit backend that cannot serve it (it is no longer silently overridden):
  - `general` -- web results (default).
  - `images` -- image URL, resolution, format, source.
  - `videos` -- duration, view count, channel, embed URL.
  - `news` -- web results with publish date.
  - `map` -- places with coordinates and structured address.
  - `music` -- tracks with audio/embed URL and duration.
  - `it` -- software: packages (name/version/license/homepage), repos, code (repository/filename/language).
  - `science` -- papers with authors/DOI/citations. Prefer the `PaperSearch` tool for scholarly work (citation graph, fetch-by-id); use this only for a quick web-style lookup.
  - `files` -- files (filename/size/type) and torrents (size, seed/leech, magnet).
  - `social media` -- posts from Mastodon/Lemmy (web results).
- Current time: {{NOW}}. Resolve relative terms to absolute dates; include the current year for "latest" queries.
