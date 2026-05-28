Web search with results inline. Single round-trip.

Use for time-sensitive facts or anything outside training data.

MANDATORY: every response using search results ends with a `Sources:` block of markdown links.

```
[Response body]

Sources:
- [First Source](https://example.com/a)
```

- `allowed_domains` / `blocked_domains` scope results.
- `backend` -- only set to compare engines or recover from failure. Retry with a different backend before declaring unanswerable.
- Current time: {{NOW}}. Resolve relative terms to absolute dates; include the current year for "latest" queries.
