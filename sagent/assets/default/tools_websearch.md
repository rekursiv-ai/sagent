- Queries the web and incorporates results into the response
- Retrieves current information for time-sensitive topics and recent developments
- Delivers results as structured blocks with markdown-formatted hyperlinks
- Appropriate for facts that fall outside the model's training data
- Each invocation completes within a single API round-trip

MANDATORY CITATION RULE — strict compliance required:
  - Every response that uses search results MUST end with a "Sources:" block
  - Format each source as a markdown link: [Title](URL)
  - Omitting this section is never acceptable
  - Reference format:

    [Response body]

    Sources:
    - [First Source](https://example.com/a)
    - [Second Source](https://example.com/b)

Additional notes:
  - Domain allow-lists and block-lists are available for scoping results
  - `backend` selects the search engine: `duckduckgo` (default)
  - If the chosen backend errors out, is rate-limited, or returns no usable
    results, retry with a different `backend` before concluding the query is
    unanswerable. Backends fail independently.

DATE AWARENESS — critical for query accuracy:
  - The current local date and time is {{NOW}}.
  - Resolve relative time terms ("today", "yesterday", "this weekend", "last week", "recently") to absolute dates before querying, and verify the chosen data source can filter to that range.
  - Incorporate the current year (and month/day when relevant) into queries about recent documentation, news, or events. A request for "latest React docs" should include the current year, not a prior year.
