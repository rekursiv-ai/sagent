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

DATE AWARENESS — critical for query accuracy:
  - Today's date is {{NOW}}. Always incorporate the current year into queries about recent documentation, news, or events.
  - For instance, a request for "latest React docs" should include the current year in the search terms rather than a prior year
