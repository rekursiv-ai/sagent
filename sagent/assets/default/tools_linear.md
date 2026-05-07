Interact with Linear issue tracking through GraphQL.

Requires `LINEAR_API_KEY` in the environment.

Operations:
- `list_issues` — list recent issues, optionally filtered by `team` or `assignee_email`; accepts `limit`.
- `get_issue` — fetch one issue by `id`, such as `ENG-42`, including comments.
- `create_issue` — create an issue with `team`, `title`, and optional `description`.
- `update_issue` — update an issue by `id`; accepts `title`, `description`, and `state_id`.
- `add_comment` — add `body` as a comment to issue `id`.

Use Linear only when the task explicitly involves Linear or issue-tracker state. For codebase discovery, use file/search tools instead.
