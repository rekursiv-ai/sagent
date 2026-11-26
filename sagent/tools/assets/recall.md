Tell you of stored sessions like one session, as each is found.

The session store embeds what each session says. This call asks it, again
every minute, which other sessions' records lie nearest what one session
last said, and writes each session it has not named before, one per line:

    session <id> p<part>#<idx> (<title>): <how the nearest record starts>

The session is yours, or that of the agent you name in `agent`: a watcher
names the agent it watches.

The call never ends by itself. Make it once, with `background: true`: each
time it names a new session you are woken with the new lines, so you need
not poll it. Stop it with `BackgroundTask` cancel.

A line is a lead, not a finding: nearness is by meaning, and is often only a
likeness of words. Read the record it names before you rely on it or tell
another agent of it. The store trails what was just said by about a minute.
That session's own parent and children are never named.

- `agent` -- label or relation of the agent whose session to search near.
- `limit` -- records each search returns, nearest first.
