Load a user-authored skill: returns its full `SKILL.md` body for you to follow.

The skills you can load arrive in a reminder, each with a trigger that says when to use it.

- `skill` -- a name from that list (`name (source): trigger` → `name`).
- `args` -- optional text passed to the skill.

Before each response, scan the skill list. If a trigger matches the request or the direction of the work, load that skill before answering or taking any other action. Do not describe or reference a skill without loading it. Do not load a skill again while its instructions are still in the conversation.
