Invoke a user-authored skill; returns its full `SKILL.md` body for the next model request to follow.

Skills are listed in the system prompt under `# Skills`, each with a trigger description.

- `skill` -- name from the `# Skills` list.
- `args` -- optional text forwarded to the skill.

When a skill matches the request or current direction of work, invoke it before answering or taking other actions. Don't re-invoke a skill whose instructions are already active in the conversation.
