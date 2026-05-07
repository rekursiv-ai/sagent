Invoke a user-authored skill by name.

The available skills are listed in the system prompt under `# Skills`. Each skill has a trigger description. When a skill matches the user's request or the current direction of work, call this tool before answering or taking other actions.

Arguments:
- `skill` — the skill name from the system prompt.
- `args` — optional text forwarded to the skill.

The tool returns the full `SKILL.md` body so the next model request can follow those instructions. Do not invoke a skill again if its instructions are already active in the conversation.
