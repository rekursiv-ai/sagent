Drop a message into another live agent's inbox.

Arguments:
- `to` (required) -- target label, `parent`, or `user` (shown on the user's screen); non-existent errors.
- `content` (required) -- message text. Your label is attached.
- `delay` (optional) -- seconds to wait before delivery (tool returns immediately regardless). The delivered message automatically notes delay time.

Replies are owed:
- A message to an agent that owes you nothing is a request: it owes you a reply.
- A message to an agent you owe is your reply, and settles the debt.
- Going idle while owing a reply brings a nudge, at every idle, until you reply with AgentSend.

Your watcher:
- If you have no parent and you have `AgentSpawn`, sending to `parent` first starts your watcher: a copy of you, on your model, prompt and tools, told how to watch you. It becomes your parent, and your message is the first thing it is asked.
- From then on it sees what you say and call. The parent is enforcing a policy. The child is pursuing a goal: ask it to `keep me honest`, not to do your task.
- It owes you a reply to each request; you owe it a reply to each of its messages.
- Without `AgentSpawn`, a root has no parent, and sending to `parent` errors.

Uses:
- Other agent coordination (they can send back to you too).
- Partial-result handoff.
- Delayed self-reminder (send to own label).
- A watcher that keeps you honest (send to `parent`).
