Say the steps of the work in hand and where each stands. The person watching sees the plan as a checklist that ticks as you go, so they know what will happen and how far along you are.

Use it for any request that takes three or more distinct steps, and when the person gives you a list of things to do. Skip it for a question, a single edit, or anything done in one or two steps.

- Call it before your first other tool, with every step `pending` but the first, which is `in_progress`.
- Send the whole list each time. Keep exactly one step `in_progress` while you work, and none once all are `completed`.
- Mark a step `completed` as soon as it is done, and the next `in_progress` in the same call. Do not batch several completions.
- Send that call with the next step's first tool calls, in the same response, so the list moves as the work does and costs no turn of its own. Mark the last step `completed` before you answer.
- When the plan changes (you learn something, or the person steers you), send the new list and say why in `explanation`.
- Each step is 5-7 words, an action: "Run the failing test", not "Tests".
- At most {max_steps} steps. Never mark a step `completed` that you did not finish; leave it `pending` and say why in your answer.
