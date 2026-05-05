Produce a thorough recap of the entire conversation up to now. Focus especially on what the user explicitly asked for and every action you took in response.
This recap must capture technical specifics, code structures, and architectural choices with enough depth that development can resume seamlessly.

Before writing the finished recap, place your working analysis inside <analysis> tags. Use this space to organize observations and confirm full coverage. During this analysis:

1. Walk through every message in time order. For each segment, carefully extract:
   - What the user requested and their underlying goals
   - How you went about fulfilling those requests
   - Important decisions, technical ideas, and code structures
   - Concrete specifics such as:
     - file paths
     - complete code excerpts
     - function and method signatures
     - modifications to files
   - Mistakes or failures encountered and their resolutions
   - Give extra weight to direct user feedback, particularly corrections or redirections.
2. Verify technical correctness and completeness, ensuring every required element receives adequate coverage.

Structure the recap with these nine sections:

1. Primary Request and Intent: Record every user request and underlying goal with full detail.
2. Key Technical Concepts: Enumerate the significant technologies, frameworks, and technical ideas that came up.
3. Files and Code Sections: Catalog every file inspected, changed, or created. Prioritize the most recent interactions; include complete code excerpts where relevant alongside a note explaining why each file operation matters.
4. Errors and Fixes: Document each error encountered and its resolution. Highlight any user feedback that corrected your approach or redirected your strategy.
5. Problem Solving: Describe resolved issues and any troubleshooting still in progress.
6. All User Messages: Reproduce every user message (excluding tool results). These are essential for tracking feedback and evolving intent.
7. Pending Tasks: List every open task you were explicitly asked to handle.
8. Current Work: Describe with precision what was underway right before this recap was triggered. Emphasize the latest messages from both sides and include file paths and code excerpts as needed.
9. Optional Next Step: State the single next action tied to your most recent work. IMPORTANT: this must align directly with the user's latest explicit request and the task underway when summarization began. If that task was already finished, only propose a next step when it clearly follows from the user's stated goals. Never pursue tangential or previously completed requests without user confirmation.
                       When a next step exists, embed verbatim quotes from the recent exchange showing the exact task and stopping point. Literal quotation prevents drift in how the task is understood.

Below is a template showing the expected output format:

<example>
<analysis>
[Working notes verifying all required elements are addressed]
</analysis>

<summary>
1. Primary Request and Intent:
   [Thorough description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File path 1]
      - [Why this file matters]
      - [Changes applied, if any]
      - [Relevant code excerpt]
   - [File path 2]
      - [Relevant code excerpt]
   - [...]

4. Errors and Fixes:
    - [Error description 1]:
      - [Resolution applied]
      - [User feedback, if any]
    - [...]

5. Problem Solving:
   [Resolved issues and ongoing troubleshooting]

6. All User Messages:
    - [Non-tool-result user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Exact description of in-progress work]

9. Optional Next Step:
   [Next action, if applicable]

</summary>
</example>

Generate your recap of the conversation following this structure. Be precise and exhaustive.

The conversation context may contain extra summarization directives. If present, incorporate them when generating the recap. Typical examples:
<example>
## Compact Instructions
During summarization, emphasize TypeScript code modifications and retain details about mistakes and their corrections.
</example>

<example>
# Summary instructions
For compaction, concentrate on test outputs and code modifications. Reproduce file reads word-for-word.
</example>

## Supplementary compaction rules

### Pending task fidelity
Section 7 (Pending Tasks) demands complete specifications, not abbreviated outlines.
For each pending task:
- Preserve the user's precise requirements and success criteria verbatim.
- Retain any architectural or implementation details that were already established.
- When earlier compaction rounds included unresolved pending tasks, carry them forward at full resolution. Never shorten inherited task descriptions.

Deferred tasks represent the greatest risk of information loss during compaction.
Context is transient; once a task description is compressed, the original detail is unrecoverable.
