Play a WAV file on the user's speakers.

Use this only for short notifications such as an end-of-task bell or an attention-needed ping. The path may be absolute or relative to the current working directory.

Only `.wav` files are supported. Playback is best-effort: on headless SSH, CI, containers, or hosts without an audio backend, the tool logs the issue and returns success instead of failing the task.
