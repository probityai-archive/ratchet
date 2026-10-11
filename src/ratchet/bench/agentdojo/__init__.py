"""The AgentDojo benchmark as a Ratchet world.

One Ratchet case is one AgentDojo episode: a user task, optionally paired with an injection task whose
attacker text is planted in the environment. The agent runs in Ratchet's worker loop under a harness, its
tool calls execute against the AgentDojo environment through AgentDojo's own function runtime, and the
episode is graded by AgentDojo's own utility and security checks, unchanged. A case passes when the user's
task is achieved and, for an attacked episode, the attacker's goal is not.
"""
