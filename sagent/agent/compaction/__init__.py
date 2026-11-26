"""Compaction: what keeps a session within its model's window.

``pruner`` drops old tool results before a compaction is due; ``compactor``
summarizes the conversation; ``fold`` shrinks one message too large for the
window; ``boundaries`` says where a history may be cut; ``usable`` is the one
retry policy their model calls share.
"""
