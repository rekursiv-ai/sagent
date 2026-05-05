"""Tool-call ID remapping for cross-provider model swaps.

Each provider's API expects tool-call IDs in a specific format
(Anthropic: ``toolu_*``, OpenAI: ``fc_*``). When conversation
history contains IDs from a different provider, the request
builder must remap them to the native format.

``IdRemapper`` generates sequential native IDs and maintains a
mapping so tool-call / tool-result pairs stay consistent within
a single request.
"""

from __future__ import annotations


class IdRemapper:
    """Generate provider-native IDs, preserving tool-call/result pairing.

    Args:
      prefix: String prefix for generated IDs (e.g. ``"call_"``).

    """

    __slots__ = ("_counter", "_map", "_prefix")

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._counter = 0
        self._map: dict[str, str] = {}

    def map(self, original_id: str) -> str:
        """Return the native ID for ``original_id``, creating one if new.

        Args:
          original_id: Provider-foreign tool-call ID.

        Returns:
          native_id: Deterministic provider-native ID for this request.

        """
        mapped = self._map.get(original_id)
        if mapped is not None:
            return mapped
        mapped = f"{self._prefix}{self._counter}"
        self._counter += 1
        self._map[original_id] = mapped
        return mapped
