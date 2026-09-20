"""Provider interfaces.

The whole point of this module is that nothing downstream of it knows which
model produced a vector. Ingest, retrieval, the chunker and the eval harness all
talk to these Protocols, so swapping MiniLM for Titan is a config change rather
than a code change.

Why Protocol rather than ABC: the implementations have no shared behaviour worth
inheriting, and structural typing keeps test doubles from needing to import and
subclass production classes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    `dimensions` and `model_id` are part of the interface because both are
    written onto every stored chunk. A vector is meaningless without knowing
    which model produced it -- cosine similarity between a MiniLM vector and a
    Titan vector is arithmetic that returns a number and means nothing.
    """

    dimensions: int
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Order of the return value matches order of `texts`."""
        ...


@runtime_checkable
class ChatProvider(Protocol):
    """Synthesises an answer from retrieved passages.

    Declared now, deliberately unused. GENERATED MODE is behind a feature flag
    that is off by default, so no deploy depends on an implementation existing.
    """

    model_id: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's text completion for a system+user prompt pair."""
        ...


class ProviderMismatchError(RuntimeError):
    """Raised when query-time and index-time providers disagree.

    See `onramp.providers.factory.assert_provider_matches` for why this is fatal
    rather than a warning.
    """
