"""Deterministic hash-based embeddings for tests.

Exists so the unit suite -- and CI -- can exercise the full ingest and retrieval
path without downloading a 90 MB model. The vectors carry no semantic meaning;
they are only required to be deterministic (same text always yields the same
vector) and unit-length, so cosine similarity behaves arithmetically the way the
real thing does.
"""

from __future__ import annotations

import hashlib
import math


class NullEmbeddingProvider:
    """Hash-derived pseudo-vectors. Never use for real retrieval."""

    model_id = "null-hash-v1"

    def __init__(self, dimensions: int = 384) -> None:
        # Defaults to 384 so Null and Local agree on shape; a test that swaps
        # them should not also have to reshape every fixture.
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        # SHA-256 gives 32 bytes per digest; we chain digests until we have one
        # byte per dimension. Chaining on a counter rather than re-hashing the
        # previous digest keeps each block independent of the last, which avoids
        # the periodicity you get from a feedback loop.
        raw = bytearray()
        counter = 0
        seed = text.encode("utf-8")
        while len(raw) < self.dimensions:
            raw.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
            counter += 1

        # Map bytes 0..255 onto [-1, 1] so vectors occupy the whole space rather
        # than one positive orthant, where every pair would look similar.
        vec = [(b / 127.5) - 1.0 for b in raw[: self.dimensions]]

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:  # unreachable for sha256 output, but cheap to guard
            return [0.0] * self.dimensions
        return [v / norm for v in vec]
