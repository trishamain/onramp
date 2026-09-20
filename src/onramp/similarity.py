"""Vector packing and cosine similarity.

Embeddings are stored in DynamoDB as packed float32 Binary, not as a list of
Numbers. At 384 dims that is 1,536 bytes instead of roughly 8 KB -- DynamoDB
serialises each Number as a decimal string, so a list of floats costs about 20
bytes per element. See the README data-model section for the arithmetic and for
what these figures become at Titan's 1024 dims.
"""

from __future__ import annotations

import struct

import numpy as np

# float32 rather than float64: the models emit float32, the extra precision
# would be invented, and it would double the stored size for nothing.
_DTYPE = np.float32


def pack_vector(vector: list[float]) -> bytes:
    """Pack a vector into little-endian float32 bytes for DynamoDB Binary."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(raw: bytes) -> np.ndarray:
    """Unpack stored bytes back into a numpy array."""
    return np.frombuffer(raw, dtype=_DTYPE)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against a matrix of vectors.

    Normalises explicitly rather than assuming unit vectors. The providers do
    normalise, but this function is also used against vectors round-tripped
    through float32 packing, and an un-normalised input would silently produce
    scores outside [-1, 1] that still sort plausibly.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=_DTYPE)

    query = query.astype(_DTYPE, copy=False)
    matrix = matrix.astype(_DTYPE, copy=False)

    q_norm = np.linalg.norm(query)
    m_norms = np.linalg.norm(matrix, axis=1)

    # Guard against zero vectors: a zero-length vector has no direction, so its
    # similarity to anything is 0, not a divide-by-zero NaN that would sort to
    # the top of a descending argsort.
    denom = q_norm * m_norms
    denom[denom == 0] = np.inf

    return (matrix @ query) / denom


def top_k(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k highest scores, best first."""
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    # argpartition is O(n) to find the k best, then we sort only those k.
    idx = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in idx[np.argsort(-scores[idx])]]
