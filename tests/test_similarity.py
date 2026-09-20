from __future__ import annotations

import numpy as np
import pytest

from onramp.providers.null import NullEmbeddingProvider
from onramp.query import VectorIndex, _dedupe_by_document
from onramp.similarity import cosine_similarity, pack_vector, top_k, unpack_vector
from onramp.storage import LoadedChunk


def test_identical_vectors_score_one():
    v = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    assert cosine_similarity(v, v.reshape(1, -1))[0] == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_vectors_score_zero():
    q = np.array([1.0, 0.0], dtype=np.float32)
    m = np.array([[0.0, 1.0]], dtype=np.float32)
    assert cosine_similarity(q, m)[0] == pytest.approx(0.0, abs=1e-6)


def test_opposite_vectors_score_minus_one():
    q = np.array([1.0, 0.0], dtype=np.float32)
    m = np.array([[-1.0, 0.0]], dtype=np.float32)
    assert cosine_similarity(q, m)[0] == pytest.approx(-1.0, abs=1e-6)


def test_magnitude_does_not_affect_score():
    """Cosine is angle-only; a scaled duplicate must score identically."""
    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    scores = cosine_similarity(q, m)
    assert scores[0] == pytest.approx(scores[1], abs=1e-6)


def test_zero_vector_scores_zero_not_nan():
    """A NaN would sort to the top of a descending argsort and poison results."""
    q = np.array([1.0, 0.0], dtype=np.float32)
    m = np.array([[0.0, 0.0]], dtype=np.float32)
    score = cosine_similarity(q, m)[0]
    assert not np.isnan(score)
    assert score == pytest.approx(0.0)


def test_empty_matrix_returns_empty():
    q = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(q, np.zeros((0, 2), dtype=np.float32)).size == 0


def test_scores_stay_in_range():
    rng = np.random.default_rng(0)
    q = rng.normal(size=64).astype(np.float32)
    m = rng.normal(size=(50, 64)).astype(np.float32)
    scores = cosine_similarity(q, m)
    assert scores.min() >= -1.0 - 1e-5
    assert scores.max() <= 1.0 + 1e-5


# --- packing ----------------------------------------------------------------


def test_pack_unpack_roundtrip():
    vec = [0.1, -0.2, 0.3, 0.4]
    out = unpack_vector(pack_vector(vec))
    assert np.allclose(out, np.array(vec, dtype=np.float32), atol=1e-7)


def test_packed_size_is_four_bytes_per_dimension():
    """The storage claim in the README, asserted rather than asserted-in-prose."""
    assert len(pack_vector([0.0] * 384)) == 1536
    assert len(pack_vector([0.0] * 1024)) == 4096


def test_packed_vector_is_far_under_the_dynamodb_item_limit():
    assert len(pack_vector([0.0] * 1024)) < 400 * 1024


# --- top_k ------------------------------------------------------------------


def test_top_k_returns_best_first():
    scores = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
    assert top_k(scores, 3) == [1, 3, 2]


def test_top_k_clamps_to_available():
    assert top_k(np.array([0.5, 0.2], dtype=np.float32), 10) == [0, 1]


def test_top_k_on_empty():
    assert top_k(np.zeros(0, dtype=np.float32), 4) == []


def test_top_k_single_element():
    assert top_k(np.array([0.42], dtype=np.float32), 4) == [0]


def test_retrieval_ranks_the_right_chunk_end_to_end():
    """Null provider + real cosine: the exact-match chunk must win."""
    provider = NullEmbeddingProvider()
    corpus = ["identity graph linking rules", "snowflake source connector", "merge policies"]
    vectors = np.array(provider.embed(corpus), dtype=np.float32)

    (query,) = provider.embed(["snowflake source connector"])
    ranked = top_k(cosine_similarity(np.array(query, dtype=np.float32), vectors), 1)
    assert ranked == [1]


def test_dedupe_by_document_keeps_best_per_url():
    """One document must not occupy every result slot."""

    def chunk(url):
        return LoadedChunk(
            text="", title="", product_area="", heading_path="", doc_url=url, s3_key="", embedding=b""
        )

    idx = VectorIndex(
        chunks=[chunk("a"), chunk("a"), chunk("a"), chunk("b"), chunk("c")],
        matrix=None,
        provider_name="x",
        dimensions=384,
    )
    # Candidates in descending score order: a, a, a, b, c
    assert _dedupe_by_document(idx, [0, 1, 2, 3, 4], 3) == [0, 3, 4]


def test_dedupe_returns_fewer_when_documents_run_out():
    c = LoadedChunk(
        text="", title="", product_area="", heading_path="", doc_url="only", s3_key="", embedding=b""
    )
    idx = VectorIndex(chunks=[c, c, c], matrix=None, provider_name="x", dimensions=384)
    assert _dedupe_by_document(idx, [0, 1, 2], 3) == [0]
