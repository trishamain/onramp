"""Question -> passages. The read path.

`scripts/local_ask.py` and the QueryLambda both call `answer_question`. Same
code, same thresholds, same response shape -- so a recall number measured on the
laptop describes the deployed system rather than a sibling of it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from onramp.config import REFUSAL_TEXT, SIMILARITY_THRESHOLD, TOP_K
from onramp.providers.base import EmbeddingProvider
from onramp.providers.factory import assert_provider_matches
from onramp.similarity import cosine_similarity, top_k
from onramp.storage import LoadedChunk, scan_all_chunks, stack_embeddings

SNIPPET_CHARS = 320


@dataclass
class Passage:
    title: str
    product_area: str
    doc_url: str
    score: float
    snippet: str


@dataclass
class Answer:
    mode: str  # "retrieval" | "generated"
    answer: str | None
    passages: list[Passage] = field(default_factory=list)
    latency_ms: int = 0
    confidence: float = 0.0
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # The API contract names this field answer_or_null to make the
        # retrieval-mode None explicit to a client reading the schema.
        d["answer_or_null"] = d.pop("answer")
        return d


@dataclass
class VectorIndex:
    """Chunks plus their stacked vectors, ready to score.

    Held in Lambda module scope across warm invocations. Loading ~1,100 vectors
    from DynamoDB is the dominant cost of a cold query; scoring them in numpy is
    microseconds.
    """

    chunks: list[LoadedChunk]
    matrix: Any
    provider_name: str
    dimensions: int

    @property
    def size(self) -> int:
        return len(self.chunks)


def load_index(table: Any) -> VectorIndex:
    chunks, provider_name, dimensions = scan_all_chunks(table)
    return VectorIndex(
        chunks=chunks,
        matrix=stack_embeddings(chunks),
        provider_name=provider_name,
        dimensions=dimensions,
    )


def _dedupe_by_document(index: VectorIndex, candidates: list[int], k: int) -> list[int]:
    """Keep the highest-scoring chunk per doc_url, preserving rank order."""
    seen: set[str] = set()
    out: list[int] = []
    for i in candidates:
        url = index.chunks[i].doc_url
        if url in seen:
            continue
        seen.add(url)
        out.append(i)
        if len(out) == k:
            break
    return out


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Extractive snippet: the opening of the chunk, cut at a word boundary.

    Deliberately extractive rather than generated. In retrieval mode nothing is
    synthesised, so what the user reads is exactly what the Adobe doc says.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0]
    return cut + "..."


def answer_question(
    question: str,
    index: VectorIndex,
    provider: EmbeddingProvider,
    k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    generate: bool = False,
) -> Answer:
    """Retrieve passages for a question, refusing when nothing scores well."""
    started = time.perf_counter()

    if index.size == 0:
        return Answer(
            mode="retrieval",
            answer=f"{REFUSAL_TEXT}. The index is empty -- run ingest first.",
            latency_ms=int((time.perf_counter() - started) * 1000),
            provider=provider.model_id,
        )

    # Fail closed before scoring: see factory.assert_provider_matches for why a
    # cross-provider comparison is noise rather than a weak signal.
    assert_provider_matches(provider, index.provider_name, index.dimensions)

    # embed_query applies the model's query-side prefix where it has one.
    query_vec = np.array(provider.embed_query(question), dtype=np.float32)
    scores = cosine_similarity(query_vec, index.matrix)

    # Over-fetch, then keep only each document's best chunk.
    #
    # Without this, one document can occupy every slot: a query for "Snowflake"
    # returned three chunks of the same connector page in the top four, wasting
    # 75% of the result set. Measured cost: recall@3 of 30% passage-level versus
    # 35% document-level on the same index. A citation names a document, so the
    # result set should too.
    ranked = _dedupe_by_document(index, top_k(scores, k * 8), k)

    best = float(scores[ranked[0]]) if ranked else 0.0

    if best < threshold:
        # Name the nearest product area even while refusing -- "I don't cover
        # that, but the closest thing I have is segmentation" is materially more
        # useful to a solutions consultant than a bare refusal.
        closest = index.chunks[ranked[0]].product_area if ranked else "unknown"
        return Answer(
            mode="retrieval",
            answer=f"{REFUSAL_TEXT}. Closest product area: {closest}.",
            passages=[],
            latency_ms=int((time.perf_counter() - started) * 1000),
            confidence=round(best, 4),
            provider=provider.model_id,
        )

    passages = [
        Passage(
            title=index.chunks[i].title,
            product_area=index.chunks[i].product_area,
            doc_url=index.chunks[i].doc_url,
            score=round(float(scores[i]), 4),
            snippet=_snippet(index.chunks[i].text),
        )
        for i in ranked
    ]

    answer_text = None
    mode = "retrieval"
    if generate:
        # GENERATED MODE is declared but not wired. Reaching here means the
        # feature flag was set without an implementation, so say so plainly
        # rather than silently downgrading and reporting the wrong mode.
        raise NotImplementedError(
            "GENERATED MODE requires a ChatProvider; none is wired up by default. "
            "See src/onramp/providers/chat.py."
        )

    return Answer(
        mode=mode,
        answer=answer_text,
        passages=passages,
        latency_ms=int((time.perf_counter() - started) * 1000),
        confidence=round(best, 4),
        provider=provider.model_id,
    )
