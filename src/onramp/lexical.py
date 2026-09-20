"""BM25 lexical scoring, hand-rolled in numpy.

WHY THIS EXISTS
---------------
The dense-only ablation topped out at 35% recall@3, and the failures were not
near-misses: the expected document ranked 126, 145, 220, 223, 249. Those are
vocabulary-gap failures. The eval questions are phrased the way a prospect
speaks -- "can the website change what it shows during the same visit" -- and
the document that answers it is titled "Edge segmentation". No bi-encoder maps
those into the same neighbourhood reliably, but a lexical match on a distinctive
term like "segmentation" or "Destination SDK" finds it immediately.

BM25 and dense retrieval fail differently, which is exactly why combining them
works: BM25 is precise on rare terms and blind to paraphrase; embeddings handle
paraphrase and wash out rare terms. Fusing them covers both.

No new dependency -- this is numpy and a tokenizer regex. rank_bm25 would do the
same thing in a package I would then have to explain in an interview.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Standard BM25 parameters. k1 controls term-frequency saturation (how quickly
# repeated terms stop adding score); b controls length normalisation.
K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")

# Stopwords hurt BM25 more than they help: they appear in nearly every chunk, so
# they contribute almost no discrimination while inflating length statistics.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "can",
        "do",
        "does",
        "not",
        "no",
        "us",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Bm25Index:
    """Sparse term statistics over the chunk corpus.

    Held alongside the dense matrix in module scope in the Lambda. Building it
    is pure CPU over text already loaded from DynamoDB, so it adds no I/O.
    """

    vocab: dict[str, int]
    # (n_chunks, n_terms) term frequencies, kept dense because this corpus is
    # small. At 6k chunks x ~20k terms that is large; see build() for why we
    # store per-chunk sparse rows instead.
    rows: list[dict[int, int]]
    doc_len: np.ndarray
    avg_len: float
    idf: np.ndarray

    @property
    def size(self) -> int:
        return len(self.rows)


def build(texts: list[str]) -> Bm25Index:
    vocab: dict[str, int] = {}
    rows: list[dict[int, int]] = []
    doc_len = np.zeros(len(texts), dtype=np.float32)

    for i, text in enumerate(texts):
        counts: dict[int, int] = {}
        tokens = tokenize(text)
        doc_len[i] = len(tokens)
        for tok in tokens:
            idx = vocab.get(tok)
            if idx is None:
                idx = len(vocab)
                vocab[tok] = idx
            counts[idx] = counts.get(idx, 0) + 1
        rows.append(counts)

    # Document frequency per term, then the BM25 idf with the +0.5 smoothing
    # that keeps very common terms from going negative.
    df = np.zeros(len(vocab), dtype=np.float32)
    for counts in rows:
        for idx in counts:
            df[idx] += 1.0

    n = max(len(texts), 1)
    idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)

    return Bm25Index(
        vocab=vocab,
        rows=rows,
        doc_len=doc_len,
        avg_len=float(doc_len.mean()) if len(texts) else 0.0,
        idf=idf,
    )


def score(index: Bm25Index, query: str) -> np.ndarray:
    """BM25 score of every chunk against the query.

    Only chunks containing at least one query term are touched, which is what
    keeps this fast without a real inverted index: most chunks score zero and
    are never visited.
    """
    scores = np.zeros(index.size, dtype=np.float32)
    if index.size == 0:
        return scores

    q_terms = [index.vocab[t] for t in tokenize(query) if t in index.vocab]
    if not q_terms:
        return scores

    q_set = set(q_terms)
    for i, counts in enumerate(index.rows):
        total = 0.0
        norm = K1 * (1.0 - B + B * (index.doc_len[i] / max(index.avg_len, 1e-6)))
        for term in q_set:
            tf = counts.get(term)
            if not tf:
                continue
            total += float(index.idf[term]) * (tf * (K1 + 1.0)) / (tf + norm)
        scores[i] = total
    return scores


def reciprocal_rank_fusion(
    dense_scores: np.ndarray,
    lexical_scores: np.ndarray,
    k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> np.ndarray:
    """Combine two rankings by rank rather than by raw score.

    RRF is used instead of normalising and adding the scores because cosine
    similarity and BM25 are not on comparable scales -- cosine is bounded in
    [-1, 1] and BM25 is unbounded and corpus-dependent. Any fixed normalisation
    is a guess that breaks when the corpus changes. RRF only reads the ordering,
    so it is scale-free.

    k=60 is the value from the original RRF paper; it damps the contribution of
    low-ranked items so a single list cannot dominate on its tail alone.
    """
    n = dense_scores.shape[0]
    fused = np.zeros(n, dtype=np.float32)

    for scores, weight in ((dense_scores, dense_weight), (lexical_scores, lexical_weight)):
        if weight == 0.0:
            continue
        order = np.argsort(-scores)
        ranks = np.empty(n, dtype=np.int32)
        ranks[order] = np.arange(n, dtype=np.int32)
        fused += weight * (1.0 / (k + 1 + ranks)).astype(np.float32)

    return fused
