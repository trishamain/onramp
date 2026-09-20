#!/usr/bin/env python3
"""Does a cross-encoder reranker close the gap that bi-encoders could not?

The bi-encoder ablation topped out at 35% recall@3. The failure mode is a
vocabulary mismatch: the eval questions are phrased the way a prospect speaks
("someone browses on their phone then buys on a laptop") and the corpus is
written the way Adobe documents ("cross-device identity stitching"). A
bi-encoder must map both into the same vector space independently, with no
chance to compare them directly.

A cross-encoder sees the question and the passage together and scores the pair.
That is far more accurate and far too slow to run over 6,343 chunks -- so the
standard shape is retrieve-then-rerank: take the bi-encoder's top N candidates,
rerank only those.

This measures whether that actually helps here, and at what latency cost,
before any of it goes near the query path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onramp.chunking import chunk_markdown  # noqa: E402
from onramp.citations import build_url, parse_toc, product_area  # noqa: E402
from onramp.similarity import cosine_similarity  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "evals" / "golden.json"

# Winner of the bi-encoder ablation.
RETRIEVER = "BAAI/bge-small-en-v1.5"
RETRIEVER_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TARGET_TOKENS = 500

# Small enough to run in a Lambda without a GPU. The larger ms-marco models are
# more accurate and too slow for a synchronous request.
RERANKERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

CANDIDATE_DEPTHS = [20, 50, 100]


def load_corpus(cache: Path):
    tocs = {}
    for toc in cache.glob("*/TOC.md"):
        guide = toc.parent.name
        tocs[guide] = parse_toc(toc.read_text(encoding="utf-8", errors="replace"), guide)

    docs = []
    for path in sorted(cache.rglob("*.md")):
        if path.name == "TOC.md":
            continue
        rel = path.relative_to(cache).as_posix()
        repo_path = "help/" + rel
        area = product_area(repo_path)
        docs.append(
            (area, build_url(repo_path, tocs.get(area)), path.read_text(encoding="utf-8", errors="replace"))
        )
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: PLC0415

    questions = json.loads(GOLDEN.read_text())["questions"]
    docs = load_corpus(Path(args.cache))

    # --- build the dense index once ---------------------------------------
    texts: list[str] = []
    urls: list[str] = []
    for area, url, raw in docs:
        _doc, chunks = chunk_markdown(raw, area, target_tokens=TARGET_TOKENS)
        for c in chunks:
            texts.append(c.embed_text)
            urls.append(url)

    print(f"corpus: {len(docs)} docs, {len(texts)} chunks, {len(questions)} questions")
    retriever = SentenceTransformer(RETRIEVER)
    matrix = retriever.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=128, show_progress_bar=False
    ).astype(np.float32)
    qvecs = retriever.encode(
        [RETRIEVER_QUERY_PREFIX + q["question"] for q in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # --- dense-only baseline ----------------------------------------------
    def doc_rank(order_urls: list[str], expected: str) -> int:
        seen: list[str] = []
        for u in order_urls:
            if u not in seen:
                seen.append(u)
            if u == expected:
                return seen.index(u) + 1
        return 9999

    base = [0, 0, 0]
    dense_orders = []
    for q, qv in zip(questions, qvecs, strict=True):
        scores = cosine_similarity(qv, matrix)
        order = np.argsort(-scores)
        dense_orders.append(order)
        r = doc_rank([urls[i] for i in order], q["expected_url"])
        base[0] += r <= 1
        base[1] += r <= 3
        base[2] += r <= 5
    n = len(questions)
    print(
        f"\ndense only            r@1={base[0]}/{n} ({base[0] / n:.0%})  "
        f"r@3={base[1]}/{n} ({base[1] / n:.0%})  r@5={base[2]}/{n} ({base[2] / n:.0%})"
    )

    # --- rerank ------------------------------------------------------------
    for model_name in RERANKERS:
        ce = CrossEncoder(model_name)
        for depth in CANDIDATE_DEPTHS:
            hits = [0, 0, 0]
            latencies = []
            for q, order in zip(questions, dense_orders, strict=True):
                cand = [int(i) for i in order[:depth]]
                pairs = [(q["question"], texts[i]) for i in cand]
                t0 = time.perf_counter()
                ce_scores = ce.predict(pairs, show_progress_bar=False)
                latencies.append((time.perf_counter() - t0) * 1000)
                reordered = [cand[j] for j in np.argsort(-np.asarray(ce_scores))]
                r = doc_rank([urls[i] for i in reordered], q["expected_url"])
                hits[0] += r <= 1
                hits[1] += r <= 3
                hits[2] += r <= 5
            print(
                f"rerank depth={depth:<4}      r@1={hits[0]}/{n} ({hits[0] / n:.0%})  "
                f"r@3={hits[1]}/{n} ({hits[1] / n:.0%})  r@5={hits[2]}/{n} ({hits[2] / n:.0%})  "
                f"[+{np.mean(latencies):.0f} ms/query]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
