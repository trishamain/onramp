#!/usr/bin/env python3
"""Does BM25 + dense fusion beat dense alone?

Measures four things against the golden set:
  - dense only (bge-small @ 500, the bi-encoder ablation winner)
  - BM25 only
  - RRF fusion at several dense:lexical weightings
and reports the added latency, because this runs on a synchronous API path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onramp import lexical  # noqa: E402
from onramp.chunking import chunk_markdown  # noqa: E402
from onramp.citations import build_url, parse_toc, product_area  # noqa: E402
from onramp.similarity import cosine_similarity  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "evals" / "golden.json"

RETRIEVER = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TARGET_TOKENS = 500

WEIGHTINGS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 1.0), (1.0, 2.0)]


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


def doc_rank(ordered_urls, expected: str) -> int:
    seen: list[str] = []
    for u in ordered_urls:
        if u not in seen:
            seen.append(u)
            if u == expected:
                return len(seen)
    return 9999


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    questions = json.loads(GOLDEN.read_text())["questions"]
    docs = load_corpus(Path(args.cache))

    texts: list[str] = []
    urls: list[str] = []
    for area, url, raw in docs:
        _doc, chunks = chunk_markdown(raw, area, target_tokens=TARGET_TOKENS)
        for c in chunks:
            texts.append(c.embed_text)
            urls.append(url)
    print(f"corpus: {len(docs)} docs, {len(texts)} chunks, {len(questions)} questions")

    model = SentenceTransformer(RETRIEVER)
    matrix = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=128, show_progress_bar=False
    ).astype(np.float32)
    qvecs = model.encode(
        [QUERY_PREFIX + q["question"] for q in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    t0 = time.perf_counter()
    bm25 = lexical.build(texts)
    print(f"bm25 index: {len(bm25.vocab)} terms, built in {time.perf_counter() - t0:.1f}s\n")

    dense_all = [cosine_similarity(qv, matrix) for qv in qvecs]
    lex_all = []
    lat = []
    for q in questions:
        t = time.perf_counter()
        lex_all.append(lexical.score(bm25, q["question"]))
        lat.append((time.perf_counter() - t) * 1000)
    print(f"bm25 scoring: {np.mean(lat):.0f} ms/query\n")

    n = len(questions)
    best = None
    for dw, lw in WEIGHTINGS:
        hits = [0, 0, 0]
        per_q = []
        for q, ds, ls in zip(questions, dense_all, lex_all, strict=True):
            fused = lexical.reciprocal_rank_fusion(ds, ls, dense_weight=dw, lexical_weight=lw)
            order = np.argsort(-fused)
            r = doc_rank((urls[i] for i in order), q["expected_url"])
            per_q.append((q["id"], r))
            hits[0] += r <= 1
            hits[1] += r <= 3
            hits[2] += r <= 5
        label = {(1.0, 0.0): "dense only", (0.0, 1.0): "bm25 only"}.get((dw, lw), f"rrf {dw}:{lw}")
        print(
            f"{label:<14} r@1={hits[0]}/{n} ({hits[0] / n:.0%})  "
            f"r@3={hits[1]}/{n} ({hits[1] / n:.0%})  r@5={hits[2]}/{n} ({hits[2] / n:.0%})"
        )
        if best is None or hits[1] > best[1][1]:
            best = (label, hits, per_q)

    print(f"\nbest: {best[0]}")
    for qid, r in best[2]:
        print(f"  {qid}: {r}{'' if r <= 3 else '  <-- miss'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
