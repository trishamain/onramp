#!/usr/bin/env python3
"""Offline retrieval ablation.

Re-chunks and re-embeds the cached corpus under several configurations and
reports document-level recall against evals/golden.json. Runs entirely on local
files so a configuration can be rejected in minutes instead of after a
re-ingest and a container build.

Document-level means: score every chunk, keep each document's best chunk, rank
documents. That is the metric that matters, because a citation names a document.
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

# Some retrieval models are trained with asymmetric instruction prefixes and
# lose a lot of their advantage without them. e5 requires "query:"/"passage:";
# bge wants an instruction on the query side only. Testing them bare would
# understate them and produce a wrong conclusion.
BGE_QUERY = "Represent this sentence for searching relevant passages: "
E5_QUERY = "query: "
E5_PASSAGE = "passage: "

CONFIGS = [
    # label, model, target_tokens, query_prefix, passage_prefix
    ("A baseline ", "sentence-transformers/all-MiniLM-L6-v2", 500, "", ""),
    ("B fit-256  ", "sentence-transformers/all-MiniLM-L6-v2", 220, "", ""),
    ("C multi-qa ", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 500, "", ""),
    ("D mq+small ", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 220, "", ""),
    ("E bge-500  ", "BAAI/bge-small-en-v1.5", 500, BGE_QUERY, ""),
    ("F bge-220  ", "BAAI/bge-small-en-v1.5", 220, BGE_QUERY, ""),
    ("G e5-500   ", "intfloat/e5-small-v2", 500, E5_QUERY, E5_PASSAGE),
    ("H e5-220   ", "intfloat/e5-small-v2", 220, E5_QUERY, E5_PASSAGE),
]


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
            (
                repo_path,
                area,
                build_url(repo_path, tocs.get(area)),
                path.read_text(encoding="utf-8", errors="replace"),
            )
        )
    return docs


def run_config(cfg: tuple, docs, questions) -> dict:
    """cfg is (label, model_name, target_tokens, query_prefix, passage_prefix)."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    label, model_name, target, qprefix, pprefix = cfg

    model = SentenceTransformer(model_name)
    started = time.perf_counter()

    embed_texts: list[str] = []
    urls: list[str] = []
    for _repo_path, area, url, raw in docs:
        _doc, chunks = chunk_markdown(raw, area, target_tokens=target)
        for c in chunks:
            embed_texts.append(pprefix + c.embed_text)
            urls.append(url)

    matrix = model.encode(
        embed_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=128, show_progress_bar=False
    ).astype(np.float32)

    qvecs = model.encode(
        [qprefix + q["question"] for q in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    r1 = r3 = r5 = 0
    ranks = []
    for q, qv in zip(questions, qvecs, strict=True):
        scores = cosine_similarity(qv, matrix)
        best: dict[str, float] = {}
        for i, u in enumerate(urls):
            s = float(scores[i])
            if u not in best or s > best[u]:
                best[u] = s
        ordered = [u for u, _ in sorted(best.items(), key=lambda kv: -kv[1])]
        rank = ordered.index(q["expected_url"]) + 1 if q["expected_url"] in ordered else 9999
        ranks.append((q["id"], rank))
        r1 += rank <= 1
        r3 += rank <= 3
        r5 += rank <= 5

    return {
        "label": label,
        "model": model_name,
        "target": target,
        "chunks": len(embed_texts),
        "seconds": time.perf_counter() - started,
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "ranks": ranks,
        "n": len(questions),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="directory of the corpus (guide/<path>.md)")
    ap.add_argument("--only", default="", help="comma-separated config labels to run, e.g. E,F,G,H")
    args = ap.parse_args()

    cache = Path(args.cache)
    docs = load_corpus(cache)
    questions = json.loads(GOLDEN.read_text())["questions"]
    print(f"corpus: {len(docs)} documents, {len(questions)} questions\n")

    results = []
    for label, model_name, target, qprefix, pprefix in CONFIGS:  # noqa: B007
        if args.only and not label.strip().startswith(tuple(args.only.split(","))):
            continue
        res = run_config((label, model_name, target, qprefix, pprefix), docs, questions)
        results.append(res)
        n = res["n"]
        print(
            f"{label} target={target:<4} chunks={res['chunks']:<6} "
            f"r@1={res['r1']}/{n} ({res['r1'] / n:.0%})  "
            f"r@3={res['r3']}/{n} ({res['r3'] / n:.0%})  "
            f"r@5={res['r5']}/{n} ({res['r5'] / n:.0%})  "
            f"[{res['seconds']:.0f}s]"
        )

    best = max(results, key=lambda r: (r["r3"], r["r1"]))
    print(f"\nbest: {best['label'].strip()}  {best['model']}  target={best['target']}")
    print("per-question ranks for best config:")
    for qid, rank in best["ranks"]:
        flag = "" if rank <= 3 else ("  <-- miss" if rank < 9999 else "  <-- NOT FOUND")
        print(f"  {qid}: {rank}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
