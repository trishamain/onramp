#!/usr/bin/env python3
"""Query the index from a laptop, using the same code the QueryLambda runs.

Imports onramp.query directly -- no reimplementation -- so a result here is the
result the deployed API would give, minus API Gateway and Lambda latency.

THE DEPLOYED LAMBDA IS THE REAL ARTIFACT. This is the development loop.

Usage:
  python scripts/local_ask.py --table <name> "how does identity stitching work?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import boto3  # noqa: E402

from onramp.config import SIMILARITY_THRESHOLD, TOP_K  # noqa: E402
from onramp.providers.base import ProviderMismatchError  # noqa: E402
from onramp.providers.factory import get_embedding_provider  # noqa: E402
from onramp.query import answer_question, load_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--table", required=True)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    ap.add_argument("--json", action="store_true", help="print the raw API response shape")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    question = " ".join(args.question)
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    table = session.resource("dynamodb").Table(args.table)

    provider = get_embedding_provider(args.provider)
    index = load_index(table)

    if index.size == 0:
        print(f"ERROR: table {args.table!r} has no chunks.")
        print("  Check: run scripts/local_ingest.py first.")
        return 1

    try:
        answer = answer_question(question, index, provider, k=args.k, threshold=args.threshold)
    except ProviderMismatchError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0

    print(f"Q: {question}")
    print(
        f"   [{index.size} chunks | {answer.provider} | {answer.latency_ms} ms | "
        f"confidence {answer.confidence}]\n"
    )

    if answer.answer:
        print(answer.answer)
        return 0

    for i, p in enumerate(answer.passages, start=1):
        print(f"{i}. [{p.score}] {p.title}  ({p.product_area})")
        print(f"   {p.doc_url}")
        print(f"   {p.snippet}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
