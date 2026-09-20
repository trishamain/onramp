#!/usr/bin/env python3
"""Ingest the corpus from a laptop, writing to the real DynamoDB table.

This exists for iteration speed. Container image builds take 8-15 minutes, and
waiting that long to discover the chunker drops table headers is a bad loop. It
imports onramp.ingest -- the same module the IngestLambda calls -- so what it
measures is what the deployed system does.

THE DEPLOYED LAMBDA IS THE REAL ARTIFACT. This is the development loop.

Usage:
  python scripts/local_ingest.py --table <name> --bucket <name> [--limit 50]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import boto3  # noqa: E402

from onramp.ingest import S3_PREFIX, ingest_document, load_toc_indexes, should_ingest  # noqa: E402
from onramp.providers.factory import get_embedding_provider  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--provider", default=None, help="override EMBEDDING_PROVIDER")
    ap.add_argument("--limit", type=int, default=0, help="stop after N documents (0 = all)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")
    table = session.resource("dynamodb").Table(args.table)

    provider = get_embedding_provider(args.provider)
    print(f"provider   : {provider.model_id} ({provider.dimensions} dims)")

    try:
        toc_indexes = load_toc_indexes(s3, args.bucket)
    except s3.exceptions.NoSuchBucket:
        print(f"ERROR: bucket {args.bucket!r} does not exist.")
        print("  Check: does `aws s3 ls` list it, and is --profile the right account?")
        return 1
    print(f"toc guides : {len(toc_indexes)} ({', '.join(sorted(toc_indexes))})")

    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=args.bucket, Prefix=S3_PREFIX):
        keys.extend(o["Key"] for o in page.get("Contents", []) if should_ingest(o["Key"]))

    if not keys:
        print(f"ERROR: no markdown found under s3://{args.bucket}/{S3_PREFIX}")
        print("  Check: run `make load` to sync the corpus, or confirm the prefix.")
        return 1

    if args.limit:
        keys = keys[: args.limit]

    print(f"documents  : {len(keys)}")
    started = time.perf_counter()
    total_chunks = 0
    failures: list[tuple[str, str]] = []

    for n, key in enumerate(keys, start=1):
        try:
            body = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read()
            result = ingest_document(
                table=table,
                s3_key=key,
                raw_markdown=body.decode("utf-8", errors="replace"),
                provider=provider,
                toc_indexes=toc_indexes,
            )
            total_chunks += result.chunks_written
        except Exception as exc:  # noqa: BLE001 - one bad doc must not stop the run
            failures.append((key, str(exc)))
            continue

        if n % 50 == 0 or n == len(keys):
            rate = n / max(time.perf_counter() - started, 1e-6)
            print(f"  {n:>5}/{len(keys)}  chunks={total_chunks:<6} {rate:.1f} docs/s")

    elapsed = time.perf_counter() - started
    print(f"\nchunks     : {total_chunks}")
    print(f"elapsed    : {elapsed:.1f}s ({len(keys) / max(elapsed, 1e-6):.1f} docs/s)")
    print(f"failures   : {len(failures)}")
    for key, err in failures[:10]:
        print(f"  {key}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
