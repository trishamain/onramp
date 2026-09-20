#!/usr/bin/env python3
"""Do the generated Experience League URLs actually resolve?

Samples stored doc_url values, issues HTTP HEAD requests, and reports the pass
rate. A citation that 404s is worse than no citation: it looks authoritative
and wastes the reader's time, and in a pre-sales conversation it is the kind of
thing that ends the conversation.

Deliberately checks URLs pulled from DynamoDB rather than recomputed from the
mapper, so this measures what the system actually serves, not what the mapping
function believes.

Usage:
  python scripts/verify_citations.py --table <name> [--sample 20]
"""

from __future__ import annotations

import argparse
import random
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Experience League serves a SPA behind a CDN that rejects unusual clients, so a
# browser-like UA avoids false negatives that have nothing to do with the URL.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)
TIMEOUT = 20


def check(url: str) -> tuple[int, str]:
    """HEAD the URL, following redirects. Returns (status, final_url)."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310
            return resp.status, resp.url
    except urllib.error.HTTPError as exc:
        # Some CDNs refuse HEAD but serve GET. Retry once before calling it a
        # failure, otherwise we would report a mapping bug that is not one.
        if exc.code in (403, 405):
            try:
                get = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
                with urllib.request.urlopen(get, timeout=TIMEOUT) as resp:  # noqa: S310
                    return resp.status, resp.url
            except Exception as inner:  # noqa: BLE001
                return getattr(inner, "code", 0), url
        return exc.code, url
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7, help="fixed so the number is reproducible")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    import boto3  # noqa: PLC0415

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    table = session.resource("dynamodb").Table(args.table)

    urls: set[str] = set()
    kwargs: dict = {"ProjectionExpression": "doc_url"}
    while True:
        resp = table.scan(**kwargs)
        urls.update(i["doc_url"] for i in resp.get("Items", []) if i.get("doc_url"))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    if not urls:
        print(f"ERROR: no doc_url values in {args.table!r}.")
        print("  Check: has ingest run? `aws dynamodb scan --table-name ... --select COUNT`")
        return 1

    ordered = sorted(urls)
    random.seed(args.seed)
    sample = random.sample(ordered, min(args.sample, len(ordered)))

    print(f"distinct doc_urls in table : {len(ordered)}")
    print(f"sampling                   : {len(sample)} (seed={args.seed})\n")

    passed = 0
    statuses: Counter[int] = Counter()
    failures: list[tuple[str, int, str]] = []

    for url in sample:
        status, final = check(url)
        statuses[status] += 1
        ok = 200 <= status < 300
        passed += ok
        mark = "ok  " if ok else "FAIL"
        print(f"{mark} {status:<4} {url}")
        if not ok:
            failures.append((url, status, final))

    rate = passed / len(sample)
    print(f"\npass rate : {passed}/{len(sample)} ({rate:.0%})")
    print(f"statuses  : {dict(statuses)}")

    if failures:
        # Group failures by product area: a whole-area failure means the mapping
        # rule is wrong for that guide, which is fixable. Scattered singletons
        # usually mean the upstream doc moved, which is not.
        print("\nfailures by product area:")
        areas: Counter[str] = Counter()
        for url, _status, _final in failures:
            parts = url.split("/experience-platform/")
            areas[parts[1].split("/")[0] if len(parts) > 1 else "?"] += 1
        for area, count in areas.most_common():
            print(f"  {area}: {count}")
        print("\n  A clustered failure means the mapping rule is wrong for that guide")
        print("  (fix src/onramp/citations.py). Scattered ones mean upstream moved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
