#!/usr/bin/env python3
"""Measure retrieval quality against the golden eval set.

Reports recall@1, recall@3, refusal rate on the out_of_scope probes, and mean
latency. Runs against EITHER the deployed API or the local path, and always
prints which one it used -- a number without its provenance is not a result.

recall@k here means: did the expected document appear in the top k passages,
compared by doc_url. Comparing URLs rather than chunk text is what makes the
metric robust to chunking changes.

Usage:
  python scripts/evaluate.py --mode local --table <name>
  python scripts/evaluate.py --mode api --endpoint <url> --api-key <key>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "evals" / "golden.json"


def _ask_local(question: str, index, provider, k: int):
    from onramp.query import answer_question  # noqa: PLC0415

    answer = answer_question(question, index, provider, k=k)
    return answer.to_dict()


def _ask_api(question: str, endpoint: str, api_key: str, k: int):
    payload = json.dumps({"question": question, "k": k}).encode()
    req = urllib.request.Request(  # noqa: S310 - endpoint is operator-supplied
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "x-api-key": api_key},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        body = json.loads(resp.read())
    # Trust the server's own latency_ms if present; otherwise measure round trip.
    body.setdefault("latency_ms", int((time.perf_counter() - started) * 1000))
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "api"], required=True)
    ap.add_argument("--table", help="required for --mode local")
    ap.add_argument("--endpoint", help="required for --mode api")
    ap.add_argument("--api-key", help="required for --mode api")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    if not GOLDEN.exists():
        print(f"ERROR: {GOLDEN} not found.")
        print("  Check: run `make eval-set` to build it from golden-draft.json.")
        return 1

    data = json.loads(GOLDEN.read_text())
    questions = data["questions"]
    out_of_scope = data["out_of_scope"]

    # --- wire up the chosen path -------------------------------------------
    if args.mode == "local":
        if not args.table:
            print("ERROR: --table is required for --mode local")
            return 1
        import boto3  # noqa: PLC0415

        from onramp.providers.factory import get_embedding_provider  # noqa: PLC0415
        from onramp.query import load_index  # noqa: PLC0415

        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        table = session.resource("dynamodb").Table(args.table)
        provider = get_embedding_provider(args.provider)
        index = load_index(table)
        if index.size == 0:
            print(f"ERROR: table {args.table!r} is empty.")
            print("  Check: run scripts/local_ingest.py first.")
            return 1

        def ask(q: str):
            return _ask_local(q, index, provider, args.k)

        target = f"LOCAL  table={args.table}  chunks={index.size}  provider={provider.model_id}"
    else:
        if not (args.endpoint and args.api_key):
            print("ERROR: --endpoint and --api-key are required for --mode api")
            return 1

        def ask(q: str):
            return _ask_api(q, args.endpoint, args.api_key, args.k)

        target = f"API    endpoint={args.endpoint}"

    print("=" * 72)
    print(f"EVALUATING AGAINST: {target}")
    print("=" * 72)

    # --- recall -------------------------------------------------------------
    hits_at_1 = 0
    hits_at_3 = 0
    latencies: list[float] = []
    misses: list[str] = []

    for q in questions:
        try:
            resp = ask(q["question"])
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"ERROR calling the API for {q['id']}: {exc}")
            print("  Check: is the endpoint correct and the API key valid?")
            return 1

        latencies.append(resp.get("latency_ms", 0))
        urls = [p["doc_url"] for p in resp.get("passages", [])]
        expected = q["expected_url"]

        if urls[:1] == [expected]:
            hits_at_1 += 1
        if expected in urls[:3]:
            hits_at_3 += 1
        else:
            misses.append(f"  {q['id']} [{q['category']}] {q['question'][:58]}...")

    # --- refusal ------------------------------------------------------------
    refused = 0
    for q in out_of_scope:
        resp = ask(q["question"])
        latencies.append(resp.get("latency_ms", 0))
        # A refusal is answer_or_null being set with no passages returned.
        if resp.get("answer_or_null") and not resp.get("passages"):
            refused += 1

    n = len(questions)
    print(f"\nrecall@1      : {hits_at_1}/{n}  ({hits_at_1 / n:.0%})")
    print(f"recall@3      : {hits_at_3}/{n}  ({hits_at_3 / n:.0%})")
    print(f"refusal rate  : {refused}/{len(out_of_scope)}  ({refused / len(out_of_scope):.0%})")
    print(f"mean latency  : {statistics.mean(latencies):.0f} ms")
    print(f"p95 latency   : {sorted(latencies)[int(len(latencies) * 0.95) - 1]:.0f} ms")

    if misses:
        print(f"\nmissed at k=3 ({len(misses)}):")
        for m in misses:
            print(m)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
