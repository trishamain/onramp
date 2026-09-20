#!/usr/bin/env python3
"""Measure p50/p95 end-to-end latency in four conditions.

  1. container cold start   -- force a new execution environment, then one call
  2. warm + vector cache    -- the normal path
  3. warm, cache disabled   -- what the module-scope cache is actually worth
  4. local script path      -- same code, no API Gateway, no Lambda

Every number here is measured. Nothing is estimated, and cold start is reported
rather than excluded -- it is the honest cost of a 1.87 GB container image, and
hiding it would make the rest of the table meaningless.

Forcing a cold start: updating a Lambda's environment variables replaces its
execution environments, so the next invocation must initialise from scratch.
That is more reliable than waiting for natural scale-down.

Usage:
  python scripts/benchmark.py --endpoint <url> --api-key <key> \\
      --table <name> --function onramp-query
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

N_QUERIES = 20


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    Not statistics.quantiles: with 20 samples its interpolation invents values
    between measurements, and a p95 that was never observed is not a measurement.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def call_api(endpoint: str, api_key: str, question: str, timeout: int = 120) -> tuple[float, int]:
    """POST one question. Returns (wall_clock_ms, http_status).

    Wall clock, not the handler's self-reported latency_ms: the caller's
    experience includes API Gateway and the network, and that is what a demo
    audience actually feels.
    """
    payload = json.dumps({"question": question}).encode()
    req = urllib.request.Request(  # noqa: S310
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "x-api-key": api_key},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    return (time.perf_counter() - t0) * 1000, status


def force_cold(function_name: str, region: str, profile: str | None, marker: str) -> None:
    """Replace the function's execution environments.

    Changing configuration -- here a throwaway env var -- tears down warm
    containers, so the next call pays full init: image pull/decompress, Python
    import, model load.
    """
    import boto3  # noqa: PLC0415

    client = boto3.Session(profile_name=profile, region_name=region).client("lambda")
    current = client.get_function_configuration(FunctionName=function_name)
    env = current.get("Environment", {}).get("Variables", {})
    env["BENCH_MARKER"] = marker
    client.update_function_configuration(FunctionName=function_name, Environment={"Variables": env})

    waiter = client.get_waiter("function_updated_v2")
    waiter.wait(FunctionName=function_name)


def set_cache_flag(function_name: str, region: str, profile: str | None, enabled: bool) -> None:
    """Toggle ENABLE_VECTOR_CACHE on the deployed function."""
    import boto3  # noqa: PLC0415

    client = boto3.Session(profile_name=profile, region_name=region).client("lambda")
    current = client.get_function_configuration(FunctionName=function_name)
    env = current.get("Environment", {}).get("Variables", {})
    env["ENABLE_VECTOR_CACHE"] = "true" if enabled else "false"
    client.update_function_configuration(FunctionName=function_name, Environment={"Variables": env})
    client.get_waiter("function_updated_v2").wait(FunctionName=function_name)


def report(label: str, samples: list[float], note: str = "") -> dict:
    row = {
        "condition": label,
        "n": len(samples),
        "p50_ms": round(percentile(samples, 50)),
        "p95_ms": round(percentile(samples, 95)),
        "min_ms": round(min(samples)) if samples else 0,
        "max_ms": round(max(samples)) if samples else 0,
        "mean_ms": round(statistics.mean(samples)) if samples else 0,
    }
    print(
        f"{label:<28} n={row['n']:<3} p50={row['p50_ms']:>6} ms  "
        f"p95={row['p95_ms']:>6} ms  min={row['min_ms']:>6}  max={row['max_ms']:>6}  {note}"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--function", default="onramp-query")
    ap.add_argument("--n", type=int, default=N_QUERIES)
    ap.add_argument("--cold-samples", type=int, default=3)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--skip-cold", action="store_true", help="skip the slow cold-start condition")
    args = ap.parse_args()

    if not GOLDEN.exists():
        print(f"ERROR: {GOLDEN} not found. Run `make eval-set` first.")
        return 1

    questions = [q["question"] for q in json.loads(GOLDEN.read_text())["questions"]]
    # Cycle the golden questions so every condition sees identical input; a
    # different question mix between conditions would confound the comparison.
    workload = [questions[i % len(questions)] for i in range(args.n)]

    results = []
    print(f"endpoint : {args.endpoint}")
    print(f"function : {args.function}")
    print(f"workload : {args.n} queries cycling {len(questions)} golden questions\n")

    # --- 1. cold start ------------------------------------------------------
    if not args.skip_cold:
        cold: list[float] = []
        for i in range(args.cold_samples):
            force_cold(args.function, args.region, args.profile, marker=f"cold-{i}-{args.n}")
            ms, status = call_api(args.endpoint, args.api_key, workload[i])
            if status != 200:
                print(f"ERROR: cold call returned HTTP {status}.")
                print("  Check: is the API key valid, and has the stage been deployed?")
                return 1
            cold.append(ms)
        results.append(report("1. container cold start", cold, "(forced, image init + model load)"))

    # --- 2. warm, cache on --------------------------------------------------
    set_cache_flag(args.function, args.region, args.profile, enabled=True)
    call_api(args.endpoint, args.api_key, workload[0])  # prime
    warm = [call_api(args.endpoint, args.api_key, q)[0] for q in workload]
    results.append(report("2. warm + vector cache", warm, "(module-scope index reused)"))

    # --- 3. warm, cache off -------------------------------------------------
    set_cache_flag(args.function, args.region, args.profile, enabled=False)
    call_api(args.endpoint, args.api_key, workload[0])  # prime past the config change
    nocache = [call_api(args.endpoint, args.api_key, q)[0] for q in workload]
    results.append(report("3. warm, cache disabled", nocache, "(DynamoDB scan every call)"))

    # Restore the shipped default so the benchmark does not leave the system
    # configured for the measurement rather than for use.
    set_cache_flag(args.function, args.region, args.profile, enabled=True)

    # --- 4. local path ------------------------------------------------------
    import boto3  # noqa: PLC0415

    from onramp.providers.factory import get_embedding_provider  # noqa: PLC0415
    from onramp.query import answer_question, load_index  # noqa: PLC0415

    table = (
        boto3.Session(profile_name=args.profile, region_name=args.region)
        .resource("dynamodb")
        .Table(args.table)
    )
    provider = get_embedding_provider()
    index = load_index(table)
    answer_question(workload[0], index, provider)  # warm the model
    local = []
    for q in workload:
        t0 = time.perf_counter()
        answer_question(q, index, provider)
        local.append((time.perf_counter() - t0) * 1000)
    results.append(report("4. local script path", local, f"(no API/Lambda, {index.size} chunks)"))

    out = REPO_ROOT / "evals" / "benchmark.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
