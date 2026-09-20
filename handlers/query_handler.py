"""QueryLambda: POST /ask -> retrieved passages with citations.

The retrieval logic is onramp.query, identical to what scripts/local_ask.py
runs. This file is the HTTP envelope and the warm-start cache, nothing more.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import boto3

from onramp.config import ENABLE_VECTOR_CACHE, SIMILARITY_THRESHOLD, TOP_K
from onramp.providers.base import ProviderMismatchError
from onramp.providers.factory import get_embedding_provider
from onramp.query import answer_question, load_index

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TABLE_NAME = os.environ["TABLE_NAME"]

_table = boto3.resource("dynamodb").Table(TABLE_NAME)
_provider = None
_index = None


def _get_provider():
    global _provider
    if _provider is None:
        t0 = time.perf_counter()
        _provider = get_embedding_provider()
        logger.info("model loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return _provider


def _get_index():
    """Load all vectors, cached in module scope across warm invocations.

    This is the single biggest lever on warm latency: scanning ~6,000 items out
    of DynamoDB dominates a cold query, while scoring them in numpy is
    sub-millisecond. ENABLE_VECTOR_CACHE=false forces a reload every call so the
    benchmark can measure what the cache is actually worth.
    """
    global _index
    if _index is None or not ENABLE_VECTOR_CACHE:
        t0 = time.perf_counter()
        idx = load_index(_table)
        logger.info("index loaded chunks=%d in %.0f ms", idx.size, (time.perf_counter() - t0) * 1000)
        if not ENABLE_VECTOR_CACHE:
            return idx
        _index = idx
    return _index


def _response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            # CORS is configured on the API Gateway method too; this covers the
            # non-preflight response.
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload),
    }


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "body must be valid JSON"})

    question = (body.get("question") or "").strip()
    if not question:
        return _response(400, {"error": "missing 'question'"})

    k = int(body.get("k") or TOP_K)
    threshold = float(body.get("threshold") or SIMILARITY_THRESHOLD)

    try:
        provider = _get_provider()
        index = _get_index()
    except Exception:
        logger.exception("failed to initialise")
        return _response(500, {"error": "initialisation failed"})

    if index.size == 0:
        return _response(503, {"error": "index is empty; run ingest"})

    try:
        answer = answer_question(question, index, provider, k=k, threshold=threshold)
    except ProviderMismatchError as exc:
        # 409: the request is well-formed but the index was built by a different
        # model, so answering would return confident nonsense.
        logger.error("provider mismatch: %s", exc)
        return _response(409, {"error": str(exc)})
    except Exception:
        logger.exception("query failed")
        return _response(500, {"error": "query failed"})

    payload = answer.to_dict()
    # Report end-to-end handler time, not just the scoring time measured inside
    # answer_question, so the number matches what a caller experiences.
    payload["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return _response(200, payload)
