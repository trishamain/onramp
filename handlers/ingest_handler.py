"""IngestLambda: SQS -> chunk -> embed -> DynamoDB.

Consumes S3 ObjectCreated notifications from SQS, in batches of 5, with partial
batch responses enabled so one malformed document fails alone rather than
poisoning its four neighbours.

Everything below the event unwrapping is onramp.ingest -- the same module
scripts/local_ingest.py calls. There is no second implementation of chunking or
embedding to drift out of sync.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import unquote_plus

import boto3

from onramp.ingest import ingest_document, load_toc_indexes, should_ingest
from onramp.providers.factory import get_embedding_provider

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TABLE_NAME = os.environ["TABLE_NAME"]
CORPUS_BUCKET = os.environ["CORPUS_BUCKET"]

# Module scope: these survive across warm invocations, so a container that has
# already paid for the model load and the TOC fetch does not pay again.
_s3 = boto3.client("s3")
_table = boto3.resource("dynamodb").Table(TABLE_NAME)
_provider = None
_toc_indexes: dict[str, Any] | None = None


def _get_provider():
    global _provider
    if _provider is None:
        _provider = get_embedding_provider()
        logger.info("provider=%s dims=%s", _provider.model_id, _provider.dimensions)
    return _provider


def _get_tocs():
    global _toc_indexes
    if _toc_indexes is None:
        _toc_indexes = load_toc_indexes(_s3, CORPUS_BUCKET)
        logger.info("loaded %d TOC indexes", len(_toc_indexes))
    return _toc_indexes


def _keys_from_record(record: dict[str, Any]) -> list[str]:
    """Unwrap S3 event notifications from the SQS message body.

    S3 also emits a TestEvent when a notification is first configured; it has no
    Records key and must not be treated as a failure.
    """
    body = json.loads(record["body"])
    if "Records" not in body:
        return []
    keys = []
    for s3_record in body["Records"]:
        # S3 URL-encodes keys in notifications; spaces arrive as '+'.
        keys.append(unquote_plus(s3_record["s3"]["object"]["key"]))
    return keys


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """SQS batch handler with partial batch response.

    Returning batchItemFailures tells SQS to redeliver only the listed messages.
    Without it, one failure would redeliver all five, re-ingesting four
    documents that already succeeded.
    """
    provider = _get_provider()
    tocs = _get_tocs()
    failures: list[dict[str, str]] = []

    for record in event.get("Records", []):
        message_id = record["messageId"]
        try:
            for key in _keys_from_record(record):
                if not should_ingest(key):
                    logger.info("skipping %s", key)
                    continue
                body = _s3.get_object(Bucket=CORPUS_BUCKET, Key=key)["Body"].read()
                result = ingest_document(
                    table=_table,
                    s3_key=key,
                    raw_markdown=body.decode("utf-8", errors="replace"),
                    provider=provider,
                    toc_indexes=tocs,
                )
                logger.info(
                    "ingested key=%s chunks=%d replaced=%d",
                    key,
                    result.chunks_written,
                    result.chunks_deleted,
                )
        except Exception:
            # Log with traceback so CloudWatch shows which document broke, then
            # hand just this message back to SQS for retry.
            logger.exception("failed messageId=%s", message_id)
            failures.append({"itemIdentifier": message_id})

    return {"batchItemFailures": failures}
