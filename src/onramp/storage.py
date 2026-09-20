"""DynamoDB item shape and access.

One table, one item per chunk:

    PK = DOC#<sha256 of the s3 key>    SK = CHUNK#<zero-padded index>

Hashing the key rather than using it directly keeps the partition key a fixed
64 characters regardless of how deep the source path is, and avoids having to
think about which characters are legal in a DynamoDB key. The zero-padded sort
key means CHUNK#000009 sorts before CHUNK#000010, which a bare integer string
would not.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from onramp.similarity import pack_vector, unpack_vector

# DynamoDB caps BatchWriteItem at 25 items per call. Not a tunable.
BATCH_WRITE_LIMIT = 25


def doc_pk(s3_key: str) -> str:
    return "DOC#" + hashlib.sha256(s3_key.encode("utf-8")).hexdigest()


def chunk_sk(index: int) -> str:
    # Six digits supports 999,999 chunks in one document -- far beyond anything
    # in this corpus, where the largest document produces low hundreds.
    return f"CHUNK#{index:06d}"


@dataclass
class ChunkItem:
    """One stored chunk. Mirrors the DynamoDB attribute set exactly."""

    s3_key: str
    index: int
    text: str
    embedding: list[float]
    title: str
    product_area: str
    heading_path: str
    doc_url: str
    token_count: int
    embedding_provider: str
    embedding_dimensions: int

    def to_item(self) -> dict[str, Any]:
        return {
            "PK": doc_pk(self.s3_key),
            "SK": chunk_sk(self.index),
            "text": self.text,
            # Binary, not a list of Numbers: DynamoDB serialises each Number as
            # a decimal string (~20 bytes), so 384 floats cost ~8 KB as a list
            # versus 1,536 bytes packed. See README for the full arithmetic.
            "embedding": pack_vector(self.embedding),
            "title": self.title,
            "product_area": self.product_area,
            "heading_path": self.heading_path,
            "s3_key": self.s3_key,
            "doc_url": self.doc_url,
            "token_count": self.token_count,
            # Written on EVERY item so a mixed-provider table is detectable
            # rather than merely unlikely. The query path refuses to score
            # against vectors a different model produced.
            "embedding_provider": self.embedding_provider,
            "embedding_dimensions": self.embedding_dimensions,
        }


def _chunked(seq: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def write_chunks(table: Any, items: list[ChunkItem]) -> int:
    """Write chunks via batch_writer, which handles retries and unprocessed items.

    boto3's batch_writer already buffers to 25 and retries UnprocessedItems with
    backoff, so hand-rolling that loop would only reproduce it worse.
    """
    written = 0
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item.to_item())
            written += 1
    return written


def delete_document(table: Any, s3_key: str) -> int:
    """Remove every chunk of one document.

    Needed because re-ingesting an edited file must not leave the previous
    version's surplus chunks behind: if a doc shrinks from 12 chunks to 8, the
    stale 9-12 would still be retrievable and would cite content that no longer
    exists in the source.
    """
    pk = doc_pk(s3_key)
    keys = [{"PK": i["PK"], "SK": i["SK"]} for i in _query_all(table, pk, projection="PK,SK")]
    with table.batch_writer() as batch:
        for key in keys:
            batch.delete_item(Key=key)
    return len(keys)


def _query_all(table: Any, pk: str, projection: str | None = None) -> Iterator[dict[str, Any]]:
    from boto3.dynamodb.conditions import Key  # noqa: PLC0415

    kwargs: dict[str, Any] = {"KeyConditionExpression": Key("PK").eq(pk)}
    if projection:
        kwargs["ProjectionExpression"] = projection
    while True:
        resp = table.query(**kwargs)
        yield from resp.get("Items", [])
        if "LastEvaluatedKey" not in resp:
            return
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]


@dataclass
class LoadedChunk:
    """A chunk read back for scoring. `embedding` stays packed until needed."""

    text: str
    title: str
    product_area: str
    heading_path: str
    doc_url: str
    s3_key: str
    embedding: bytes


def scan_all_chunks(table: Any) -> tuple[list[LoadedChunk], str, int]:
    """Load every chunk, returning them with the provider that produced them.

    A full Scan is the right call at this corpus size and the wrong one later --
    see the README tradeoff on when DynamoDB brute force stops being defensible
    against OpenSearch Serverless or S3 Vectors.
    """
    chunks: list[LoadedChunk] = []
    provider_name = ""
    dimensions = 0

    kwargs: dict[str, Any] = {}
    while True:
        resp = table.scan(**kwargs)
        for item in resp.get("Items", []):
            raw = item["embedding"]
            # boto3 returns Binary; .value is the underlying bytes.
            blob = raw.value if hasattr(raw, "value") else bytes(raw)
            chunks.append(
                LoadedChunk(
                    text=item.get("text", ""),
                    title=item.get("title", ""),
                    product_area=item.get("product_area", ""),
                    heading_path=item.get("heading_path", ""),
                    doc_url=item.get("doc_url", ""),
                    s3_key=item.get("s3_key", ""),
                    embedding=blob,
                )
            )
            if not provider_name:
                provider_name = item.get("embedding_provider", "")
                dimensions = int(item.get("embedding_dimensions", 0))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    return chunks, provider_name, dimensions


def stack_embeddings(chunks: list[LoadedChunk]) -> Any:
    """Unpack every chunk's vector into one (n, dims) matrix for scoring."""
    import numpy as np  # noqa: PLC0415

    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack([unpack_vector(c.embedding) for c in chunks])
