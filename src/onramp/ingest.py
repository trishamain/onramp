"""Document -> chunks -> DynamoDB items.

This module is the ingest path. `scripts/local_ingest.py` and the IngestLambda
both call `ingest_document`; neither reimplements any of it. That is the whole
point -- if the laptop loop and the deployed Lambda chunked differently, the
recall numbers measured locally would not describe the deployed system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from onramp.chunking import chunk_markdown
from onramp.citations import TocIndex, build_url, parse_toc, product_area
from onramp.providers.base import EmbeddingProvider
from onramp.storage import ChunkItem, delete_document, write_chunks

# The corpus is stored under corpus/<guide>/... in S3 but the citation mapper
# speaks in repo paths (help/<guide>/...). One place to convert.
S3_PREFIX = "corpus/"


def s3_key_to_repo_path(s3_key: str) -> str:
    """corpus/xdm/schema/composition.md -> help/xdm/schema/composition.md"""
    rel = s3_key[len(S3_PREFIX) :] if s3_key.startswith(S3_PREFIX) else s3_key
    return "help/" + rel


@dataclass
class IngestResult:
    s3_key: str
    chunks_written: int
    chunks_deleted: int
    title: str
    doc_url: str


def load_toc_indexes(s3_client: Any, bucket: str) -> dict[str, TocIndex]:
    """Read every guide's TOC.md once, up front.

    Building these per-document would re-download the same 28 KB sources TOC for
    each of its 351 files. Built once, they are passed into every call.
    """
    indexes: dict[str, TocIndex] = {}
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/TOC.md"):
                continue
            guide = key[len(S3_PREFIX) :].split("/")[0]
            body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
            indexes[guide] = parse_toc(body.decode("utf-8", errors="replace"), guide)
    return indexes


def build_chunk_items(
    s3_key: str,
    raw_markdown: str,
    provider: EmbeddingProvider,
    toc_indexes: dict[str, TocIndex] | None = None,
) -> tuple[list[ChunkItem], str, str]:
    """Chunk one document and embed its chunks. Returns (items, title, doc_url)."""
    repo_path = s3_key_to_repo_path(s3_key)
    area = product_area(repo_path)
    toc = (toc_indexes or {}).get(area)
    doc_url = build_url(repo_path, toc)

    doc, chunks = chunk_markdown(raw_markdown, area)

    # Skip documents the source marks unpublished. They exist in the git repo
    # but are not served on Experience League, so any citation to one is a
    # guaranteed 404 -- which is worse than not retrieving the document at all.
    # Measured: 60 of 1,102 documents (5.4%) carry these flags.
    if doc.hidden:
        return [], doc.title, doc_url

    if not chunks:
        return [], doc.title, doc_url

    # One batched call rather than one per chunk: MiniLM embeds a list in a
    # single forward pass, which is most of why local ingest is fast.
    vectors = provider.embed([c.embed_text for c in chunks])

    items = [
        ChunkItem(
            s3_key=s3_key,
            index=c.index,
            text=c.text,
            embedding=vec,
            title=doc.title,
            product_area=area,
            heading_path=c.heading_path,
            doc_url=doc_url,
            token_count=c.token_count,
            embedding_provider=provider.model_id,
            embedding_dimensions=provider.dimensions,
        )
        for c, vec in zip(chunks, vectors, strict=True)
    ]
    return items, doc.title, doc_url


def ingest_document(
    table: Any,
    s3_key: str,
    raw_markdown: str,
    provider: EmbeddingProvider,
    toc_indexes: dict[str, TocIndex] | None = None,
    replace: bool = True,
) -> IngestResult:
    """Chunk, embed and store one document.

    `replace` deletes the document's existing chunks first. Without it, an edited
    document that produces fewer chunks than before would leave the surplus
    behind, and those stale chunks stay retrievable while citing text that no
    longer exists in the source.
    """
    items, title, doc_url = build_chunk_items(s3_key, raw_markdown, provider, toc_indexes)

    deleted = delete_document(table, s3_key) if replace else 0
    written = write_chunks(table, items) if items else 0

    return IngestResult(
        s3_key=s3_key,
        chunks_written=written,
        chunks_deleted=deleted,
        title=title,
        doc_url=doc_url,
    )


def should_ingest(s3_key: str) -> bool:
    """Skip non-markdown and skip TOC files.

    TOC.md is navigation, not content: it is a list of link titles that embeds
    near everything and answers nothing. It is read for URL mapping and then
    excluded from the index.
    """
    return s3_key.endswith(".md") and not s3_key.endswith("/TOC.md")
