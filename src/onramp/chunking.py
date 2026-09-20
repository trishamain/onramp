"""Heading-aware markdown chunking.

Two properties matter more than elegance here:

1. A fenced code block or a markdown table is never split. Half a code sample
   retrieves as confidently as a whole one and is worse than useless in a
   pre-sales conversation, and half a table loses its header row and therefore
   its meaning.
2. Every chunk knows where it came from. The product area, document title and
   heading path are prepended to the embedded text, so a chunk that reads
   "Select **Create audience**" still embeds near "segmentation" and
   "segment builder" rather than floating free of its context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from onramp.config import CHUNK_OVERLAP_RATIO, CHUNK_TARGET_TOKENS

_FRONTMATTER = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*(?:\{#[^}]*\})?\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|")


def approx_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token.

    Deliberately not tiktoken. The tokenizer that matters is whichever model is
    configured, and they disagree; adding a dependency to be precisely wrong for
    two of three providers is not worth it. Chunk size is a tuning knob, not a
    hard API limit, so a stable heuristic beats a precise-but-irrelevant count.
    """
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    text: str  # raw chunk body, what gets shown to the user
    embed_text: str  # body plus context prefix, what actually gets embedded
    heading_path: str
    index: int
    token_count: int


@dataclass
class ParsedDoc:
    title: str
    description: str
    body: str
    # True when the source marks this document as unpublished. Adobe uses
    # `hide: true` / `hidefromtoc: yes` for pages that exist in the repo but are
    # not served on Experience League -- citing one produces a guaranteed 404.
    hidden: bool = False


def parse_frontmatter(raw: str) -> ParsedDoc:
    """Strip YAML frontmatter, pulling out title and description.

    Hand-rolled rather than PyYAML: we need exactly two scalar keys, and the
    Lambda image is already carrying torch. A 40-line parser is cheaper than
    another dependency and cannot surprise us with YAML's type coercion.
    """
    title = ""
    description = ""
    hidden = False
    body = raw

    m = _FRONTMATTER.match(raw)
    if m:
        body = raw[m.end() :]
        for line in m.group("body").splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip().strip("'\"")
            if key == "title" and not title:
                title = value
            elif key == "description" and not description:
                description = value
            elif key in ("hide", "hidefromtoc") and value.lower() in ("true", "yes"):
                hidden = True

    # Fall back to the first H1 when frontmatter carries no title, which is
    # common in this corpus.
    if not title:
        for line in body.splitlines():
            hm = _HEADING.match(line)
            if hm and len(hm.group("hashes")) == 1:
                title = hm.group("text").strip()
                break

    return ParsedDoc(title=title, description=description, body=body, hidden=hidden)


@dataclass
class _Block:
    """An atomic unit. Blocks are never split, only grouped."""

    text: str
    heading_path: str
    tokens: int


def _consume_fence(lines: list[str], start: int, marker: str) -> tuple[str, int]:
    """Return the whole fenced block and the index just past it.

    An unterminated fence runs to EOF rather than raising -- malformed source
    should degrade into one large chunk, not fail the whole ingest.
    """
    code = [lines[start]]
    i = start + 1
    while i < len(lines):
        code.append(lines[i])
        if lines[i].strip().startswith(marker):
            i += 1
            break
        i += 1
    return "\n".join(code), i


def _consume_table(lines: list[str], start: int) -> tuple[str, int]:
    """Return consecutive pipe-prefixed rows as one block."""
    i = start
    rows: list[str] = []
    while i < len(lines) and _TABLE_ROW.match(lines[i]):
        rows.append(lines[i])
        i += 1
    return "\n".join(rows), i


def _split_blocks(body: str) -> list[_Block]:
    """Break the document into atomic blocks, tracking the heading stack."""
    lines = body.splitlines()
    blocks: list[_Block] = []
    heading_stack: list[str] = []
    buf: list[str] = []
    i = 0

    def heading_path() -> str:
        return " > ".join(heading_stack)

    def add(text: str) -> None:
        if text.strip():
            blocks.append(_Block(text=text, heading_path=heading_path(), tokens=approx_tokens(text)))

    def flush() -> None:
        nonlocal buf
        add("\n".join(buf).strip())
        buf = []

    while i < len(lines):
        line = lines[i]

        fence = _FENCE.match(line)
        if fence:
            flush()
            text, i = _consume_fence(lines, i, fence.group(1))
            add(text)
            continue

        if _TABLE_ROW.match(line):
            flush()
            text, i = _consume_table(lines, i)
            add(text)
            continue

        hm = _HEADING.match(line)
        if hm:
            flush()
            # Truncate the stack to the parent level, then push this heading.
            del heading_stack[len(hm.group("hashes")) - 1 :]
            heading_stack.append(hm.group("text").strip())
            add(line.strip())
            i += 1
            continue

        if not line.strip():
            flush()
        else:
            buf.append(line)
        i += 1

    flush()
    return blocks


def _context_prefix(product_area: str, title: str, heading_path: str) -> str:
    bits = [b for b in (product_area, title, heading_path) if b]
    return " | ".join(bits)


def chunk_markdown(
    raw: str,
    product_area: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
) -> tuple[ParsedDoc, list[Chunk]]:
    """Chunk one markdown document.

    Returns the parsed document (for title/description) alongside its chunks.
    """
    doc = parse_frontmatter(raw)
    blocks = _split_blocks(doc.body)
    if not blocks:
        return doc, []

    overlap_budget = int(target_tokens * overlap_ratio)
    chunks: list[Chunk] = []
    current: list[_Block] = []
    current_tokens = 0

    def emit() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(b.text for b in current).strip()
        if not text:
            current, current_tokens = [], 0
            return
        # Attribute the chunk to the heading path of its first substantive
        # block, which is the section a reader would say it belongs to.
        heading_path = next((b.heading_path for b in current if b.heading_path), "")
        prefix = _context_prefix(product_area, doc.title, heading_path)
        chunks.append(
            Chunk(
                text=text,
                embed_text=f"{prefix}\n\n{text}" if prefix else text,
                heading_path=heading_path,
                index=len(chunks),
                token_count=approx_tokens(text),
            )
        )
        current, current_tokens = [], 0

    for block in blocks:
        # An oversized atomic block (a long code sample) becomes its own chunk
        # rather than being split. Better one large chunk than two broken ones.
        if block.tokens > target_tokens:
            emit()
            prefix = _context_prefix(product_area, doc.title, block.heading_path)
            chunks.append(
                Chunk(
                    text=block.text,
                    embed_text=f"{prefix}\n\n{block.text}" if prefix else block.text,
                    heading_path=block.heading_path,
                    index=len(chunks),
                    token_count=block.tokens,
                )
            )
            continue

        if current_tokens + block.tokens > target_tokens and current:
            carry = _carry_for_overlap(current, overlap_budget)
            emit()
            current = list(carry)
            current_tokens = sum(b.tokens for b in current)

        current.append(block)
        current_tokens += block.tokens

    emit()
    return doc, chunks


def _carry_for_overlap(blocks: list[_Block], budget: int) -> list[_Block]:
    """Pick trailing whole blocks to repeat in the next chunk.

    Overlapping by whole blocks rather than by character count is what keeps the
    no-split-code-blocks guarantee true across the overlap as well.
    """
    if budget <= 0:
        return []
    carried: list[_Block] = []
    total = 0
    for block in reversed(blocks):
        if total + block.tokens > budget:
            break
        carried.insert(0, block)
        total += block.tokens
    return carried
