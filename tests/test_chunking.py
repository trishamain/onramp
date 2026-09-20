from __future__ import annotations

from onramp.chunking import (
    approx_tokens,
    chunk_markdown,
    parse_frontmatter,
)

FRONTMATTER_DOC = """---
title: Identity linking logic
description: How identities are linked across devices.
solution: Experience Platform
---

# Identity linking logic

Intro paragraph about linking.

## Deterministic linking

Deterministic detail here.
"""


def test_frontmatter_is_stripped_and_parsed():
    doc = parse_frontmatter(FRONTMATTER_DOC)
    assert doc.title == "Identity linking logic"
    assert doc.description == "How identities are linked across devices."
    assert not doc.body.lstrip().startswith("---")
    assert "solution: Experience Platform" not in doc.body


def test_title_falls_back_to_first_h1():
    doc = parse_frontmatter("# Just A Heading\n\nBody text.\n")
    assert doc.title == "Just A Heading"


def test_heading_path_tracks_nesting():
    """Sections must be deep enough to split, or one chunk swallows the document.

    A chunk is attributed to the heading path of its first substantive block --
    where the chunk starts -- so the deep path only appears once the deep
    section owns a chunk of its own.
    """
    filler = " ".join(["padding words to force a chunk boundary here"] * 12)
    raw = f"# Top\n\n{filler}\n\n## Middle\n\n{filler}\n\n### Deep\n\n{filler}\n"
    _, chunks = chunk_markdown(raw, "identity-service", target_tokens=140, overlap_ratio=0.0)
    paths = {c.heading_path for c in chunks}
    assert "Top > Middle > Deep" in paths, paths


def test_sibling_heading_pops_the_stack():
    """An H2 after a deeper H3 must not inherit the H3."""
    raw = "# Top\n\n## One\n\n### Deep\n\nx\n\n## Two\n\ny\n"
    _, chunks = chunk_markdown(raw, "xdm", target_tokens=6, overlap_ratio=0.0)
    assert any(c.heading_path == "Top > Two" for c in chunks), [c.heading_path for c in chunks]


def test_fenced_code_block_is_never_split():
    code = "\n".join(f'    "field_{i}": "value_{i}",' for i in range(80))
    raw = f"# Doc\n\nIntro.\n\n```json\n{{\n{code}\n}}\n```\n\nTrailer.\n"
    _, chunks = chunk_markdown(raw, "xdm", target_tokens=50, overlap_ratio=0.0)

    holders = [c for c in chunks if "```json" in c.text]
    assert len(holders) == 1, "fence appeared in more than one chunk"
    # Opening and closing fence must live in the same chunk.
    assert holders[0].text.count("```") == 2


def test_markdown_table_is_never_split():
    rows = "\n".join(f"| row{i} | value{i} | note{i} |" for i in range(60))
    raw = f"# Doc\n\n| A | B | C |\n| --- | --- | --- |\n{rows}\n\nAfter.\n"
    _, chunks = chunk_markdown(raw, "sources", target_tokens=40, overlap_ratio=0.0)

    holders = [c for c in chunks if "| --- |" in c.text]
    assert len(holders) == 1
    assert "row59" in holders[0].text, "table tail was separated from its header"


def test_oversized_block_becomes_its_own_chunk():
    big = "\n".join(f"line {i} of a very long sample" for i in range(200))
    raw = f"# Doc\n\n```text\n{big}\n```\n"
    _, chunks = chunk_markdown(raw, "sources", target_tokens=30, overlap_ratio=0.0)
    assert any(c.token_count > 30 for c in chunks), "oversized block was split"


def test_overlap_repeats_trailing_content():
    paras = "\n\n".join(f"Paragraph number {i} with some filler words in it." for i in range(30))
    raw = f"# Doc\n\n{paras}\n"
    _, with_overlap = chunk_markdown(raw, "profile", target_tokens=60, overlap_ratio=0.25)
    _, without = chunk_markdown(raw, "profile", target_tokens=60, overlap_ratio=0.0)

    assert len(with_overlap) >= len(without)
    # Overlap means total emitted text exceeds the source length.
    assert sum(len(c.text) for c in with_overlap) > sum(len(c.text) for c in without)


def test_embed_text_carries_context_prefix():
    _, chunks = chunk_markdown(FRONTMATTER_DOC, "identity-service", target_tokens=500)
    first = chunks[0]
    assert first.embed_text.startswith("identity-service | Identity linking logic")
    # The raw text shown to the user stays clean of the prefix.
    assert not first.text.startswith("identity-service |")


def test_chunks_are_indexed_in_order():
    raw = "# D\n\n" + "\n\n".join(f"Para {i} text here." for i in range(40))
    _, chunks = chunk_markdown(raw, "rtcdp", target_tokens=30, overlap_ratio=0.0)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_document_yields_no_chunks():
    _, chunks = chunk_markdown("---\ntitle: Empty\n---\n\n", "xdm")
    assert chunks == []


def test_approx_tokens_is_monotonic():
    assert approx_tokens("a") >= 1
    assert approx_tokens("a" * 400) > approx_tokens("a" * 40)


def test_hidden_frontmatter_is_detected():
    """Adobe marks unpublished pages with hide/hidefromtoc; citing them 404s."""
    for flag in ("hide: true", "hidefromtoc: yes", "hide: yes", "hidefromtoc: true"):
        doc = parse_frontmatter(f"---\ntitle: X\n{flag}\n---\n\n# X\n\nBody.\n")
        assert doc.hidden, flag


def test_visible_documents_are_not_flagged():
    doc = parse_frontmatter("---\ntitle: X\ndescription: Y\n---\n\n# X\n\nBody.\n")
    assert not doc.hidden
