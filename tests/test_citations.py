"""URL mapping tests.

The headline test drives every question in the eval set through the mapper and
asserts it reproduces the hand-researched URL. That is the regression guard for
the whole citation feature -- if someone "simplifies" the mapper back to the
naive rule, seven of these fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onramp.citations import (
    TocIndex,
    build_url,
    naive_url,
    parse_toc,
    product_area,
)
from onramp.config import EXPERIENCE_LEAGUE_BASE

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "evals" / "golden.json"

IDENTITY_TOC = """---
audience: user
---

# Adobe Experience Platform Identity Service {#identity}

- [Identity Service overview](home.md)
- [Identity Service and Real-Time Customer Profile](identity-and-profile.md)
- Features {#features}
  - [Identity namespace](./features/namespaces.md)
  - [Identity linking logic](./features/identity-linking-logic.md)
  - Identity Graph Linking Rules {#identity-graph-linking-rules}
    - [Overview](identity-graph-linking-rules/overview.md)
- [Guardrails](guardrails.md)
"""

SOURCES_TOC = """# Sources {#sources}

- [Sources overview](home.md)
- Available source connectors {#connectors}
  - Databases {#databases}
    - [Snowflake](connectors/databases/snowflake.md)
- UI tutorials {#ui-tutorials}
  - Create {#create}
    - Streaming {#streaming}
      - [HTTP](tutorials/ui/create/streaming/http.md)
"""


def test_guide_anchor_renames_directory():
    """identity-service/ publishes as /identity/ because the TOC says so."""
    toc = parse_toc(IDENTITY_TOC, "identity-service")
    assert toc.guide_anchor == "identity"
    url = build_url("help/identity-service/guardrails.md", toc)
    assert url == EXPERIENCE_LEAGUE_BASE + "identity/guardrails"


def test_section_anchor_is_inserted():
    """A doc nested under `Features {#features}` gains a features/ segment."""
    toc = parse_toc(IDENTITY_TOC, "identity-service")
    url = build_url("help/identity-service/features/namespaces.md", toc)
    assert url == EXPERIENCE_LEAGUE_BASE + "identity/features/namespaces"


def test_nested_sections_accumulate():
    """Two levels of TOC section produce two URL segments."""
    toc = parse_toc(IDENTITY_TOC, "identity-service")
    url = build_url("help/identity-service/identity-graph-linking-rules/overview.md", toc)
    assert url == EXPERIENCE_LEAGUE_BASE + "identity/features/identity-graph-linking-rules/overview"


def test_toc_anchor_overrides_filesystem_path():
    """The q18 case: files live in tutorials/ui/ but publish under ui-tutorials/."""
    toc = parse_toc(SOURCES_TOC, "sources")
    url = build_url("help/sources/tutorials/ui/create/streaming/http.md", toc)
    assert url == EXPERIENCE_LEAGUE_BASE + "sources/ui-tutorials/create/streaming/http"


def test_asterisk_bullets_are_parsed():
    """Regression: privacy-service/TOC.md uses `*` where other guides use `-`.

    Matching only `-` made the whole guide fall through to the naive rule, which
    produced /privacy-service/home instead of /privacy/home. Silent, because the
    fallback returns a plausible URL rather than failing.
    """
    toc_text = """# Adobe Experience Platform Privacy Service {#privacy}

* [Privacy Service overview](./home.md)
* Privacy Service API {#api}
  * [Overview](./api/overview.md)
"""
    toc = parse_toc(toc_text, "privacy-service")
    assert toc.guide_anchor == "privacy"
    assert build_url("help/privacy-service/home.md", toc) == EXPERIENCE_LEAGUE_BASE + "privacy/home"
    assert (
        build_url("help/privacy-service/api/overview.md", toc)
        == EXPERIENCE_LEAGUE_BASE + "privacy/api/overview"
    )


def test_naive_fallback_when_doc_absent_from_toc():
    """Unknown documents still get a plausible URL rather than nothing."""
    toc = parse_toc(SOURCES_TOC, "sources")
    url = build_url("help/sources/connectors/databases/postgres.md", toc)
    assert url == EXPERIENCE_LEAGUE_BASE + "sources/connectors/databases/postgres"


def test_naive_rule_strips_prefix_and_suffix():
    assert naive_url("help/xdm/schema/composition.md") == (EXPERIENCE_LEAGUE_BASE + "xdm/schema/composition")


def test_build_url_without_toc_uses_naive():
    assert build_url("help/profile/event-expirations.md", None) == (
        EXPERIENCE_LEAGUE_BASE + "profile/event-expirations"
    )


def test_empty_toc_index_falls_back():
    toc = TocIndex(guide_dir="destinations", guide_anchor="destinations")
    assert build_url("help/destinations/destination-sdk/overview.md", toc) == (
        EXPERIENCE_LEAGUE_BASE + "destinations/destination-sdk/overview"
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("help/identity-service/guardrails.md", "identity-service"),
        ("help/xdm/schema/composition.md", "xdm"),
        ("segmentation/ui/segment-builder.md", "segmentation"),
    ],
)
def test_product_area(path, expected):
    assert product_area(path) == expected


@pytest.mark.skipif(not GOLDEN.exists(), reason="evals/golden.json not built yet")
def test_every_golden_url_reproduced():
    """End-to-end: the mapper must reproduce all 20 researched URLs.

    TOC text is embedded in golden.json as `_toc_fixtures` at build time so this
    test does not depend on a cloned repo being present on disk.
    """
    data = json.loads(GOLDEN.read_text())
    fixtures = data.get("_toc_fixtures", {})
    indexes = {d: parse_toc(text, d) for d, text in fixtures.items()}

    failures = []
    for q in data["questions"]:
        path = q["expected_doc_path"]
        guide = product_area(path)
        got = build_url(path, indexes.get(guide))
        if got != q["expected_url"]:
            failures.append(f"{q['id']}: got {got} want {q['expected_url']}")

    assert not failures, "URL mapping regressions:\n" + "\n".join(failures)
