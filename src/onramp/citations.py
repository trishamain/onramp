"""Repo path -> Experience League URL.

THE NAIVE RULE IS WRONG AND I MEASURED IT
-----------------------------------------
The obvious mapping is:

    help/<path>.md  ->  https://experienceleague.adobe.com/en/docs/experience-platform/<path>

Checked against the 20 hand-researched URLs in the eval set, that rule produces
13/20 correct and 7 wrong. The failures are not random -- they fall into three
groups, and all three come from one mechanism.

Experience League does not publish by filesystem path. It publishes by the
structure declared in each guide's TOC.md:

    # Adobe Experience Platform Identity Service {#identity}
    - Features {#features}
      - [Identity namespace](./features/namespaces.md)

The `{#anchor}` on the guide heading becomes the first URL segment, and each
nested TOC section contributes its own anchor. So:

  * help/identity-service/... publishes under /identity/...  (guide anchor)
  * help/privacy-service/home.md -> /privacy/home            (guide anchor)
  * help/sources/tutorials/ui/create/streaming/http.md
        -> /sources/ui-tutorials/create/streaming/http
    because the TOC section is `UI tutorials {#ui-tutorials}` even though the
    files live under tutorials/ui/.

Deriving the URL from TOC.md therefore fixes all seven with one rule instead of
seven special cases, and it keeps working for documents the eval set never
mentions. The naive rule stays as the fallback for files absent from a TOC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from onramp.config import EXPERIENCE_LEAGUE_BASE

# `# Title {#anchor}` -- the guide-level heading.
_GUIDE_HEADING = re.compile(r"^#\s+.*\{#(?P<anchor>[^}]+)\}\s*$")

# Bullet marker: this corpus mixes `-` and `*` across guides (identity-service
# uses `-`, privacy-service uses `*`), so both must be accepted. Matching only
# `-` silently drops a whole guide into the naive fallback.
_BULLET = r"[-*+]"

# `  - Section name {#anchor}` -- a TOC section with no link of its own.
_SECTION_LINE = re.compile(
    rf"^(?P<indent>\s*){_BULLET}\s+(?P<label>[^\[\]]*?)\s*\{{#(?P<anchor>[^}}]+)\}}\s*$"
)

# `  - [Link text](some/path.md)` -- a TOC leaf pointing at a document.
_LEAF_LINE = re.compile(rf"^(?P<indent>\s*){_BULLET}\s+\[[^\]]*\]\((?P<target>[^)]+\.md)\)")


def _normalise_target(target: str) -> str:
    """TOC links use ./x.md, x.md and ../x.md interchangeably."""
    target = target.split("#", 1)[0].strip()
    while target.startswith("./"):
        target = target[2:]
    return target


@dataclass
class TocIndex:
    """Maps a guide-relative markdown path to its published URL path."""

    guide_dir: str
    guide_anchor: str
    # e.g. {"tutorials/ui/create/streaming/http.md": "ui-tutorials/create/streaming"}
    section_path: dict[str, str] = field(default_factory=dict)

    def url_path_for(self, rel_path: str) -> str | None:
        """Return the published path (without base URL) or None if not in TOC."""
        rel_path = _normalise_target(rel_path)
        if rel_path not in self.section_path:
            return None
        stem = rel_path[: -len(".md")].split("/")[-1]
        sections = self.section_path[rel_path]
        parts = [self.guide_anchor]
        if sections:
            parts.append(sections)
        parts.append(stem)
        return "/".join(p for p in parts if p)


def parse_toc(toc_text: str, guide_dir: str) -> TocIndex:
    """Build a TocIndex from the text of one guide's TOC.md.

    Indentation carries the hierarchy, so we keep a stack of (indent, anchor)
    and pop it back whenever a line dedents. Sections contribute anchors; leaves
    consume whatever section stack is active at their depth.
    """
    guide_anchor = guide_dir
    for line in toc_text.splitlines():
        m = _GUIDE_HEADING.match(line)
        if m:
            guide_anchor = m.group("anchor")
            break

    index = TocIndex(guide_dir=guide_dir, guide_anchor=guide_anchor)
    stack: list[tuple[int, str]] = []  # (indent width, section anchor)

    for line in toc_text.splitlines():
        leaf = _LEAF_LINE.match(line)
        if leaf:
            indent = len(leaf.group("indent"))
            # Only sections strictly shallower than this leaf are its ancestors.
            active = [anchor for ind, anchor in stack if ind < indent]
            target = _normalise_target(leaf.group("target"))
            index.section_path[target] = "/".join(active)
            continue

        section = _SECTION_LINE.match(line)
        if section:
            indent = len(section.group("indent"))
            # Drop any sibling or deeper sections still on the stack.
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, section.group("anchor")))

    return index


def naive_url(repo_path: str) -> str:
    """The documented fallback: strip help/ and .md, prepend the base."""
    path = repo_path
    if path.startswith("help/"):
        path = path[len("help/") :]
    if path.endswith(".md"):
        path = path[: -len(".md")]
    return EXPERIENCE_LEAGUE_BASE + path


def build_url(repo_path: str, toc_index: TocIndex | None = None) -> str:
    """Map a repo path such as `help/xdm/schema/composition.md` to its URL.

    Uses the guide's TOC when one is available and contains the document; falls
    back to the naive rule otherwise. The fallback is correct for the majority
    of paths -- it is only wrong where a TOC anchor renames a segment.
    """
    if toc_index is not None:
        rel = repo_path
        prefix = f"help/{toc_index.guide_dir}/"
        if rel.startswith(prefix):
            rel = rel[len(prefix) :]
        url_path = toc_index.url_path_for(rel)
        if url_path:
            return EXPERIENCE_LEAGUE_BASE + url_path
    return naive_url(repo_path)


def product_area(repo_path: str) -> str:
    """Top-level directory under help/, used as a facet and in refusals."""
    parts = repo_path.split("/")
    if parts and parts[0] == "help":
        parts = parts[1:]
    return parts[0] if parts else "unknown"
