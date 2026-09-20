"""Central configuration.

Every tunable the Lambdas and the local scripts share lives here so that the
deployed artifact and the laptop iteration loop cannot drift apart. Values come
from environment variables because that is the only configuration channel a
Lambda container gets; CDK sets them at deploy time.
"""

from __future__ import annotations

import os

# --- Provider selection -----------------------------------------------------
# A single env var chooses the embedding implementation. CDK passes this through
# from one context value, so "which model embeds my text" is one decision made
# in one place rather than a constant scattered across ingest and query.
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local").strip().lower()

# Generation is off by default. The query path ships as retrieval-only; the
# ChatProvider interface exists so turning this on is a config change, not a
# rewrite. See README "Design Tradeoffs" for why retrieval-only is defensible.
ENABLE_GENERATION = os.environ.get("ENABLE_GENERATION", "false").strip().lower() == "true"
CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "none").strip().lower()

# --- Storage ----------------------------------------------------------------
TABLE_NAME = os.environ.get("TABLE_NAME", "")
CORPUS_BUCKET = os.environ.get("CORPUS_BUCKET", "")

# --- Chunking ---------------------------------------------------------------
# ~500 tokens balances two failure modes: chunks too small lose the surrounding
# explanation a passage depends on, chunks too large dilute the embedding so
# cosine similarity stops discriminating between topics.
CHUNK_TARGET_TOKENS = int(os.environ.get("CHUNK_TARGET_TOKENS", "500"))

# 15% overlap so a passage split across a boundary still appears whole in one of
# the two neighbouring chunks.
CHUNK_OVERLAP_RATIO = float(os.environ.get("CHUNK_OVERLAP_RATIO", "0.15"))

# --- Retrieval --------------------------------------------------------------
TOP_K = int(os.environ.get("TOP_K", "4"))

# Below this cosine score we refuse rather than return a bad passage. A pre-sales
# tool that confidently cites the wrong doc is worse than one that says it does
# not know, so the default leans toward refusing.
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.30"))

# The query Lambda caches all vectors in module scope across warm invocations.
# Set to "false" to measure what the cache is actually buying (see benchmark.py).
ENABLE_VECTOR_CACHE = os.environ.get("ENABLE_VECTOR_CACHE", "true").strip().lower() == "true"

# --- Citations --------------------------------------------------------------
EXPERIENCE_LEAGUE_BASE = "https://experienceleague.adobe.com/en/docs/experience-platform/"

REFUSAL_TEXT = "That is not covered in the Adobe documentation I have indexed"
