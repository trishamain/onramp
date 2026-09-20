"""Local sentence-transformers embeddings. This is what ships.

MODEL CHOICE IS MEASURED, NOT ASSUMED
-------------------------------------
The default is BAAI/bge-small-en-v1.5, chosen by ablation over 1,102 real
documents and the 20-question golden set (scripts/ablate_retrieval.py):

    all-MiniLM-L6-v2   @500   recall@3 20%     <- the obvious first choice
    all-MiniLM-L6-v2   @220   recall@3 30%
    multi-qa-MiniLM    @500   recall@3 25%
    multi-qa-MiniLM    @220   recall@3 25%
    bge-small-en-v1.5  @500   recall@3 35%     <- default
    bge-small-en-v1.5  @220   recall@3 20%
    e5-small-v2        @500   recall@3 20%
    e5-small-v2        @220   recall@3 15%

Two things that measurement taught, both of which are silent failures:

1. CHUNK SIZE MUST MATCH THE MODEL'S WINDOW. all-MiniLM truncates at 256 word
   pieces, so 500-token chunks lost half their text with no error raised --
   which is why shrinking chunks helped it and hurt bge, whose window is 512.
   `max_seq_tokens` is therefore a property of the provider, not a free-floating
   config value.
2. bge WANTS AN INSTRUCTION PREFIX ON THE QUERY SIDE ONLY. Embedding a question
   without it measurably degrades retrieval, and nothing warns you.

Every model here is 384 dimensions, so switching between them is a config
change with no storage or schema consequence.
"""

from __future__ import annotations

import os
from typing import Any

# Model registry: name -> (dimensions, max sequence length, query prefix).
# The query prefix is part of the model's contract, not a tuning knob.
MODELS: dict[str, tuple[int, int, str]] = {
    "BAAI/bge-small-en-v1.5": (
        384,
        512,
        "Represent this sentence for searching relevant passages: ",
    ),
    "sentence-transformers/all-MiniLM-L6-v2": (384, 256, ""),
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": (384, 512, ""),
    "intfloat/e5-small-v2": (384, 512, "query: "),
}

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Where the Dockerfile bakes the model; the image sets
# SENTENCE_TRANSFORMERS_HOME to match and ships the weights inside, so Lambda
# never reaches the network at cold start.
CONTAINER_MODEL_HOME = "/opt/models"


def _resolve_model_home(explicit: str | None) -> str | None:
    """Pick a cache directory valid on both a laptop and in the image.

    The writability check matters: hardcoding /opt/models made every laptop run
    fail with PermissionError, because that path only exists inside the image.
    """
    if explicit:
        return explicit
    env = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.environ.get("HF_HOME")
    if env:
        return env
    if os.path.isdir(CONTAINER_MODEL_HOME) and os.access(CONTAINER_MODEL_HOME, os.W_OK):
        return CONTAINER_MODEL_HOME
    return None


class LocalEmbeddingProvider:
    """CPU sentence-transformers embeddings, 384 dims."""

    def __init__(self, model_name: str | None = None, model_home: str | None = None) -> None:
        self.model_id = model_name or os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
        if self.model_id not in MODELS:
            raise ValueError(f"unknown model {self.model_id!r}; expected one of {sorted(MODELS)}")
        self.dimensions, self.max_seq_tokens, self.query_prefix = MODELS[self.model_id]
        self._model_home = _resolve_model_home(model_home)
        self._model: Any | None = None

    def _load(self) -> Any:
        """Import and construct lazily.

        Deliberately not a module-level import: CI and the unit suite select the
        Null provider, and importing sentence_transformers drags in torch.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model_id, cache_folder=self._model_home)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed passages. No prefix -- see embed_query for the asymmetric side."""
        if not texts:
            return []
        model = self._load()
        # normalize_embeddings=True makes cosine similarity a plain dot product
        # downstream, which is the hot loop in the query path.
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a question, applying the model's query-side instruction prefix.

        Split from embed() because bge and e5 are asymmetric: the prefix belongs
        on the query and must NOT go on the passages. Sending both through one
        symmetric method silently costs recall.
        """
        return self.embed([self.query_prefix + text])[0]
