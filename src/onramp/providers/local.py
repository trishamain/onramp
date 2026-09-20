"""sentence-transformers all-MiniLM-L6-v2. This is what ships.

Chosen because it runs identically on an Apple Silicon laptop and inside an
arm64 Lambda container, needs no network at inference time, and has no quota. At
384 dimensions it is a third the storage of Titan's 1024 and measurably faster
to score in numpy.
"""

from __future__ import annotations

import os
from typing import Any

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Where the Dockerfile bakes the model. Set here as well as in the image so the
# laptop and the container resolve the same path, and so a missing env var fails
# loudly rather than silently reaching for the network.
DEFAULT_MODEL_HOME = "/opt/models"


class LocalEmbeddingProvider:
    """MiniLM embeddings, 384 dims, CPU only."""

    model_id = MODEL_NAME
    dimensions = 384

    def __init__(self, model_home: str | None = None) -> None:
        self._model_home = model_home or os.environ.get("SENTENCE_TRANSFORMERS_HOME", DEFAULT_MODEL_HOME)
        self._model: Any | None = None

    def _load(self) -> Any:
        """Import and construct lazily.

        Deliberately not a module-level import: CI and the unit suite select the
        Null provider, and importing sentence_transformers drags in torch, which
        is slow and enormous. Nothing should pay that cost unless it actually
        embeds something.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model_id, cache_folder=self._model_home)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
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
