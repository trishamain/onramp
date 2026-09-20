"""Provider selection and the index/query consistency guard."""

from __future__ import annotations

from onramp.providers.base import EmbeddingProvider, ProviderMismatchError
from onramp.providers.bedrock import BedrockEmbeddingProvider
from onramp.providers.local import LocalEmbeddingProvider
from onramp.providers.null import NullEmbeddingProvider

# Registry rather than an if/elif chain so the valid set is introspectable --
# error messages can list what was actually available.
_PROVIDERS = {
    "local": LocalEmbeddingProvider,
    "bedrock": BedrockEmbeddingProvider,
    "null": NullEmbeddingProvider,
}


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    """Build the configured provider.

    `name` overrides the environment, which is what the local scripts and tests
    use. Everything else reads EMBEDDING_PROVIDER, set by CDK from a single
    context value.
    """
    if name is None:
        from onramp import config  # noqa: PLC0415  (import here so tests can monkeypatch env first)

        name = config.EMBEDDING_PROVIDER

    key = (name or "").strip().lower()
    if key not in _PROVIDERS:
        raise ValueError(f"unknown EMBEDDING_PROVIDER {name!r}; expected one of {available_providers()}")
    return _PROVIDERS[key]()


def assert_provider_matches(
    query_provider: EmbeddingProvider,
    indexed_provider_name: str,
    indexed_dimensions: int,
) -> None:
    """Refuse to run a query whose provider disagrees with the index.

    WHY THIS IS FATAL RATHER THAN A WARNING
    ---------------------------------------
    Cosine similarity between vectors from two different models is arithmetic
    that succeeds and means nothing. MiniLM's 384-dim space and Titan's 1024-dim
    space share no axes; a dot product across them is not a weaker signal, it is
    noise wearing the costume of a score.

    The dangerous case is not a dimension mismatch -- that raises a shape error
    in numpy and is self-announcing. It is a same-dimension, different-model
    mismatch, or a table that was partially re-ingested after a provider switch.
    Those return plausible-looking scores and silently wrong passages, and a
    pre-sales tool that cites the wrong Adobe doc with a confident score is worse
    than one that refuses to start.

    So: every chunk records embedding_provider and embedding_dimensions at write
    time, and the query path compares before it scores anything. Mixed-provider
    tables are detectable rather than merely unlikely.
    """
    if indexed_provider_name != query_provider.model_id:
        raise ProviderMismatchError(
            f"index was built with {indexed_provider_name!r} but this query uses "
            f"{query_provider.model_id!r}. Re-ingest the corpus with the current "
            f"provider, or set EMBEDDING_PROVIDER back to the one that built the index."
        )
    if indexed_dimensions != query_provider.dimensions:
        raise ProviderMismatchError(
            f"index has {indexed_dimensions}-dim vectors but this query produces "
            f"{query_provider.dimensions}-dim. Re-ingest required."
        )
