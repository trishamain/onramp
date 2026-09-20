from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock

import pytest

from onramp import config
from onramp.providers.base import EmbeddingProvider, ProviderMismatchError
from onramp.providers.bedrock import BedrockEmbeddingProvider
from onramp.providers.factory import (
    assert_provider_matches,
    available_providers,
    get_embedding_provider,
)
from onramp.providers.null import NullEmbeddingProvider

# --- Null provider ----------------------------------------------------------


def test_null_is_deterministic():
    a = NullEmbeddingProvider().embed(["identity graph linking rules"])
    b = NullEmbeddingProvider().embed(["identity graph linking rules"])
    assert a == b


def test_null_differs_across_inputs():
    [a], [b] = NullEmbeddingProvider().embed(["alpha"]), NullEmbeddingProvider().embed(["beta"])
    assert a != b


def test_null_vectors_are_unit_length():
    (vec,) = NullEmbeddingProvider().embed(["some text"])
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_null_respects_dimensions():
    provider = NullEmbeddingProvider(dimensions=1024)
    (vec,) = provider.embed(["x"])
    assert len(vec) == 1024


def test_null_spans_negative_and_positive():
    """Vectors confined to one orthant would make everything look similar."""
    (vec,) = NullEmbeddingProvider().embed(["a reasonably long piece of text"])
    assert min(vec) < 0 < max(vec)


def test_null_batch_order_is_preserved():
    texts = ["one", "two", "three"]
    out = NullEmbeddingProvider().embed(texts)
    assert out == [NullEmbeddingProvider().embed([t])[0] for t in texts]


# --- Protocol conformance ---------------------------------------------------


def test_implementations_satisfy_the_protocol():
    assert isinstance(NullEmbeddingProvider(), EmbeddingProvider)
    assert isinstance(BedrockEmbeddingProvider(client=MagicMock()), EmbeddingProvider)


# --- Factory / selection ----------------------------------------------------


def test_available_providers():
    assert available_providers() == ["bedrock", "local", "null"]


def test_explicit_name_wins():
    assert isinstance(get_embedding_provider("null"), NullEmbeddingProvider)


def test_selection_is_case_and_space_insensitive():
    assert isinstance(get_embedding_provider("  NULL  "), NullEmbeddingProvider)


def test_unknown_provider_raises_with_the_valid_set():
    with pytest.raises(ValueError, match="unknown EMBEDDING_PROVIDER") as exc:
        get_embedding_provider("titan")
    assert "null" in str(exc.value)


def test_env_var_selects_provider(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "null")
    assert isinstance(get_embedding_provider(), NullEmbeddingProvider)


def test_default_is_local(monkeypatch):
    """Guards the documented default without constructing it (torch is heavy)."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    importlib.reload(config)
    assert config.EMBEDDING_PROVIDER == "local"


# --- The mixed-provider guard ----------------------------------------------


def test_matching_provider_passes():
    p = NullEmbeddingProvider()
    assert assert_provider_matches(p, p.model_id, p.dimensions) is None


def test_different_model_same_dimensions_is_rejected():
    """The dangerous case: shapes agree, so numpy would happily return noise."""
    p = NullEmbeddingProvider(dimensions=384)
    with pytest.raises(ProviderMismatchError, match="Re-ingest"):
        assert_provider_matches(p, "sentence-transformers/all-MiniLM-L6-v2", 384)


def test_dimension_mismatch_is_rejected():
    p = NullEmbeddingProvider(dimensions=384)
    with pytest.raises(ProviderMismatchError, match="1024-dim"):
        assert_provider_matches(p, p.model_id, 1024)


# --- Bedrock against a mocked client ---------------------------------------


def _mock_bedrock(vector):
    client = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps({"embedding": vector}).encode()
    client.invoke_model.return_value = {"body": body}
    return client


def test_bedrock_parses_the_response_envelope():
    client = _mock_bedrock([0.1] * 1024)
    out = BedrockEmbeddingProvider(client=client).embed(["hello"])
    assert len(out) == 1
    assert len(out[0]) == 1024


def test_bedrock_sends_expected_request():
    client = _mock_bedrock([0.0] * 1024)
    BedrockEmbeddingProvider(client=client).embed(["identity stitching"])

    kwargs = client.invoke_model.call_args.kwargs
    assert kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
    payload = json.loads(kwargs["body"])
    assert payload["inputText"] == "identity stitching"
    assert payload["dimensions"] == 1024


def test_bedrock_calls_once_per_text():
    """Titan has no batch endpoint; this documents the per-item cost."""
    client = _mock_bedrock([0.0] * 1024)
    BedrockEmbeddingProvider(client=client).embed(["a", "b", "c"])
    assert client.invoke_model.call_count == 3


def test_bedrock_declares_1024_dimensions():
    assert BedrockEmbeddingProvider(client=MagicMock()).dimensions == 1024
