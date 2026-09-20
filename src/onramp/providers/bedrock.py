"""Amazon Titan Text Embeddings V2 via Bedrock.

NOT the default and NOT exercised by any deploy. This account has a provisioning
defect: every Bedrock on-demand inference quota reads 0.0, so InvokeModel throws
ThrottlingException regardless of entitlement. Rather than wait on a support
case, the project ships on LocalEmbeddingProvider and keeps this class unit
tested against a mocked boto3 client, so the swap is a config change whenever
quota arrives.

Titan Text Embeddings V2 returns 1024 dimensions -- 2.7x MiniLM's 384. See the
README data-model note for the storage consequence before enabling this.
"""

from __future__ import annotations

import json
from typing import Any

MODEL_ID = "amazon.titan-embed-text-v2:0"


class BedrockEmbeddingProvider:
    """Titan V2 embeddings, 1024 dims."""

    model_id = MODEL_ID
    dimensions = 1024

    def __init__(self, client: Any | None = None, region: str = "us-east-1") -> None:
        # The client is injected so unit tests can pass a stub. Constructed
        # lazily otherwise, because importing this module must not require
        # credentials -- the factory imports every provider to build its table.
        self._client = client
        self._region = region

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3  # noqa: PLC0415

            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch.

        Titan's InvokeModel takes exactly one inputText per call -- there is no
        batch endpoint -- so a "batch" here is a loop. That is a real cost
        difference against MiniLM, which embeds a list in one forward pass, and
        it is part of why local is the default beyond the quota problem.
        """
        client = self._get_client()
        out: list[list[float]] = []
        for text in texts:
            response = client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text, "dimensions": self.dimensions, "normalize": True}),
            )
            payload = json.loads(response["body"].read())
            out.append(payload["embedding"])
        return out
