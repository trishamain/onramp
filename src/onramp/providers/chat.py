"""ChatProvider stubs for GENERATED MODE.

Deliberately inert. The deploy must not depend on either implementation, so both
raise on use. They exist to prove the retrieval path and the generation path are
separable, and to make the shape of the change obvious to a reviewer.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You answer questions about Adobe Real-Time CDP using only the passages "
    "provided. Cite the doc_url for every claim. If the passages do not contain "
    "the answer, say so rather than inferring."
)


class BedrockChatProvider:
    """Claude via Bedrock. Not wired up -- see module docstring."""

    # Newer Claude models reject bare model IDs on InvokeModel and require a
    # cross-region inference profile, hence the "us." prefix.
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def __init__(self, client: Any | None = None, region: str = "us-east-1") -> None:
        self._client = client
        self._region = region

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError(
            "GENERATED MODE is off by default. Bedrock on-demand quota is 0.0 on "
            "this account; enabling this path requires quota plus the bedrock "
            "IAM grant behind the same CDK context flag."
        )


class AnthropicApiChatProvider:
    """Claude via the Anthropic API directly, bypassing Bedrock quota entirely.

    The escape hatch if Bedrock quota never arrives: it needs an API key in
    Secrets Manager rather than an IAM grant.
    """

    model_id = "claude-sonnet-4-5-20250929"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError(
            "GENERATED MODE is off by default. Wiring this requires an API key in "
            "Secrets Manager; no secret is provisioned by the default deploy."
        )
