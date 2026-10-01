"""The only module that talks to Amazon Bedrock."""
import json
import time

import boto3
from botocore.exceptions import ClientError

from app import config

_client = None

_RETRYABLE = {"ThrottlingException", "ModelTimeoutException", "ServiceUnavailableException"}


def _get_client():
    """Create the Bedrock Runtime client lazily (credentials come from the AWS default chain)."""
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)
    return _client


def _call_with_retry(fn, attempts: int = 4):
    """Run fn(); retry with exponential backoff on throttling/timeouts."""
    for attempt in range(attempts):
        try:
            return fn()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            message = e.response["Error"]["Message"]
            if code in _RETRYABLE and attempt < attempts - 1:
                time.sleep(2**attempt)
                continue
            if code == "ValidationException" and "inference profile" in message.lower():
                raise RuntimeError(
                    f"Model '{config.LLM_MODEL_ID}' cannot be invoked on-demand in "
                    f"{config.AWS_REGION}; it needs an inference profile. Set LLM_MODEL_ID "
                    "to the profile ID, e.g. 'us.amazon.nova-lite-v1:0'. "
                    f"AWS said: {message}"
                ) from e
            raise


def embed(text: str) -> list[float]:
    """Return a normalized Titan Text Embeddings V2 vector for text."""
    body = json.dumps(
        {"inputText": text, "dimensions": config.EMBED_DIM, "normalize": True}
    )

    def call():
        resp = _get_client().invoke_model(
            modelId=config.EMBED_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(resp["body"].read())["embedding"]

    return _call_with_retry(call)


def generate(prompt: str, max_tokens: int = 512) -> str:
    """Send prompt to Nova via the Converse API and return the reply text."""

    def call():
        resp = _get_client().converse(
            modelId=config.LLM_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        return resp["output"]["message"]["content"][0]["text"]

    return _call_with_retry(call)
