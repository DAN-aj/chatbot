"""Embedding adapters and bounded retry for external services."""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_PROVIDER

LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENAI_MODEL = "text-embedding-3-small"
FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_NAME = {
    "local": LOCAL_MODEL, "openai": OPENAI_MODEL,
    "fastembed": FASTEMBED_MODEL,
}.get(EMBEDDING_PROVIDER, "unsupported")
MODEL_ID = f"{EMBEDDING_PROVIDER}:{_MODEL_NAME}|parent_v1"


def retryable(exc: BaseException) -> bool:
    """Retry only rate limits, transient server failures, and transport errors."""
    code = getattr(exc, "status_code", getattr(exc, "code", None))
    detail = str(exc).lower()
    if any(term in detail for term in (
            "insufficient_quota", "credit_balance_exhausted", "no credits remaining")):
        return False
    if code == 429 or isinstance(code, int) and 500 <= code < 600:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in {
        "APIConnectionError", "APITimeoutError", "ConnectError",
        "ReadTimeout", "ConnectTimeout", "RemoteProtocolError"}


def external_retry():
    """Construct a new full-jitter policy for one external API operation."""
    from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential
    return retry(reraise=True, retry=retry_if_exception(retryable),
                 wait=wait_random_exponential(multiplier=1, max=45),
                 stop=stop_after_attempt(5))


class Embedder(ABC):
    model_id: str
    dimensions: int
    batch_size: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed one bounded batch of child search text."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a question in the same vector space."""


class LocalTransformerEmbedder(Embedder):
    model_id = LOCAL_MODEL
    dimensions = 384
    batch_size = 16

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from .transform import embed_many
        return embed_many(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder(Embedder):
    model_id = OPENAI_MODEL
    dimensions = 1536
    batch_size = EMBEDDING_BATCH_SIZE

    def __init__(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(max_retries=0, timeout=60)

    @external_retry()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_id, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FastEmbedAdapter(Embedder):
    model_id = FASTEMBED_MODEL
    dimensions = 384
    batch_size = EMBEDDING_BATCH_SIZE

    def __init__(self) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("Install fastembed to use EMBEDDING_PROVIDER=fastembed") from exc
        self.model = TextEmbedding(model_name=self.model_id)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Choose a local or hosted implementation through configuration."""
    factories = {
        "local": LocalTransformerEmbedder,
        "openai": OpenAIEmbedder,
        "fastembed": FastEmbedAdapter,
    }
    if EMBEDDING_PROVIDER not in factories:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
    return factories[EMBEDDING_PROVIDER]()
