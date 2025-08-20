"""Protocol definitions for better architecture and testability."""

from typing import Any
from typing import Protocol

import numpy as np


class DatabaseManagerProtocol(Protocol):
    """Protocol for database managers."""

    def add_document(self, filename: str, pages: int) -> int:
        """Add a document and return its ID."""
        ...

    def add_chunk(self, doc_id: int, page: int, position: int, text: str) -> int:
        """Add a chunk and return its ID."""
        ...

    def store_tokens(self, chunk_id: int, tokens: dict[str, float]) -> None:
        """Store tokens for a chunk."""
        ...


class SearcherProtocol(Protocol):
    """Protocol for search implementations."""

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search and return (chunk_id, score) pairs."""
        ...


class EmbeddingManagerProtocol(Protocol):
    """Protocol for embedding managers."""

    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for query text."""
        ...

    def embed_and_store_chunks(self, chunk_data: list[dict[str, Any]]) -> None:
        """Generate and store embeddings for chunks."""
        ...


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients."""

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Generate chat completion."""
        ...


class RetrieverProtocol(Protocol):
    """Protocol for retrieval systems."""

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 8,
        rerank_k: int = 20,
        threshold: float = 0.18,
        alpha: float | None = None,
        lambda_param: float | None = None,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """Retrieve relevant chunks with metadata."""
        ...


class ProcessorProtocol(Protocol):
    """Protocol for document processors."""

    def extract_text_by_page(self, content: bytes) -> list[tuple[int, str]]:
        """Extract text from document by page."""
        ...


class ChunkerProtocol(Protocol):
    """Protocol for text chunkers."""

    def chunk_text(self, pages_text: list[tuple[int, str]]) -> list[dict[str, Any]]:
        """Chunk text into segments."""
        ...
