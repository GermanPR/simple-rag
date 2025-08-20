"""Mistral API client for embeddings and chat completions."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any
from typing import cast
from typing import final

import numpy as np
from mistralai import Mistral
from mistralai.models import MessagesTypedDict
from mistralai.models.embeddingresponse import EmbeddingResponse
from pydantic import BaseModel

from app.core.config import config
from app.core.exceptions import APIError
from app.core.exceptions import ConfigurationError
from app.core.monitoring import monitor

logger = logging.getLogger(__name__)


def _convert_messages(messages: Sequence[dict[str, str]]) -> list[MessagesTypedDict]:
    """Convert dict messages to Mistral-compatible format.

    This is safe because our dict[str, str] with 'role' and 'content' keys
    is structurally compatible with MessagesTypedDict.
    """
    return cast(list[MessagesTypedDict], list(messages))


# Keep the existing exception for backward compatibility
class MistralAPIError(APIError):
    """Exception for Mistral API errors."""


@final
class MistralClient:
    """Client for interacting with Mistral AI API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.MISTRAL_API_KEY
        self.base_url = config.MISTRAL_BASE_URL
        self.embed_model = config.MISTRAL_EMBED_MODEL
        self.chat_model = config.MISTRAL_CHAT_MODEL

        if not self.api_key:
            raise ConfigurationError("Mistral API key is required")

        # Initialize the official Mistral AI SDK client
        self.client = Mistral(api_key=self.api_key)

    @monitor
    def get_embeddings(
        self, texts: list[str], model: str | None = None
    ) -> list[np.ndarray]:
        """
        Get embeddings for a list of texts using the official Mistral AI client.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to configured embed model)

        Returns:
            List of embedding vectors as numpy arrays

        Raises:
            MistralAPIError: If the API request fails
        """
        if not texts:
            return []

        model = model or self.embed_model

        try:
            embeddings_response: EmbeddingResponse = self.client.embeddings.create(
                model=model,
                inputs=texts,
            )

            embeddings: list[np.ndarray] = []
            for item in embeddings_response.data:
                embeddings.append(np.array(item.embedding, dtype=np.float32))

            return embeddings

        except Exception as e:
            logger.error(f"Mistral embeddings API request failed: {e}")
            raise MistralAPIError(f"Embeddings API request failed: {e}") from e

    @monitor
    async def get_embeddings_async(
        self, texts: list[str], model: str | None = None
    ) -> list[np.ndarray]:
        """
        Async version of get_embeddings.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to configured embed model)

        Returns:
            List of embedding vectors as numpy arrays

        Raises:
            MistralAPIError: If the API request fails
        """
        return await asyncio.to_thread(self.get_embeddings, texts, model)

    def get_single_embedding(self, text: str, model: str | None = None) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed
            model: Model to use (defaults to configured embed model)

        Returns:
            Embedding vector as numpy array
        """
        embeddings: list[np.ndarray] = self.get_embeddings([text], model)
        return embeddings[0] if embeddings else np.ndarray([])

    async def get_single_embedding_async(
        self, text: str, model: str | None = None
    ) -> np.ndarray:
        """
        Async version of get_single_embedding.

        Args:
            text: Text to embed
            model: Model to use (defaults to configured embed model)

        Returns:
            Embedding vector as numpy array
        """
        embeddings: list[np.ndarray] = await self.get_embeddings_async([text], model)
        return embeddings[0] if embeddings else np.ndarray([])

    @monitor
    def chat_completion(
        self,
        messages: Sequence[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """
        Get chat completion using the official Mistral AI client.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to configured chat model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text

        Raises:
            MistralAPIError: If the API request fails
        """
        model = model or self.chat_model

        try:
            chat_response = self.client.chat.complete(
                model=model,
                messages=_convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = chat_response.choices[0].message.content
            if content is None:
                return ""
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()

        except Exception as e:
            logger.error(f"Mistral chat API request failed: {e}")
            raise MistralAPIError(f"Chat API request failed: {e}") from e

    @monitor
    async def chat_completion_async(
        self,
        messages: Sequence[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """
        Async version of chat_completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to configured chat model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text

        Raises:
            MistralAPIError: If the API request fails
        """
        return await asyncio.to_thread(
            self.chat_completion, messages, model, temperature, max_tokens
        )

    @monitor
    def structured_chat_completion(
        self,
        messages: Sequence[dict[str, str]],
        response_format: type[BaseModel],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> BaseModel:
        """
        Get structured chat completion using the official Mistral AI client.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Pydantic model class for structured output
            model: Model to use (defaults to configured chat model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed Pydantic model instance

        Raises:
            MistralAPIError: If the API request fails
        """
        model = model or self.chat_model

        try:
            chat_response = self.client.chat.parse(
                model=model,
                messages=_convert_messages(messages),
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Return the parsed Pydantic object
            if not chat_response.choices:
                raise MistralAPIError("No choices returned from API")
            choice = chat_response.choices[0]
            if choice.message is None or choice.message.parsed is None:
                raise MistralAPIError("No parsed response received")
            return choice.message.parsed

        except Exception as e:
            logger.error(f"Mistral structured chat API request failed: {e}")
            raise MistralAPIError(f"Structured chat API request failed: {e}") from e

    @monitor
    async def structured_chat_completion_async(
        self,
        messages: Sequence[dict[str, str]],
        response_format: type[BaseModel],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> BaseModel:
        """
        Async version of structured_chat_completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Pydantic model class for structured output
            model: Model to use (defaults to configured chat model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed Pydantic model instance

        Raises:
            MistralAPIError: If the API request fails
        """
        return await asyncio.to_thread(
            self.structured_chat_completion,
            messages,
            response_format,
            model,
            temperature,
            max_tokens,
        )


class EmbeddingManager:
    """Manages text embeddings using Mistral API."""

    def __init__(self, db_manager, mistral_client: MistralClient | None = None):
        self.db_manager = db_manager
        self.mistral_client = mistral_client or MistralClient()

    def embed_and_store_chunks(self, chunk_data: list[dict[str, Any]]):
        """
        Generate embeddings for chunks and store them in the database.

        Args:
            chunk_data: List of chunk dictionaries with 'chunk_id' and 'text'
        """
        if not chunk_data:
            return

        # Extract texts and chunk IDs
        texts = [chunk["text"] for chunk in chunk_data]
        chunk_ids = [chunk["chunk_id"] for chunk in chunk_data]

        try:
            # Get embeddings in batch
            embeddings = self.mistral_client.get_embeddings(texts)

            # Store embeddings in database
            for chunk_id, embedding in zip(chunk_ids, embeddings, strict=False):
                self.db_manager.store_embedding(chunk_id, embedding)

            logger.info(f"Successfully embedded and stored {len(chunk_data)} chunks")

        except MistralAPIError as e:
            logger.error(f"Failed to embed chunks: {e}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query text to embed

        Returns:
            Query embedding as numpy array
        """
        try:
            return self.mistral_client.get_single_embedding(query)
        except MistralAPIError as e:
            logger.error(f"Failed to embed query: {e}")
            raise

    async def embed_query_async(self, query: str) -> np.ndarray:
        """
        Async version of embed_query.

        Args:
            query: Query text to embed

        Returns:
            Query embedding as numpy array
        """
        try:
            return await self.mistral_client.get_single_embedding_async(query)
        except MistralAPIError as e:
            logger.error(f"Failed to embed query: {e}")
            raise


class AsyncEmbeddingManager:
    """Async version of EmbeddingManager."""

    def __init__(self, async_db_manager, mistral_client: MistralClient | None = None):
        self.db_manager = async_db_manager
        self.mistral_client = mistral_client or MistralClient()

    async def embed_and_store_chunks(self, chunk_data: list[dict[str, Any]]):
        """
        Generate embeddings for chunks and store them in the database.

        Args:
            chunk_data: List of chunk dictionaries with 'chunk_id' and 'text'
        """
        if not chunk_data:
            return

        # Extract texts and chunk IDs
        texts = [chunk["text"] for chunk in chunk_data]
        chunk_ids = [chunk["chunk_id"] for chunk in chunk_data]

        try:
            # Get embeddings in batch
            embeddings = await self.mistral_client.get_embeddings_async(texts)

            # Store embeddings in database
            for chunk_id, embedding in zip(chunk_ids, embeddings, strict=False):
                await self.db_manager.store_embedding(chunk_id, embedding)

            logger.info(f"Successfully embedded and stored {len(chunk_data)} chunks")

        except MistralAPIError as e:
            logger.error(f"Failed to embed chunks: {e}")
            raise

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query text to embed

        Returns:
            Query embedding as numpy array
        """
        try:
            return await self.mistral_client.get_single_embedding_async(query)
        except MistralAPIError as e:
            logger.error(f"Failed to embed query: {e}")
            raise


# Global instances
_mistral_client = None
_embedding_manager = None
_async_embedding_manager = None


def get_mistral_client() -> MistralClient:
    """Get a global Mistral client instance."""
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = MistralClient()
    return _mistral_client


def get_embedding_manager(db_manager) -> EmbeddingManager:
    """Get a global embedding manager instance."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(db_manager, get_mistral_client())
    return _embedding_manager


def get_async_embedding_manager(async_db_manager) -> AsyncEmbeddingManager:
    """Get a global async embedding manager instance."""
    global _async_embedding_manager
    if _async_embedding_manager is None:
        _async_embedding_manager = AsyncEmbeddingManager(
            async_db_manager, get_mistral_client()
        )
    return _async_embedding_manager
