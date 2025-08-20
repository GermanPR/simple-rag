"""Semantic search using cosine similarity on embeddings."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SemanticSearcher:
    """Handles semantic search using cosine similarity on L2-normalized embeddings."""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self._cached_embeddings = None
        self._cached_chunk_ids = None

    def _load_embeddings_cache(self):
        """Load all embeddings into memory for efficient search."""
        try:
            embeddings_data = self.db_manager.get_all_embeddings()

            if not embeddings_data:
                self._cached_embeddings = np.array([])
                self._cached_chunk_ids = []
                return

            # Separate chunk IDs and embeddings
            chunk_ids, embeddings = zip(*embeddings_data, strict=False)

            # Stack embeddings into a matrix (num_chunks, embedding_dim)
            self._cached_embeddings = np.vstack(embeddings)
            self._cached_chunk_ids = list(chunk_ids)

            logger.info(f"Loaded {len(self._cached_chunk_ids)} embeddings into cache")

        except Exception as e:
            logger.error(f"Error loading embeddings cache: {e}")
            self._cached_embeddings = np.array([])
            self._cached_chunk_ids = []

    def _refresh_cache_if_needed(self):
        """Refresh the embeddings cache if it's not initialized."""
        if self._cached_embeddings is None:
            self._load_embeddings_cache()

    def search(
        self, query_embedding: np.ndarray, top_k: int = 20, threshold: float = 0.0
    ) -> list[tuple[int, float]]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding: L2-normalized query embedding
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (chunk_id, cosine_similarity) tuples, sorted by similarity descending
        """
        self._refresh_cache_if_needed()

        if self._cached_embeddings is None or self._cached_embeddings.size == 0:
            logger.warning("No embeddings found in database")
            return []

        try:
            # Ensure query embedding is L2-normalized
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                logger.warning("Query embedding has zero norm")
                return []

            normalized_query = query_embedding / query_norm  # unit length

            # Since stored embeddings are already L2-normalized,
            # cosine similarity = dot product
            if self._cached_embeddings is None:
                return []
            similarities = np.dot(self._cached_embeddings, normalized_query)

            # Filter by threshold
            valid_indices = np.where(similarities >= threshold)[0]

            if len(valid_indices) == 0:
                return []

            # Get top-k indices sorted by similarity (descending)
            sorted_indices = valid_indices[
                np.argsort(similarities[valid_indices])[::-1]
            ]
            top_indices = sorted_indices[:top_k]

            # Return chunk IDs and similarities
            if self._cached_chunk_ids is None:
                return []
            return [
                (self._cached_chunk_ids[idx], float(similarities[idx]))
                for idx in top_indices
            ]

        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    def search_with_scores_dict(
        self, query_embedding: np.ndarray, top_k: int = 20, threshold: float = 0.0
    ) -> dict[int, float]:
        """
        Search and return results as a dictionary mapping chunk_id to similarity score.

        Args:
            query_embedding: L2-normalized query embedding
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            Dictionary mapping chunk_id to cosine similarity score
        """
        results = self.search(query_embedding, top_k, threshold)
        return dict(results)

    def get_similarity(self, chunk_id: int, query_embedding: np.ndarray) -> float:
        """
        Get similarity score between a specific chunk and query.

        Args:
            chunk_id: ID of the chunk
            query_embedding: L2-normalized query embedding

        Returns:
            Cosine similarity score, or 0.0 if chunk not found
        """
        chunk_embedding = self.db_manager.get_embedding(chunk_id)

        if chunk_embedding is None:
            return 0.0

        try:
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return 0.0

            normalized_query = query_embedding / query_norm

            # Cosine similarity (chunk embedding is already L2-normalized)
            return float(np.dot(chunk_embedding, normalized_query))

        except Exception as e:
            logger.error(f"Error calculating similarity for chunk {chunk_id}: {e}")
            return 0.0

    def clear_cache(self):
        """Clear the embeddings cache to force reload."""
        self._cached_embeddings = None
        self._cached_chunk_ids = None
        logger.info("Embeddings cache cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the current cache state."""
        self._refresh_cache_if_needed()

        return {
            "cached_chunks": len(self._cached_chunk_ids)
            if self._cached_chunk_ids
            else 0,
            "embedding_dimension": self._cached_embeddings.shape[1]
            if self._cached_embeddings is not None and self._cached_embeddings.size > 0
            else 0,
            "cache_loaded": self._cached_embeddings is not None,
        }


class AsyncSemanticSearcher:
    """Async version of SemanticSearcher using cosine similarity on L2-normalized embeddings."""

    def __init__(self, async_db_manager):
        self.db_manager = async_db_manager
        self._cached_embeddings = None
        self._cached_chunk_ids = None

    async def _load_embeddings_cache(self):
        """Load all embeddings into memory for efficient search."""
        try:
            embeddings_data = await self.db_manager.get_all_embeddings()

            if not embeddings_data:
                self._cached_embeddings = np.array([])
                self._cached_chunk_ids = []
                return

            # Separate chunk IDs and embeddings
            chunk_ids, embeddings = zip(*embeddings_data, strict=False)

            # Stack embeddings into a matrix (num_chunks, embedding_dim)
            self._cached_embeddings = np.vstack(embeddings)
            self._cached_chunk_ids = list(chunk_ids)

            logger.info(f"Loaded {len(self._cached_chunk_ids)} embeddings into cache")

        except Exception as e:
            logger.error(f"Error loading embeddings cache: {e}")
            self._cached_embeddings = np.array([])
            self._cached_chunk_ids = []

    async def _refresh_cache_if_needed(self):
        """Refresh the embeddings cache if it's not initialized."""
        if self._cached_embeddings is None:
            await self._load_embeddings_cache()

    async def search(
        self, query_embedding: np.ndarray, top_k: int = 20, threshold: float = 0.0
    ) -> list[tuple[int, float]]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding: L2-normalized query embedding
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (chunk_id, cosine_similarity) tuples, sorted by similarity descending
        """
        await self._refresh_cache_if_needed()

        if self._cached_embeddings is None or self._cached_embeddings.size == 0:
            logger.warning("No embeddings found in database")
            return []

        try:
            # Ensure query embedding is L2-normalized
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                logger.warning("Query embedding has zero norm")
                return []

            normalized_query = query_embedding / query_norm  # unit length

            # Since stored embeddings are already L2-normalized,
            # cosine similarity = dot product
            if self._cached_embeddings is None:
                return []
            similarities = np.dot(self._cached_embeddings, normalized_query)

            # Filter by threshold
            valid_indices = np.where(similarities >= threshold)[0]

            if len(valid_indices) == 0:
                return []

            # Get top-k indices sorted by similarity (descending)
            sorted_indices = valid_indices[
                np.argsort(similarities[valid_indices])[::-1]
            ]
            top_indices = sorted_indices[:top_k]

            # Return chunk IDs and similarities
            if self._cached_chunk_ids is None:
                return []
            return [
                (self._cached_chunk_ids[idx], float(similarities[idx]))
                for idx in top_indices
            ]

        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    async def search_with_scores_dict(
        self, query_embedding: np.ndarray, top_k: int = 20, threshold: float = 0.0
    ) -> dict[int, float]:
        """
        Search and return results as a dictionary mapping chunk_id to similarity score.

        Args:
            query_embedding: L2-normalized query embedding
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            Dictionary mapping chunk_id to cosine similarity score
        """
        results = await self.search(query_embedding, top_k, threshold)
        return dict(results)

    async def get_similarity(self, chunk_id: int, query_embedding: np.ndarray) -> float:
        """
        Get similarity score between a specific chunk and query.

        Args:
            chunk_id: ID of the chunk
            query_embedding: L2-normalized query embedding

        Returns:
            Cosine similarity score, or 0.0 if chunk not found
        """
        chunk_embedding = await self.db_manager.get_embedding(chunk_id)

        if chunk_embedding is None:
            return 0.0

        try:
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return 0.0

            normalized_query = query_embedding / query_norm

            # Cosine similarity (chunk embedding is already L2-normalized)
            return float(np.dot(chunk_embedding, normalized_query))

        except Exception as e:
            logger.error(f"Error calculating similarity for chunk {chunk_id}: {e}")
            return 0.0

    def clear_cache(self):
        """Clear the embeddings cache to force reload."""
        self._cached_embeddings = None
        self._cached_chunk_ids = None
        logger.info("Embeddings cache cleared")

    async def get_cache_info(self) -> dict[str, Any]:
        """Get information about the current cache state."""
        await self._refresh_cache_if_needed()

        return {
            "cached_chunks": len(self._cached_chunk_ids)
            if self._cached_chunk_ids
            else 0,
            "embedding_dimension": self._cached_embeddings.shape[1]
            if self._cached_embeddings is not None and self._cached_embeddings.size > 0
            else 0,
            "cache_loaded": self._cached_embeddings is not None,
        }
