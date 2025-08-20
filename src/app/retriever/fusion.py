"""Hybrid search fusion and MMR diversification."""

import asyncio
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ScoreNormalizer:
    """Normalizes scores to [0, 1] range using min-max normalization."""

    @staticmethod
    def min_max_normalize(scores: list[float]) -> list[float]:
        """
        Apply min-max normalization to scores.

        Args:
            scores: List of scores to normalize

        Returns:
            List of normalized scores in [0, 1] range
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # Handle case where all scores are the same
        if max_score == min_score:
            return [0.5] * len(scores)

        return [(score - min_score) / (max_score - min_score) for score in scores]

    @staticmethod
    def normalize_score_dict(score_dict: dict[int, float]) -> dict[int, float]:
        """
        Normalize scores in a dictionary.

        Args:
            score_dict: Dictionary mapping chunk_id to score

        Returns:
            Dictionary with normalized scores
        """
        if not score_dict:
            return {}

        scores = list(score_dict.values())
        chunk_ids = list(score_dict.keys())

        normalized_scores = ScoreNormalizer.min_max_normalize(scores)

        return dict(zip(chunk_ids, normalized_scores, strict=True))


class HybridFusion:
    """Combines keyword and semantic search results using weighted fusion."""

    def __init__(self, alpha: float = 0.65):
        """
        Initialize hybrid fusion.

        Args:
            alpha: Weight for semantic scores (0-1).
                  Final score = alpha * semantic + (1-alpha) * keyword
        """
        self.alpha = alpha
        self.normalizer = ScoreNormalizer()

    def fuse_results(
        self, semantic_results: dict[int, float], keyword_results: dict[int, float]
    ) -> list[tuple[int, float]]:
        """
        Fuse semantic and keyword search results.

        Args:
            semantic_results: Dictionary mapping chunk_id to semantic similarity score
            keyword_results: Dictionary mapping chunk_id to TF-IDF score

        Returns:
            List of (chunk_id, fused_score) tuples, sorted by fused score descending
        """
        # Normalize scores to [0, 1] range
        normalized_semantic = self.normalizer.normalize_score_dict(semantic_results)
        normalized_keyword = self.normalizer.normalize_score_dict(keyword_results)

        # Get all chunk IDs from both result sets
        all_chunk_ids = set(normalized_semantic.keys()) | set(normalized_keyword.keys())

        fused_scores = {}

        for chunk_id in all_chunk_ids:
            semantic_score = normalized_semantic.get(chunk_id, 0.0)
            keyword_score = normalized_keyword.get(chunk_id, 0.0)

            # Weighted combination
            fused_score = self.alpha * semantic_score + (1 - self.alpha) * keyword_score
            fused_scores[chunk_id] = fused_score

        # Sort by fused score descending
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def set_alpha(self, alpha: float):
        """Update the alpha parameter for fusion weighting."""
        self.alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]


class MMRDiversifier:
    """Implements Maximal Marginal Relevance (MMR) for result diversification."""

    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR diversifier.

        Args:
            lambda_param: Balance between relevance and diversity (0-1).
                         Score = lambda * relevance - (1-lambda) * max_similarity_to_selected
        """
        self.lambda_param = lambda_param

    def diversify(
        self,
        ranked_results: list[tuple[int, float]],
        chunk_embeddings: dict[int, np.ndarray],
        query_embedding: np.ndarray,
        max_results: int,
    ) -> list[tuple[int, float]]:
        """
        Apply MMR diversification to search results.

        Args:
            ranked_results: List of (chunk_id, relevance_score) tuples, sorted by relevance
            chunk_embeddings: Dictionary mapping chunk_id to embedding vector
            query_embedding: Query embedding vector (L2-normalized)
            max_results: Maximum number of results to return

        Returns:
            List of (chunk_id, mmr_score) tuples with diversified selection
        """
        if not ranked_results or max_results <= 0:
            return []

        if max_results >= len(ranked_results):
            return ranked_results[:max_results]

        # Ensure query embedding is normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            # do not rerank, shouldnt happen since its already checked before
            return ranked_results[:max_results]

        normalized_query = query_embedding / query_norm

        selected = []
        remaining = list(ranked_results)

        # Select first result (highest relevance)
        if remaining:
            selected.append(remaining.pop(0))

        # Iteratively select remaining results using MMR
        while len(selected) < max_results and remaining:
            best_chunk = None
            best_score = float("-inf")
            best_idx = -1

            for i, (chunk_id, _relevance_score) in enumerate(remaining):
                if chunk_id not in chunk_embeddings:
                    continue

                # Calculate relevance component (cosine similarity with query)
                chunk_emb = chunk_embeddings[chunk_id]
                relevance = float(np.dot(chunk_emb, normalized_query))

                # Calculate redundancy component (max similarity to already selected)
                max_similarity = 0.0

                for selected_chunk_id, _ in selected:
                    if selected_chunk_id in chunk_embeddings:
                        selected_emb = chunk_embeddings[selected_chunk_id]
                        similarity = float(np.dot(chunk_emb, selected_emb))
                        max_similarity = max(max_similarity, similarity)

                # MMR score
                mmr_score = (
                    self.lambda_param * relevance
                    - (1 - self.lambda_param) * max_similarity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_chunk = (chunk_id, mmr_score)
                    best_idx = i

            if best_chunk is not None:
                selected.append(best_chunk)
                remaining.pop(best_idx)
            else:
                # Fallback: select by original relevance
                selected.append(remaining.pop(0))

        return selected

    def set_lambda(self, lambda_param: float):
        """Update the lambda parameter for MMR weighting."""
        self.lambda_param = max(0.0, min(1.0, lambda_param))  # Clamp to [0, 1]


class HybridRetriever:
    """Main class combining keyword, semantic, fusion, and MMR components."""

    def __init__(
        self,
        db_manager,
        keyword_searcher,
        semantic_searcher,
        alpha: float = 0.65,
        lambda_param: float = 0.7,
    ):
        self.db_manager = db_manager
        self.keyword_searcher = keyword_searcher
        self.semantic_searcher = semantic_searcher

        self.fusion = HybridFusion(alpha)
        self.mmr = MMRDiversifier(lambda_param)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 8,
        rerank_k: int = 20,
        threshold: float = 0.18,
        use_mmr: bool = True,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """
        Perform hybrid retrieval with optional MMR diversification.

        Args:
            query: Query text
            query_embedding: Query embedding vector
            top_k: Final number of results to return
            rerank_k: Number of candidates to consider for reranking
            threshold: Minimum semantic similarity threshold
            use_mmr: Whether to apply MMR diversification

        Returns:
            List of (chunk_id, final_score, chunk_info) tuples
        """
        # Get keyword search results
        keyword_results = self.keyword_searcher.search(query, top_k=rerank_k)
        keyword_dict = dict(keyword_results)

        # Get semantic search results
        semantic_results = self.semantic_searcher.search(
            query_embedding, top_k=rerank_k, threshold=threshold
        )
        semantic_dict = dict(semantic_results)

        # Check semantic threshold - if top result is below threshold, return empty
        if semantic_results and semantic_results[0][1] < threshold:
            logger.info(
                f"Top semantic score {semantic_results[0][1]:.3f} below threshold {threshold}"
            )
            return []

        # Fuse the results
        fused_results = self.fusion.fuse_results(semantic_dict, keyword_dict)

        if not fused_results:
            return []

        # Limit to rerank_k candidates
        candidates = fused_results[:rerank_k]

        # Apply MMR diversification if requested
        if use_mmr and len(candidates) > top_k:
            # Get embeddings for MMR calculation
            chunk_ids = [chunk_id for chunk_id, _ in candidates]
            chunk_embeddings = {}

            for chunk_id in chunk_ids:
                embedding = self.db_manager.get_embedding(chunk_id)
                if embedding is not None:
                    chunk_embeddings[chunk_id] = embedding

            # Apply MMR
            mmr_results = self.mmr.diversify(
                candidates, chunk_embeddings, query_embedding, top_k
            )
            final_results = mmr_results
        else:
            final_results = candidates[:top_k]

        # Get chunk information
        chunk_ids = [chunk_id for chunk_id, _ in final_results]
        chunks_info = self.db_manager.get_chunks_info(chunk_ids)
        chunks_dict = {info["chunk_id"]: info for info in chunks_info}

        # Combine results with chunk info
        final_with_info = []
        for chunk_id, score in final_results:
            chunk_info = chunks_dict.get(chunk_id, {})
            final_with_info.append((chunk_id, score, chunk_info))

        return final_with_info

    def update_parameters(
        self, alpha: float | None = None, lambda_param: float | None = None
    ):
        """Update fusion and MMR parameters."""
        if alpha is not None:
            self.fusion.set_alpha(alpha)
        if lambda_param is not None:
            self.mmr.set_lambda(lambda_param)


class AsyncHybridRetriever:
    """Async version of HybridRetriever combining keyword, semantic, fusion, and MMR components."""

    def __init__(
        self,
        async_db_manager,
        async_keyword_searcher,
        async_semantic_searcher,
        alpha: float = 0.65,
        lambda_param: float = 0.7,
    ):
        self.db_manager = async_db_manager
        self.keyword_searcher = async_keyword_searcher
        self.semantic_searcher = async_semantic_searcher

        self.fusion = HybridFusion(alpha)
        self.mmr = MMRDiversifier(lambda_param)

    async def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 8,
        rerank_k: int = 20,
        threshold: float = 0.18,
        use_mmr: bool = True,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """
        Perform hybrid retrieval with optional MMR diversification.

        Args:
            query: Query text
            query_embedding: Query embedding vector
            top_k: Final number of results to return
            rerank_k: Number of candidates to consider for reranking
            threshold: Minimum semantic similarity threshold
            use_mmr: Whether to apply MMR diversification

        Returns:
            List of (chunk_id, final_score, chunk_info) tuples
        """
        # Run keyword and semantic search in parallel
        keyword_task = asyncio.create_task(
            self.keyword_searcher.search(query, top_k=rerank_k)
        )
        semantic_task = asyncio.create_task(
            self.semantic_searcher.search(
                query_embedding, top_k=rerank_k, threshold=threshold
            )
        )

        keyword_results, semantic_results = await asyncio.gather(
            keyword_task, semantic_task
        )

        keyword_dict = dict(keyword_results)
        semantic_dict = dict(semantic_results)

        # Check semantic threshold - if top result is below threshold, return empty
        if semantic_results and semantic_results[0][1] < threshold:
            logger.info(
                f"Top semantic score {semantic_results[0][1]:.3f} below threshold {threshold}"
            )
            return []

        # Fuse the results
        fused_results = self.fusion.fuse_results(semantic_dict, keyword_dict)

        if not fused_results:
            return []

        # Limit to rerank_k candidates
        candidates = fused_results[:rerank_k]

        # Apply MMR diversification if requested
        if use_mmr and len(candidates) > top_k:
            # Get embeddings for MMR calculation
            chunk_ids = [chunk_id for chunk_id, _ in candidates]
            chunk_embeddings = {}

            # Get embeddings in parallel
            embedding_tasks = [
                self.db_manager.get_embedding(chunk_id) for chunk_id in chunk_ids
            ]
            embeddings = await asyncio.gather(*embedding_tasks)

            for chunk_id, embedding in zip(chunk_ids, embeddings, strict=False):
                if embedding is not None:
                    chunk_embeddings[chunk_id] = embedding

            # Apply MMR
            mmr_results = self.mmr.diversify(
                candidates, chunk_embeddings, query_embedding, top_k
            )
            final_results = mmr_results
        else:
            final_results = candidates[:top_k]

        # Get chunk information
        chunk_ids = [chunk_id for chunk_id, _ in final_results]
        chunks_info = await self.db_manager.get_chunks_info(chunk_ids)
        chunks_dict = {info["chunk_id"]: info for info in chunks_info}

        # Combine results with chunk info
        final_with_info = []
        for chunk_id, score in final_results:
            chunk_info = chunks_dict.get(chunk_id, {})
            final_with_info.append((chunk_id, score, chunk_info))

        return final_with_info

    def update_parameters(
        self, alpha: float | None = None, lambda_param: float | None = None
    ):
        """Update fusion and MMR parameters."""
        if alpha is not None:
            self.fusion.set_alpha(alpha)
        if lambda_param is not None:
            self.mmr.set_lambda(lambda_param)
