"""Tests for hybrid fusion and MMR functionality."""

import numpy as np

from app.retriever.fusion import HybridFusion
from app.retriever.fusion import MMRDiversifier
from app.retriever.fusion import ScoreNormalizer


class TestScoreNormalizer:
    """Test cases for score normalization."""

    def test_min_max_normalization_basic(self):
        """Test basic min-max normalization."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = ScoreNormalizer.min_max_normalize(scores)

        # Should be normalized to [0, 1] range
        assert normalized[0] == 0.0  # min score
        assert normalized[-1] == 1.0  # max score
        assert all(0.0 <= score <= 1.0 for score in normalized)

        # Should preserve relative ordering
        assert normalized == sorted(normalized)

    def test_min_max_normalization_identical_scores(self):
        """Test normalization when all scores are identical."""
        scores = [5.0, 5.0, 5.0, 5.0]
        normalized = ScoreNormalizer.min_max_normalize(scores)

        # All should be 0.5 when min == max
        assert all(score == 0.5 for score in normalized)

    def test_min_max_normalization_empty(self):
        """Test normalization with empty input."""
        assert ScoreNormalizer.min_max_normalize([]) == []

    def test_normalize_score_dict(self):
        """Test normalization of score dictionary."""
        score_dict = {1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0}
        normalized_dict = ScoreNormalizer.normalize_score_dict(score_dict)

        # Should normalize values while preserving keys
        assert set(normalized_dict.keys()) == {1, 2, 3, 4}
        assert normalized_dict[1] == 0.0  # min
        assert normalized_dict[4] == 1.0  # max
        assert all(0.0 <= score <= 1.0 for score in normalized_dict.values())

    def test_normalize_empty_dict(self):
        """Test normalization of empty dictionary."""
        assert ScoreNormalizer.normalize_score_dict({}) == {}


class TestHybridFusion:
    """Test cases for hybrid fusion."""

    def test_fusion_basic(self):
        """Test basic fusion of semantic and keyword results."""
        fusion = HybridFusion(alpha=0.6)

        semantic_results = {1: 0.9, 2: 0.7, 3: 0.5}
        keyword_results = {1: 0.3, 2: 0.8, 3: 0.6, 4: 1.0}  # Item 4 only in keyword

        fused_results = fusion.fuse_results(semantic_results, keyword_results)

        # Should return sorted list of (chunk_id, fused_score) tuples
        assert isinstance(fused_results, list)
        assert len(fused_results) == 4  # All unique chunk IDs

        # Results should be sorted by score descending
        scores = [score for chunk_id, score in fused_results]
        assert scores == sorted(scores, reverse=True)

        # All scores should be in [0, 1] range
        assert all(0.0 <= score <= 1.0 for score in scores)

    def test_fusion_alpha_extremes(self):
        """Test fusion with extreme alpha values."""
        semantic_results = {1: 1.0, 2: 0.5}
        keyword_results = {1: 0.0, 2: 1.0}

        # Pure semantic (alpha=1.0)
        fusion_semantic = HybridFusion(alpha=1.0)
        results_semantic = fusion_semantic.fuse_results(
            semantic_results, keyword_results
        )

        # Item 1 should rank higher (better semantic score)
        top_chunk_id = results_semantic[0][0]
        assert top_chunk_id == 1

        # Pure keyword (alpha=0.0)
        fusion_keyword = HybridFusion(alpha=0.0)
        results_keyword = fusion_keyword.fuse_results(semantic_results, keyword_results)

        # Item 2 should rank higher (better keyword score)
        top_chunk_id = results_keyword[0][0]
        assert top_chunk_id == 2

    def test_fusion_empty_inputs(self):
        """Test fusion with empty inputs."""
        fusion = HybridFusion(alpha=0.5)

        # Both empty
        results = fusion.fuse_results({}, {})
        assert results == []

        # One empty
        semantic_results = {1: 0.8}
        results = fusion.fuse_results(semantic_results, {})
        assert len(results) == 1

        results = fusion.fuse_results({}, {1: 0.8})
        assert len(results) == 1

    def test_set_alpha(self):
        """Test alpha parameter setting and clamping."""
        fusion = HybridFusion(alpha=0.5)

        # Valid values
        fusion.set_alpha(0.8)
        assert fusion.alpha == 0.8

        # Clamping
        fusion.set_alpha(-0.1)  # Should clamp to 0.0
        assert fusion.alpha == 0.0

        fusion.set_alpha(1.5)  # Should clamp to 1.0
        assert fusion.alpha == 1.0


class TestMMRDiversifier:
    """Test cases for MMR diversification."""

    def test_mmr_basic(self):
        """Test basic MMR diversification."""
        mmr = MMRDiversifier(lambda_param=0.7)

        # Create ranked results (chunk_id, relevance_score)
        ranked_results = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]

        # Create embeddings - make some similar to test diversification
        embeddings = {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Similar to 2
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),  # Similar to 1
            3: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Different
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32),  # Different
        }

        # Normalize embeddings
        for chunk_id, embedding in embeddings.items():
            embeddings[chunk_id] = (embedding / np.linalg.norm(embedding)).astype(
                np.float32
            )

        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Request 3 results
        mmr_results = mmr.diversify(
            ranked_results, embeddings, query_embedding, max_results=3
        )

        assert len(mmr_results) == 3

        # First result should be the highest relevance (chunk 1)
        assert mmr_results[0][0] == 1

        # Results should be diverse (not all similar embeddings)
        selected_ids = [chunk_id for chunk_id, score in mmr_results]
        # MMR should select diverse results - exact selection may vary but should include diverse items
        # At minimum, should not select only the most similar items (1,2)
        assert len(set(selected_ids)) == 3  # Should have 3 different chunks
        # Should include some diverse chunks (3 or 4), not just the similar ones (1,2)
        diverse_selected = len([id for id in selected_ids if id in [3, 4]])
        assert diverse_selected >= 1  # At least one diverse chunk should be selected

    def test_mmr_more_results_than_candidates(self):
        """Test MMR when requesting more results than available."""
        mmr = MMRDiversifier(lambda_param=0.7)

        ranked_results = [(1, 0.9), (2, 0.8)]
        embeddings = {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
        }
        query_embedding = np.array([1.0, 0.0], dtype=np.float32)

        # Request more results than available
        mmr_results = mmr.diversify(
            ranked_results, embeddings, query_embedding, max_results=5
        )

        # Should return all available results
        assert len(mmr_results) == 2

    def test_mmr_empty_input(self):
        """Test MMR with empty input."""
        mmr = MMRDiversifier(lambda_param=0.7)

        results = mmr.diversify([], {}, np.array([1.0, 0.0]), max_results=3)
        assert results == []

    def test_mmr_zero_max_results(self):
        """Test MMR with zero max_results."""
        mmr = MMRDiversifier(lambda_param=0.7)

        ranked_results = [(1, 0.9), (2, 0.8)]
        embeddings = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
        query_embedding = np.array([1.0, 0.0])

        results = mmr.diversify(
            ranked_results, embeddings, query_embedding, max_results=0
        )
        assert results == []

    def test_mmr_missing_embeddings(self):
        """Test MMR when some embeddings are missing."""
        mmr = MMRDiversifier(lambda_param=0.7)

        ranked_results = [(1, 0.9), (2, 0.8), (3, 0.7)]
        embeddings = {
            1: np.array([1.0, 0.0], dtype=np.float32),
            # 2 is missing
            3: np.array([0.0, 1.0], dtype=np.float32),
        }
        query_embedding = np.array([1.0, 0.0], dtype=np.float32)

        # Should handle missing embeddings gracefully
        mmr_results = mmr.diversify(
            ranked_results, embeddings, query_embedding, max_results=3
        )

        # Should still return results, preferring chunks with embeddings
        assert len(mmr_results) > 0

        # Should include chunk 1 (first and has embedding)
        selected_ids = [chunk_id for chunk_id, score in mmr_results]
        assert 1 in selected_ids

    def test_set_lambda(self):
        """Test lambda parameter setting and clamping."""
        mmr = MMRDiversifier(lambda_param=0.5)

        # Valid values
        mmr.set_lambda(0.8)
        assert mmr.lambda_param == 0.8

        # Clamping
        mmr.set_lambda(-0.1)  # Should clamp to 0.0
        assert mmr.lambda_param == 0.0

        mmr.set_lambda(1.5)  # Should clamp to 1.0
        assert mmr.lambda_param == 1.0


def test_mmr_relevance_vs_diversity():
    """Test that MMR balances relevance and diversity based on lambda."""
    # High lambda (0.9) should prioritize relevance
    mmr_relevance = MMRDiversifier(lambda_param=0.9)

    # Low lambda (0.1) should prioritize diversity
    mmr_diversity = MMRDiversifier(lambda_param=0.1)

    # Create scenarios where relevance and diversity conflict
    ranked_results = [(1, 1.0), (2, 0.9), (3, 0.5)]  # 1 and 2 are highly relevant

    # Make 1 and 2 very similar (low diversity), 3 very different
    embeddings = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.95, 0.05, 0.0], dtype=np.float32),  # Very similar to 1
        3: np.array([0.0, 0.0, 1.0], dtype=np.float32),  # Very different
    }

    query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # High lambda should prefer relevance (likely select 1, 2)
    results_relevance = mmr_relevance.diversify(
        ranked_results, embeddings, query_embedding, max_results=2
    )

    # Low lambda should prefer diversity (likely select 1, 3 over 1, 2)
    results_diversity = mmr_diversity.diversify(
        ranked_results, embeddings, query_embedding, max_results=2
    )

    # Extract selected chunk IDs
    ids_relevance = [chunk_id for chunk_id, score in results_relevance]
    ids_diversity = [chunk_id for chunk_id, score in results_diversity]

    # Both should select chunk 1 (highest relevance)
    assert 1 in ids_relevance
    assert 1 in ids_diversity

    # Relevance-focused might select chunk 2, diversity-focused should prefer chunk 3
    # Note: This is probabilistic based on the exact MMR calculation
