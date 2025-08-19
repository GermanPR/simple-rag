"""Tests for semantic search functionality."""

import pytest
import numpy as np
from app.retriever.semantic import SemanticSearcher
from app.retriever.index import DatabaseManager
import tempfile
import os


class TestSemanticSearcher:
    """Test cases for SemanticSearcher class."""
    
    @pytest.fixture
    def temp_db_with_embeddings(self):
        """Create a temporary database with sample embeddings."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        db_manager = DatabaseManager(path)
        
        # Add sample documents and chunks
        doc_id = db_manager.add_document("test.pdf", 1)
        
        # Add chunks
        chunk1_id = db_manager.add_chunk(doc_id, 1, 0, "machine learning algorithms")
        chunk2_id = db_manager.add_chunk(doc_id, 1, 1, "artificial intelligence systems")
        chunk3_id = db_manager.add_chunk(doc_id, 1, 2, "cooking recipes and food")
        
        # Create sample embeddings (L2-normalized)
        # Similar embeddings for ML-related chunks
        embedding1 = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        
        embedding2 = np.array([0.75, 0.65, 0.1], dtype=np.float32)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Different embedding for cooking chunk
        embedding3 = np.array([0.1, 0.2, 0.975], dtype=np.float32)
        embedding3 = embedding3 / np.linalg.norm(embedding3)
        
        # Store embeddings
        db_manager.store_embedding(chunk1_id, embedding1)
        db_manager.store_embedding(chunk2_id, embedding2)
        db_manager.store_embedding(chunk3_id, embedding3)
        
        yield db_manager, (chunk1_id, chunk2_id, chunk3_id), (embedding1, embedding2, embedding3)
        
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    def test_semantic_search_basic(self, temp_db_with_embeddings):
        """Test basic semantic search functionality."""
        db_manager, chunk_ids, embeddings = temp_db_with_embeddings
        chunk1_id, chunk2_id, chunk3_id = chunk_ids
        embedding1, embedding2, embedding3 = embeddings
        
        searcher = SemanticSearcher(db_manager)
        
        # Query with embedding similar to embedding1
        query_embedding = np.array([0.85, 0.52, 0.05], dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        results = searcher.search(query_embedding, top_k=3, threshold=0.0)
        
        # Should return results in order of similarity
        assert len(results) == 3
        
        # Results should be tuples of (chunk_id, similarity_score)
        chunk_ids_result = [chunk_id for chunk_id, score in results]
        scores = [score for chunk_id, score in results]
        
        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True)
        
        # All scores should be between 0 and 1 (cosine similarity)
        assert all(0.0 <= score <= 1.0 for score in scores)
    
    def test_semantic_search_with_threshold(self, temp_db_with_embeddings):
        """Test semantic search with similarity threshold."""
        db_manager, chunk_ids, embeddings = temp_db_with_embeddings
        
        searcher = SemanticSearcher(db_manager)
        
        # Query similar to ML embeddings
        query_embedding = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # High threshold should filter out dissimilar results
        results = searcher.search(query_embedding, top_k=10, threshold=0.7)
        
        # Should only return very similar embeddings
        assert len(results) <= 3
        
        # All returned scores should be above threshold
        for chunk_id, score in results:
            assert score >= 0.7
    
    def test_semantic_search_empty_database(self):
        """Test semantic search with empty database."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        db_manager = DatabaseManager(path)
        searcher = SemanticSearcher(db_manager)
        
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = searcher.search(query_embedding, top_k=5)
        
        assert len(results) == 0
        
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    def test_semantic_search_zero_norm_query(self, temp_db_with_embeddings):
        """Test semantic search with zero-norm query embedding."""
        db_manager, chunk_ids, embeddings = temp_db_with_embeddings
        
        searcher = SemanticSearcher(db_manager)
        
        # Zero vector query
        query_embedding = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        results = searcher.search(query_embedding, top_k=5)
        
        # Should return empty results for zero-norm query
        assert len(results) == 0
    
    def test_get_similarity_single_chunk(self, temp_db_with_embeddings):
        """Test getting similarity for a single chunk."""
        db_manager, chunk_ids, embeddings = temp_db_with_embeddings
        chunk1_id, chunk2_id, chunk3_id = chunk_ids
        
        searcher = SemanticSearcher(db_manager)
        
        # Query embedding
        query_embedding = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Get similarity to specific chunk
        similarity = searcher.get_similarity(chunk1_id, query_embedding)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        
        # Should be high similarity since query is similar to chunk1's embedding
        assert similarity > 0.8
    
    def test_get_similarity_nonexistent_chunk(self, temp_db_with_embeddings):
        """Test getting similarity for non-existent chunk."""
        db_manager, chunk_ids, embeddings = temp_db_with_embeddings
        
        searcher = SemanticSearcher(db_manager)
        
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Non-existent chunk ID
        similarity = searcher.get_similarity(99999, query_embedding)
        assert similarity == 0.0
    
    def test_cache_functionality(self, temp_db_with_embeddings):
        """Test embedding cache loading and clearing."""
        db_manager, chunk_ids, embeddings = temp_db_with_embeddings
        
        searcher = SemanticSearcher(db_manager)
        
        # Get cache info - should trigger cache loading
        cache_info = searcher.get_cache_info()
        
        assert cache_info["cached_chunks"] == 3
        assert cache_info["embedding_dimension"] == 3
        assert cache_info["cache_loaded"] == True
        
        # Clear cache
        searcher.clear_cache()
        
        # Cache should be cleared but will reload on next operation
        assert searcher._cached_embeddings is None
        assert searcher._cached_chunk_ids is None
    
    def test_search_with_scores_dict(self, temp_db_with_embeddings):
        """Test search method that returns dictionary of scores."""
        db_manager, chunk_ids, embeddings = temp_db_with_embeddings
        
        searcher = SemanticSearcher(db_manager)
        
        query_embedding = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        scores_dict = searcher.search_with_scores_dict(
            query_embedding, top_k=3, threshold=0.0
        )
        
        # Should return dictionary mapping chunk_id to score
        assert isinstance(scores_dict, dict)
        assert len(scores_dict) == 3
        
        # All chunk IDs should be present
        chunk1_id, chunk2_id, chunk3_id = chunk_ids
        for chunk_id in [chunk1_id, chunk2_id, chunk3_id]:
            assert chunk_id in scores_dict
            assert isinstance(scores_dict[chunk_id], float)
            assert 0.0 <= scores_dict[chunk_id] <= 1.0


def test_cosine_similarity_calculation():
    """Test that cosine similarity is calculated correctly."""
    # Create two L2-normalized vectors
    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vec3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Same as vec1
    
    # Manual cosine similarity calculation
    # For L2-normalized vectors, cosine similarity = dot product
    sim_orthogonal = np.dot(vec1, vec2)  # Should be 0 (orthogonal)
    sim_identical = np.dot(vec1, vec3)   # Should be 1 (identical)
    
    assert abs(sim_orthogonal - 0.0) < 1e-6
    assert abs(sim_identical - 1.0) < 1e-6


def test_l2_normalization():
    """Test that embeddings are properly L2-normalized."""
    # Create a non-normalized vector
    vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    
    # Normalize it
    normalized = vec / np.linalg.norm(vec)
    
    # Check that norm is 1
    norm = np.linalg.norm(normalized)
    assert abs(norm - 1.0) < 1e-6
    
    # Check values
    expected = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    assert np.allclose(normalized, expected)