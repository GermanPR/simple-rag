"""Tests for TF-IDF implementation."""

import pytest
import math
from app.retriever.keyword import TokenizerTFIDF, KeywordSearcher
from app.retriever.index import DatabaseManager
import tempfile
import os


class TestTokenizerTFIDF:
    """Test cases for TF-IDF tokenizer and calculator."""
    
    def test_tokenization_basic(self):
        """Test basic tokenization functionality."""
        tokenizer = TokenizerTFIDF()
        
        text = "This is a simple test text with some words."
        tokens = tokenizer.tokenize(text)
        
        # Should be lowercase and exclude stopwords
        expected_tokens = ["simple", "test", "text", "words"]
        assert all(token in tokens for token in expected_tokens)
        assert "this" not in tokens  # stopword
        assert "is" not in tokens    # stopword
        assert "a" not in tokens     # stopword
    
    def test_tokenization_special_characters(self):
        """Test tokenization with special characters and numbers."""
        tokenizer = TokenizerTFIDF()
        
        text = "API_KEY123 test-case @#$% hello_world 42"
        tokens = tokenizer.tokenize(text)
        
        # Should include alphanumeric and underscores only (lowercase)
        assert "api_key123" in tokens  # Should be lowercase
        assert "hello_world" in tokens
        assert "42" in tokens
        assert "test" in tokens
        assert "case" in tokens
        
        # Special characters should be filtered out
        assert "@#$%" not in tokens
    
    def test_tokenization_empty_and_short(self):
        """Test tokenization of empty and very short strings."""
        tokenizer = TokenizerTFIDF()
        
        # Empty string
        assert tokenizer.tokenize("") == []
        
        # Only stopwords
        assert tokenizer.tokenize("the a an") == []
        
        # Very short tokens (less than 2 chars)
        assert tokenizer.tokenize("a b c") == []
        # "be" is not in the stopwords list, but "to" and "or" are
        result = tokenizer.tokenize("to or not")  # "not" is also stopword
        assert result == []  # Should be empty after filtering
    
    def test_tf_calculation(self):
        """Test term frequency calculation."""
        tokenizer = TokenizerTFIDF()
        
        tokens = ["test", "word", "test", "another", "word", "test"]
        tf_scores = tokenizer.calculate_tf(tokens)
        
        # tf = 1 + log(count)
        expected_tf_test = 1 + math.log(3)  # "test" appears 3 times
        expected_tf_word = 1 + math.log(2)  # "word" appears 2 times
        expected_tf_another = 1 + math.log(1)  # "another" appears 1 time
        
        assert abs(tf_scores["test"] - expected_tf_test) < 1e-6
        assert abs(tf_scores["word"] - expected_tf_word) < 1e-6
        assert abs(tf_scores["another"] - expected_tf_another) < 1e-6
    
    def test_tf_calculation_empty(self):
        """Test TF calculation with empty input."""
        tokenizer = TokenizerTFIDF()
        
        assert tokenizer.calculate_tf([]) == {}
        assert tokenizer.calculate_tf(None) == {}
    
    def test_idf_calculation(self):
        """Test inverse document frequency calculation."""
        tokenizer = TokenizerTFIDF()
        
        # idf = log(N / (1 + df))
        total_docs = 100
        df = 10
        
        expected_idf = math.log(total_docs / (1 + df))
        calculated_idf = tokenizer.calculate_idf("test", df, total_docs)
        
        assert abs(calculated_idf - expected_idf) < 1e-6
    
    def test_idf_edge_cases(self):
        """Test IDF calculation edge cases."""
        tokenizer = TokenizerTFIDF()
        
        # Zero total docs
        assert tokenizer.calculate_idf("test", 5, 0) == 0.0
        
        # Zero document frequency
        total_docs = 100
        df = 0
        expected_idf = math.log(total_docs / 1)  # 1 + 0
        calculated_idf = tokenizer.calculate_idf("test", df, total_docs)
        assert abs(calculated_idf - expected_idf) < 1e-6
    
    def test_tfidf_scoring(self):
        """Test complete TF-IDF scoring."""
        tokenizer = TokenizerTFIDF()
        
        query_tokens = ["test", "word", "missing"]
        chunk_tokens = {"test": 2.0, "word": 1.5, "other": 1.0}  # TF scores
        token_stats = {
            "test": (5, 100),    # df=5, total_docs=100
            "word": (10, 100),   # df=10, total_docs=100
            "missing": (1, 100), # df=1, total_docs=100
            "other": (2, 100)    # df=2, total_docs=100
        }
        
        score = tokenizer.calculate_tfidf_scores(query_tokens, chunk_tokens, token_stats)
        
        # Should only score for tokens present in both query and chunk
        expected_test_score = 2.0 * math.log(100 / 6)  # tf * idf for "test"
        expected_word_score = 1.5 * math.log(100 / 11) # tf * idf for "word"
        expected_total = expected_test_score + expected_word_score
        
        assert abs(score - expected_total) < 1e-6
    
    def test_tfidf_scoring_no_overlap(self):
        """Test TF-IDF scoring with no overlap between query and chunk."""
        tokenizer = TokenizerTFIDF()
        
        query_tokens = ["missing", "absent"]
        chunk_tokens = {"present": 2.0, "available": 1.5}
        token_stats = {}  # Empty stats
        
        score = tokenizer.calculate_tfidf_scores(query_tokens, chunk_tokens, token_stats)
        assert score == 0.0


class TestKeywordSearcher:
    """Test cases for KeywordSearcher class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        db_manager = DatabaseManager(path)
        yield db_manager
        
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    def test_index_and_search(self, temp_db):
        """Test indexing chunks and searching."""
        searcher = KeywordSearcher(temp_db)
        
        # Add some test documents and chunks
        doc_id = temp_db.add_document("test.pdf", 1)
        
        # Add chunks with different content
        chunk1_id = temp_db.add_chunk(doc_id, 1, 0, "machine learning algorithms are powerful")
        chunk2_id = temp_db.add_chunk(doc_id, 1, 1, "artificial intelligence and machine learning")
        chunk3_id = temp_db.add_chunk(doc_id, 1, 2, "deep learning neural networks")
        
        # Index the chunks
        searcher.index_chunk(chunk1_id, "machine learning algorithms are powerful")
        searcher.index_chunk(chunk2_id, "artificial intelligence and machine learning")
        searcher.index_chunk(chunk3_id, "deep learning neural networks")
        
        # Search for "machine learning"
        results = searcher.search("machine learning", top_k=10)
        
        # Should return chunks that contain these terms
        chunk_ids = [chunk_id for chunk_id, score in results]
        assert chunk1_id in chunk_ids
        assert chunk2_id in chunk_ids
        
        # Scores should be positive and in descending order
        scores = [score for chunk_id, score in results]
        assert all(score > 0 for score in scores)
        assert scores == sorted(scores, reverse=True)
    
    def test_search_no_results(self, temp_db):
        """Test searching with no matching results."""
        searcher = KeywordSearcher(temp_db)
        
        # Add a chunk that won't match the query
        doc_id = temp_db.add_document("test.pdf", 1)
        chunk_id = temp_db.add_chunk(doc_id, 1, 0, "completely different content")
        searcher.index_chunk(chunk_id, "completely different content")
        
        # Search for unrelated terms
        results = searcher.search("machine learning artificial intelligence", top_k=10)
        
        assert len(results) == 0
    
    def test_search_empty_query(self, temp_db):
        """Test searching with empty query."""
        searcher = KeywordSearcher(temp_db)
        
        results = searcher.search("", top_k=10)
        assert len(results) == 0
        
        results = searcher.search("   ", top_k=10)
        assert len(results) == 0
    
    def test_get_query_tokens(self, temp_db):
        """Test query tokenization method."""
        searcher = KeywordSearcher(temp_db)
        
        query = "Machine Learning and AI algorithms"
        tokens = searcher.get_query_tokens(query)
        
        expected_tokens = ["machine", "learning", "algorithms"]
        assert all(token in tokens for token in expected_tokens)
        assert "and" not in tokens  # stopword should be removed