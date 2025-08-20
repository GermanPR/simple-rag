"""TF-IDF based keyword search implementation."""

import math
import re
from collections import Counter


class TokenizerTFIDF:
    """Tokenizer and TF-IDF calculator for keyword search."""

    def __init__(self):
        # Regex pattern for tokens: alphanumeric and underscores
        self.token_pattern = re.compile(r"[A-Za-z0-9_]+")
        self.stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
            "d",
            "ll",
            "m",
            "o",
            "re",
            "ve",
            "y",
            "ain",
            "aren",
            "couldn",
            "didn",
            "doesn",
            "hadn",
            "hasn",
            "haven",
            "isn",
            "ma",
            "mightn",
            "mustn",
            "needn",
            "shan",
            "shouldn",
            "wasn",
            "weren",
            "won",
            "wouldn",
            "this",
            "is",
            "be",
            "are",
            "was",
            "were",
            "been",
            "being",
            "have",
            "has",
            "had",
        }

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into lowercase alphanumeric tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase tokens, excluding stopwords
        """
        if not text:
            return []

        tokens = self.token_pattern.findall(text.lower())
        # Filter out stopwords and very short tokens
        min_token_length = 2
        return [
            token
            for token in tokens
            if len(token) >= min_token_length and token not in self.stopwords
        ]

    def calculate_tf(self, tokens: list[str]) -> dict[str, float]:
        """
        Calculate term frequency using tf = 1 + log(count) formula.

        Args:
            tokens: List of tokens from a document/chunk

        Returns:
            Dictionary mapping token to TF score
        """
        if not tokens:
            return {}

        token_counts = Counter(tokens)
        tf_scores = {}

        for token, count in token_counts.items():
            tf_scores[token] = 1.0 + math.log(count)

        return tf_scores

    def calculate_idf(self, token: str, df: int, total_docs: int) -> float:
        """
        Calculate inverse document frequency using idf = log(N / (1 + df)).

        Args:
            token: The token to calculate IDF for
            df: Document frequency (number of documents containing the token)
            total_docs: Total number of documents

        Returns:
            IDF score for the token
        """
        if total_docs == 0:
            return 0.0

        return math.log(total_docs / (1.0 + df))

    def calculate_tfidf_scores(
        self,
        query_tokens: list[str],
        chunk_tokens: dict[str, float],
        token_stats: dict[str, tuple[int, int]],
    ) -> float:
        """
        Calculate TF-IDF score for a chunk given a query.

        Args:
            query_tokens: Tokens from the query
            chunk_tokens: TF scores for tokens in the chunk
            token_stats: Dictionary mapping token to (df, total_docs) tuples

        Returns:
            Total TF-IDF score for the chunk
        """
        if not query_tokens or not chunk_tokens:
            return 0.0

        total_score = 0.0

        for query_token in query_tokens:
            if query_token in chunk_tokens:
                tf_score = chunk_tokens[query_token]

                if query_token in token_stats:
                    df, total_docs = token_stats[query_token]
                    idf_score = self.calculate_idf(query_token, df, total_docs)
                    total_score += tf_score * idf_score

        return total_score


class KeywordSearcher:
    """Handles keyword-based search using TF-IDF scoring."""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.tokenizer = TokenizerTFIDF()

    def index_chunk(self, chunk_id: int, text: str):
        """
        Index a chunk's text for keyword search.

        Args:
            chunk_id: ID of the chunk
            text: Text content of the chunk
        """
        tokens = self.tokenizer.tokenize(text)
        tf_scores = self.tokenizer.calculate_tf(tokens)

        if tf_scores:
            self.db_manager.store_tokens(chunk_id, tf_scores)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Search for chunks using keyword matching with TF-IDF scoring.

        Args:
            query: Search query text
            top_k: Maximum number of results to return

        Returns:
            List of (chunk_id, tfidf_score) tuples, sorted by score descending
        """
        query_tokens = self.tokenizer.tokenize(query)

        if not query_tokens:
            return []

        # Get chunks that contain any of the query tokens
        candidate_chunks = self.db_manager.search_chunks_by_tokens(
            query_tokens, limit=top_k * 2
        )

        if not candidate_chunks:
            return []

        # Get token statistics for IDF calculation
        token_stats = {}
        for token in query_tokens:
            df, total_docs = self.db_manager.get_token_stats(token)
            token_stats[token] = (df, total_docs)

        # Re-score using proper TF-IDF calculation
        scored_chunks = []
        for chunk_id, _initial_score in candidate_chunks:
            chunk_tokens = self.db_manager.get_chunk_tokens(chunk_id)

            tfidf_score = self.tokenizer.calculate_tfidf_scores(
                query_tokens, chunk_tokens, token_stats
            )

            # Handle edge case where TF-IDF might be negative (single document)
            # In this case, use absolute value or fall back to TF-only scoring
            if tfidf_score != 0:
                final_score = abs(tfidf_score) if tfidf_score < 0 else tfidf_score
                scored_chunks.append((chunk_id, final_score))

        # Sort by score descending and limit results
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def get_query_tokens(self, query: str) -> list[str]:
        """Get tokenized query for use in other components."""
        return self.tokenizer.tokenize(query)


class AsyncKeywordSearcher:
    """Async version of KeywordSearcher."""

    def __init__(self, async_db_manager):
        self.db_manager = async_db_manager
        self.tokenizer = TokenizerTFIDF()

    async def index_chunk(self, chunk_id: int, text: str):
        """
        Index a chunk's text for keyword search.

        Args:
            chunk_id: ID of the chunk
            text: Text content of the chunk
        """
        tokens = self.tokenizer.tokenize(text)
        tf_scores = self.tokenizer.calculate_tf(tokens)

        if tf_scores:
            await self.db_manager.store_tokens(chunk_id, tf_scores)

    async def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Search for chunks using keyword matching with TF-IDF scoring.

        Args:
            query: Search query text
            top_k: Maximum number of results to return

        Returns:
            List of (chunk_id, tfidf_score) tuples, sorted by score descending
        """
        query_tokens = self.tokenizer.tokenize(query)

        if not query_tokens:
            return []

        # Get chunks that contain any of the query tokens
        candidate_chunks = await self.db_manager.search_chunks_by_tokens(
            query_tokens, limit=top_k * 2
        )

        if not candidate_chunks:
            return []

        # Get token statistics for IDF calculation
        token_stats = {}
        for token in query_tokens:
            df, total_docs = await self.db_manager.get_token_stats(token)
            token_stats[token] = (df, total_docs)

        # Re-score using proper TF-IDF calculation
        scored_chunks = []
        for chunk_id, _initial_score in candidate_chunks:
            chunk_tokens = await self.db_manager.get_chunk_tokens(chunk_id)

            tfidf_score = self.tokenizer.calculate_tfidf_scores(
                query_tokens, chunk_tokens, token_stats
            )

            # Handle edge case where TF-IDF might be negative (single document)
            # In this case, use absolute value or fall back to TF-only scoring
            if tfidf_score != 0:
                final_score = abs(tfidf_score) if tfidf_score < 0 else tfidf_score
                scored_chunks.append((chunk_id, final_score))

        # Sort by score descending and limit results
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def get_query_tokens(self, query: str) -> list[str]:
        """Get tokenized query for use in other components."""
        return self.tokenizer.tokenize(query)
