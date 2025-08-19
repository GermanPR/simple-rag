"""Intent detection for query processing."""

import re
from typing import Dict, Any
from enum import Enum


class QueryIntent(Enum):
    """Supported query intent types."""
    SMALLTALK = "smalltalk"
    QA = "qa"
    STRUCTURED = "structured"  # Lists, tables, steps
    COMPARISON = "comparison"
    SUMMARY = "summary"


class IntentDetector:
    """Detects intent from user queries."""
    
    def __init__(self):
        # Smalltalk patterns
        self.smalltalk_patterns = [
            r'\b(?:hi|hello|hey|greetings?)\b',
            r'\b(?:thanks?|thank you|thx)\b',
            r'\b(?:bye|goodbye|see you|farewell)\b',
            r'\b(?:how are you|what\'?s up|how\'?s it going)\b',
            r'\b(?:nice|good|great|awesome|cool)\b(?:\s+(?:job|work|stuff))?$',
        ]
        
        # Structured query patterns (lists, steps, tables)
        self.structured_patterns = [
            r'\b(?:list|enumerate|itemize)\b',
            r'\b(?:steps?|procedures?|process|workflow)\b',
            r'\b(?:table|chart|comparison|matrix)\b',
            r'\b(?:bullet points?|numbered list)\b',
            r'\b(?:outline|structure|organize)\b',
            r'(?:^|\s)(?:1\.|•|\*|\-)\s',  # Starts with list marker
        ]
        
        # Comparison patterns
        self.comparison_patterns = [
            r'\b(?:compar[ei]|contrast|vs|versus|difference)\b',
            r'\b(?:similar|different|alike|unlike)\b',
            r'\b(?:better|worse|best|worst|prefer)\b',
            r'\b(?:between|among)\b.*\band\b',
        ]
        
        # Summary patterns
        self.summary_patterns = [
            r'\b(?:summar[yi]ze?|overview|brief)\b',
            r'\b(?:main points?|key points?|highlights?)\b',
            r'\b(?:explain|describe|tell me about)\b',
            r'\bwhat is\b',
        ]
    
    def detect_intent(self, query: str) -> QueryIntent:
        """
        Detect the intent of a user query.
        
        Args:
            query: User query text
            
        Returns:
            Detected QueryIntent
        """
        query_lower = query.lower().strip()
        
        # Check smalltalk first (highest priority for short queries)
        if len(query_lower.split()) <= 3:  # Short queries
            for pattern in self.smalltalk_patterns:
                if re.search(pattern, query_lower):
                    return QueryIntent.SMALLTALK
        
        # Check structured patterns
        for pattern in self.structured_patterns:
            if re.search(pattern, query_lower):
                return QueryIntent.STRUCTURED
        
        # Check comparison patterns
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                return QueryIntent.COMPARISON
        
        # Check summary patterns
        for pattern in self.summary_patterns:
            if re.search(pattern, query_lower):
                return QueryIntent.SUMMARY
        
        # Default to QA
        return QueryIntent.QA
    
    def get_response_template(self, intent: QueryIntent) -> str:
        """
        Get the appropriate response template for an intent.
        
        Args:
            intent: Detected query intent
            
        Returns:
            Template string for formatting responses
        """
        templates = {
            QueryIntent.SMALLTALK: "I'm here to help answer questions about your documents. Please ask me something about the content you've uploaded.",
            QueryIntent.QA: "Answer the question based only on the provided context. If you cannot answer from the context, say 'Insufficient evidence to answer this question.'",
            QueryIntent.STRUCTURED: "Provide the answer in a structured format using bullet points or numbered lists. Base your answer only on the provided context.",
            QueryIntent.COMPARISON: "Compare the items mentioned in the query based only on the provided context. Structure your comparison clearly.",
            QueryIntent.SUMMARY: "Provide a concise summary based only on the provided context. Focus on the key points."
        }
        
        return templates.get(intent, templates[QueryIntent.QA])


class QueryRewriter:
    """Handles query rewriting and normalization."""
    
    def __init__(self):
        # Common acronym expansions (can be extended based on domain)
        self.acronym_expansions = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'nlp': 'natural language processing',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize and clean the query text.
        
        Args:
            query: Raw query text
            
        Returns:
            Normalized query text
        """
        if not query:
            return ""
        
        # Basic normalization
        normalized = query.strip()
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove extra punctuation at the end
        normalized = re.sub(r'[.!?]+$', '', normalized)
        
        # Expand common acronyms
        words = normalized.lower().split()
        expanded_words = []
        
        for word in words:
            # Remove punctuation for acronym matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.acronym_expansions:
                expanded_words.append(self.acronym_expansions[clean_word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words).strip()
    
    def rewrite_query(self, query: str, intent: QueryIntent) -> str:
        """
        Rewrite query based on detected intent.
        
        Args:
            query: Original query
            intent: Detected intent
            
        Returns:
            Rewritten query optimized for the intent
        """
        normalized = self.normalize_query(query)
        
        # Intent-specific rewriting
        if intent == QueryIntent.STRUCTURED:
            # Add explicit structure request if not present
            if not any(word in normalized.lower() for word in ['list', 'steps', 'enumerate']):
                normalized = f"list the {normalized}"
        
        elif intent == QueryIntent.COMPARISON:
            # Ensure comparison keywords are present
            if 'compare' not in normalized.lower() and 'vs' not in normalized.lower():
                normalized = f"compare {normalized}"
        
        elif intent == QueryIntent.SUMMARY:
            # Add summary request if not explicit
            if not any(word in normalized.lower() for word in ['summary', 'summarize', 'overview']):
                normalized = f"provide a summary of {normalized}"
        
        return normalized