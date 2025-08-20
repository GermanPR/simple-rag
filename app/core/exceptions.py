"""Custom exception classes for the RAG system."""


class RAGError(Exception):
    """Base exception for RAG system errors."""


class ConfigurationError(RAGError):
    """Raised when configuration is invalid."""


class DatabaseError(RAGError):
    """Raised when database operations fail."""


class EmbeddingError(RAGError):
    """Raised when embedding operations fail."""


class RetrievalError(RAGError):
    """Raised when retrieval operations fail."""


class GenerationError(RAGError):
    """Raised when text generation fails."""


class ValidationError(RAGError):
    """Raised when input validation fails."""


class ProcessingError(RAGError):
    """Raised when document processing fails."""


class APIError(RAGError):
    """Raised when external API calls fail."""
