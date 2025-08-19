"""Configuration settings for the RAG system."""

import os
from typing import Optional


class Config:
    """Configuration class for environment variables and settings."""
    
    # Mistral API settings
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_EMBED_MODEL: str = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    MISTRAL_CHAT_MODEL: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")
    MISTRAL_BASE_URL: str = "https://api.mistral.ai/v1"
    
    # Database settings
    DB_PATH: str = os.getenv("DB_PATH", "rag.sqlite3")
    
    # Backend URL for Streamlit (optional)
    BACKEND_URL: Optional[str] = os.getenv("BACKEND_URL") or None
    
    # Chunking settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval settings
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "8"))
    DEFAULT_RERANK_K: int = int(os.getenv("DEFAULT_RERANK_K", "20"))
    DEFAULT_THRESHOLD: float = float(os.getenv("DEFAULT_THRESHOLD", "0.18"))
    DEFAULT_ALPHA: float = float(os.getenv("DEFAULT_ALPHA", "0.65"))  # Semantic vs keyword blend
    DEFAULT_LAMBDA: float = float(os.getenv("DEFAULT_LAMBDA", "0.7"))  # MMR relevance vs diversity
    
    # Hallucination filter threshold
    SENTENCE_THRESHOLD: float = float(os.getenv("SENTENCE_THRESHOLD", "0.16"))
    
    @classmethod
    def validate_mistral_config(cls) -> bool:
        """Validate that required Mistral API configuration is present."""
        return bool(cls.MISTRAL_API_KEY and cls.MISTRAL_API_KEY != "your_mistral_api_key_here")


# Global config instance
config = Config()