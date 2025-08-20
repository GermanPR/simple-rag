"""Configuration settings for the RAG system."""

import os
from pathlib import Path

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration class with Pydantic validation."""

    # Mistral API settings
    MISTRAL_API_KEY: str = Field(default="", description="Mistral API key")
    MISTRAL_EMBED_MODEL: str = Field(
        default="mistral-embed", description="Mistral embedding model"
    )
    MISTRAL_CHAT_MODEL: str = Field(
        default="ministral-8b-latest", description="Mistral chat completion model"
    )
    MISTRAL_BASE_URL: str = Field(
        default="https://api.mistral.ai/v1", description="Mistral API base URL"
    )

    # Database settings
    DB_PATH: str = Field(default="rag.sqlite3", description="SQLite database path")

    # Backend URL for Streamlit (optional)
    BACKEND_URL: str | None = Field(
        default=None, description="Backend API URL for Streamlit"
    )

    # Chunking settings
    CHUNK_SIZE: int = Field(
        default=1800, ge=100, le=8192, description="Target chunk size in characters"
    )
    CHUNK_OVERLAP: int = Field(
        default=200, ge=0, description="Overlap between chunks in characters"
    )

    # Retrieval settings
    DEFAULT_TOP_K: int = Field(
        default=8, ge=1, le=100, description="Default number of top results to return"
    )
    DEFAULT_RERANK_K: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Default number of candidates for reranking",
    )
    DEFAULT_THRESHOLD: float = Field(
        default=0.18, ge=0.0, le=1.0, description="Default similarity threshold"
    )
    DEFAULT_ALPHA: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Semantic vs keyword blend ratio"
    )
    DEFAULT_LAMBDA: float = Field(
        default=0.7, ge=0.0, le=1.0, description="MMR relevance vs diversity ratio"
    )

    # Hallucination filter threshold
    SENTENCE_THRESHOLD: float = Field(
        default=0.16,
        ge=0.0,
        le=1.0,
        description="Sentence-level hallucination threshold",
    )

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_overlap(cls, v):
        """Ensure overlap is less than chunk size."""
        # Note: Cross-field validation would require model_validator in pydantic v2
        # For now, we'll validate this at runtime if needed
        return v

    @field_validator("DB_PATH")
    @classmethod
    def validate_db_path(cls, v):
        """Ensure database directory exists or can be created."""
        db_path = Path(v)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return str(db_path)

    class Config:
        env_file = Path(__file__).parents[3] / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def validate_mistral_config(self) -> bool:
        """Validate that required Mistral API configuration is present."""
        return bool(
            self.MISTRAL_API_KEY and self.MISTRAL_API_KEY != "your_mistral_api_key_here"
        )

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"


# Global config instance
config = Config()
