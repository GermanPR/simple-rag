"""Tests for configuration management."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.core.config import Config


class TestConfig:
    """Test configuration management."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()

        assert config.CHUNK_SIZE == 1800
        assert config.CHUNK_OVERLAP == 200
        assert config.DEFAULT_TOP_K == 8
        assert config.DEFAULT_THRESHOLD == 0.4
        assert config.MISTRAL_EMBED_MODEL == "mistral-embed"

    def test_environment_variables(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CHUNK_SIZE": "2000",
                "DEFAULT_TOP_K": "10",
                "MISTRAL_API_KEY": "test_key",
            },
        ):
            config = Config()
            assert config.CHUNK_SIZE == 2000
            assert config.DEFAULT_TOP_K == 10
            assert config.MISTRAL_API_KEY == "test_key"

    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        with pytest.raises(ValidationError):
            Config(
                MISTRAL_API_KEY="test",
                MISTRAL_EMBED_MODEL="mistral-embed",
                MISTRAL_CHAT_MODEL="mistral-large-latest",
                CHUNK_SIZE=100,
                CHUNK_OVERLAP=150,
            )

    def test_validate_mistral_config(self):
        """Test Mistral configuration validation."""
        config = Config(
            MISTRAL_API_KEY="valid_key",
            MISTRAL_EMBED_MODEL="mistral-embed",
            MISTRAL_CHAT_MODEL="mistral-large-latest",
        )
        assert config.validate_mistral_config() is True

        config = Config(
            MISTRAL_API_KEY="",
            MISTRAL_EMBED_MODEL="mistral-embed",
            MISTRAL_CHAT_MODEL="mistral-large-latest",
        )
        assert config.validate_mistral_config() is False

    def test_field_constraints(self):
        """Test field constraint validation."""
        # Test invalid chunk size
        with pytest.raises(ValidationError):
            Config(
                MISTRAL_API_KEY="test",
                MISTRAL_EMBED_MODEL="mistral-embed",
                MISTRAL_CHAT_MODEL="mistral-large-latest",
                CHUNK_SIZE=50,
            )  # Too small

        with pytest.raises(ValidationError):
            Config(
                MISTRAL_API_KEY="test",
                MISTRAL_EMBED_MODEL="mistral-embed",
                MISTRAL_CHAT_MODEL="mistral-large-latest",
                CHUNK_SIZE=10000,
            )  # Too large

        # Test invalid alpha value
        with pytest.raises(ValidationError):
            Config(
                MISTRAL_API_KEY="test",
                MISTRAL_EMBED_MODEL="mistral-embed",
                MISTRAL_CHAT_MODEL="mistral-large-latest",
                DEFAULT_ALPHA=1.5,
            )  # > 1.0

    @pytest.mark.parametrize(
        ("env_name", "expected"),
        [
            ("development", False),
            ("production", True),
            ("test", False),
        ],
    )
    def test_is_production(self, env_name, expected):
        """Test production environment detection."""
        with patch.dict(os.environ, {"ENVIRONMENT": env_name}):
            config = Config()
            assert config.is_production == expected
