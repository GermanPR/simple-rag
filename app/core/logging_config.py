"""Centralized logging configuration for the RAG system."""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Any


def get_logging_config(
    log_level: str = "INFO", log_dir: str = "logs"
) -> dict[str, Any]:
    """
    Get logging configuration dictionary.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files

    Returns:
        Dictionary configuration for logging
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "detailed",
                "filename": f"{log_dir}/rag_system.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": f"{log_dir}/rag_errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "app": {
                "handlers": ["console", "file", "error_file"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "streamlit": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {"level": log_level, "handlers": ["console", "file", "error_file"]},
    }


def setup_logging(
    log_level: str | None = None, log_dir: str = "logs", config_file: str | None = None
) -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level override
        log_dir: Directory for log files
        config_file: Path to custom logging config file
    """
    if config_file and Path(config_file).exists():
        logging.config.fileConfig(config_file)
    else:
        # Use environment variable or default
        if log_level is None:
            log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        config = get_logging_config(log_level, log_dir)
        logging.config.dictConfig(config)

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration."""
    return logging.getLogger(f"app.{name}")
