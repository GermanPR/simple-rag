#!/usr/bin/env python3
"""Convenience script for testing intent detection from CLI."""

import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.cli.commands import test_intent

if __name__ == "__main__":
    # Run the test-intent function directly with typer
    import typer

    typer.run(test_intent)
