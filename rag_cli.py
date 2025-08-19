#!/usr/bin/env python3
"""
Convenience script to run RAG CLI commands.
Usage: python rag_cli.py <command> [args...]
"""

import subprocess
import sys

if __name__ == "__main__":
    # Forward all arguments to the CLI module
    cmd = ["uv", "run", "python", "-m", "app.cli.commands"] + sys.argv[1:]
    subprocess.run(cmd, check=False)
