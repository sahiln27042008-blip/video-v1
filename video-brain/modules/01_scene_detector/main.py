#!/usr/bin/env python3
"""Entry point for scene detection module."""

import sys

# Fix Windows console encoding issues
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.cli import main

if __name__ == "__main__":
    main()