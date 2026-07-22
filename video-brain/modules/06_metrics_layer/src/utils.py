"""Utility functions for metrics layer."""

import re
from typing import List, Dict, Any

def extract_text_from_segments(segments: List[Dict[str, Any]]) -> str:
    """Extract concatenated text from timeline segments."""
    return " ".join([seg.get("text", "") for seg in segments])

def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(re.findall(r"\b\w+\b", text))

def unique_words(text: str) -> set:
    """Get set of unique words."""
    return set(w.lower() for w in re.findall(r"\b\w+\b", text))