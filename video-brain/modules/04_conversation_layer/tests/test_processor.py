"""Tests for conversation processor."""

import pytest
from src.processor import ConversationProcessor

def test_processor_init():
    # Should not raise without HF token if we don't use diarization
    proc = ConversationProcessor(hf_token=None)
    assert proc.model_size == "base"