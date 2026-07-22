"""Tests for metrics calculator."""

import json
import tempfile
from pathlib import Path
import pytest
from src.metrics import MetricsCalculator
from src.models import MetricsResult

def test_metrics_calculation():
    # Create a minimal timeline_with_words.json for testing
    data = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker": "SPEAKER_00",
                "text": "Hello world.",
                "scene_id": 1,
                "words": [
                    {"word": "Hello", "start": 0.1, "end": 0.5},
                    {"word": "world.", "start": 0.6, "end": 1.0}
                ]
            },
            {
                "start": 3.0,
                "end": 5.0,
                "speaker": "SPEAKER_01",
                "text": "This is a test.",
                "scene_id": 1,
                "words": [
                    {"word": "This", "start": 3.1, "end": 3.5},
                    {"word": "is", "start": 3.6, "end": 3.9},
                    {"word": "a", "start": 4.0, "end": 4.2},
                    {"word": "test.", "start": 4.3, "end": 4.7}
                ]
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = Path(f.name)

    calc = MetricsCalculator(tmp_path)
    result = calc.compute_all()
    assert isinstance(result, MetricsResult)
    assert result.summary.num_scenes == 1
    assert result.speech.speakers_count == 2
    # Clean up
    tmp_path.unlink()