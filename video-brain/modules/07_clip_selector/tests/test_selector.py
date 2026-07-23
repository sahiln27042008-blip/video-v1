"""Tests for clip selector."""

import json
import tempfile
from pathlib import Path
import pytest
from src.selector import ClipSelector
from src.config import ClipSelectorConfig
from src.models import CandidateClipsResult

def test_selector_smoke():
    # Create minimal timeline and metrics data
    timeline = {
        "segments": [
            {
                "segment_id": 1,
                "start": 0.0,
                "end": 3.0,
                "speaker_label": "SPEAKER_00",
                "speaker": "SPEAKER_00",
                "person": "person_1",
                "text": "Hello world.",
                "scene_id": 1,
            }
        ],
        "video": "video.mp4"
    }
    segment_metrics = [
        {
            "segment_id": 1,
            "duration": 3.0,
            "speaker": "SPEAKER_00",
            "person": "person_1",
            "scene": 1,
            "word_count": 5,
            "words_per_minute": 100,
            "filler_count": 0,
            "vocabulary_richness": 0.8,
            "reading_grade": 5.0,
            "reading_ease": 80.0,
            "keywords": ["hello", "world"],
            "pause_before": 0.0,
            "pause_after": 0.0,
            "speech_density": 0.9,
            "engagement_score": 75.0
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump(timeline, tf)
        timeline_path = Path(tf.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as sf:
        json.dump(segment_metrics, sf)
        metrics_path = Path(sf.name)

    config = ClipSelectorConfig(top_k=1)
    selector = ClipSelector(config=config)
    result = selector.produce_candidates(str(timeline_path), str(metrics_path))
    assert isinstance(result, CandidateClipsResult)
    assert len(result.candidates) == 1
    assert result.candidates[0].score > 0

    # Cleanup
    timeline_path.unlink()
    metrics_path.unlink()