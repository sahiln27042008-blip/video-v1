"""Tests for AI planner."""

import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from src.planner import Planner
from src.models import FinalClipPlan

def test_planner_prepare_context():
    # Create minimal input files
    clips = {"candidates": [{"clip_id": 1, "score": 90}]}
    timeline = {"segments": [{"start": 0, "end": 10}]}
    people = {"people": [{"person_id": "p1"}]}
    metrics = {"summary": {"duration": 100}}
    seg_metrics = [{"segment_id": 1, "engagement": 80}]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as cf:
        json.dump(clips, cf)
        clips_path = Path(cf.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump(timeline, tf)
        timeline_path = Path(tf.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pf:
        json.dump(people, pf)
        people_path = Path(pf.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as mf:
        json.dump(metrics, mf)
        metrics_path = Path(mf.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as sf:
        json.dump(seg_metrics, sf)
        seg_metrics_path = Path(sf.name)

    planner = Planner()
    context = planner.prepare_context(
        str(clips_path),
        str(timeline_path),
        str(people_path),
        str(metrics_path),
        str(seg_metrics_path),
    )
    assert "clips" in context
    assert "timeline" in context

    # Cleanup
    clips_path.unlink()
    timeline_path.unlink()
    people_path.unlink()
    metrics_path.unlink()
    seg_metrics_path.unlink()