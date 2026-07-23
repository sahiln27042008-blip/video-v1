"""Tests for renderer."""

import json
import tempfile
from pathlib import Path
import pytest
from src.renderer import Renderer
from src.models import RenderConfig

def test_renderer_initialization():
    config = RenderConfig(
        video_path="video.mp4",
        editor_plan_path="editor.json",
        candidate_clips_path="candidates.json",
        timeline_words_path="timeline.json",
        output_dir="./output",
        temp_dir="./temp"
    )
    renderer = Renderer(config)
    assert renderer.config == config