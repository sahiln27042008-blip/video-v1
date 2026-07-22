"""Tests for the scene detector."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.detector import SceneDetector
from src.models import Scene, SceneDetectionResult

def test_detector_init():
    """Test detector initialization."""
    detector = SceneDetector(threshold=40.0, min_scene_len=20)
    assert detector.threshold == 40.0
    assert detector.min_scene_len == 20

@patch("src.detector.VideoManager")
@patch("src.detector.SceneManager")
def test_detect_success(mock_scene_manager, mock_video_manager):
    """Test successful detection with mocked video manager."""
    # Setup mocks
    mock_vm_instance = MagicMock()
    mock_vm_instance.get_framerate.return_value = 30.0
    mock_vm_instance.get_duration.return_value = 120.0
    mock_vm_instance.get_num_frames.return_value = 3600
    mock_video_manager.return_value = mock_vm_instance

    mock_sm_instance = MagicMock()
    mock_sm_instance.get_scene_list.return_value = [
        (MagicMock(get_seconds=lambda: 0.0), MagicMock(get_seconds=lambda: 8.42)),
        (MagicMock(get_seconds=lambda: 8.42), MagicMock(get_seconds=lambda: 24.81)),
    ]
    mock_scene_manager.return_value = mock_sm_instance

    detector = SceneDetector()
    result = detector.detect("dummy.mp4")

    assert isinstance(result, SceneDetectionResult)
    assert result.video == "dummy.mp4"
    assert result.duration == 120.0
    assert result.fps == 30.0
    assert result.scene_count == 2
    assert result.scenes[0].id == 1
    assert result.scenes[0].start == "00:00:00.000"
    assert result.scenes[0].end == "00:00:08.420"
    assert result.scenes[1].id == 2
    assert result.scenes[1].start == "00:00:08.420"
    assert result.scenes[1].end == "00:00:24.810"

def test_detect_file_not_found():
    """Test detection with missing video file."""
    detector = SceneDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("nonexistent.mp4")