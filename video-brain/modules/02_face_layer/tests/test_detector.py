"""Tests for the face detector."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.detector import FaceDetector
from src.models import Face, SceneFaces, FaceDetectionResult

def test_detector_init():
    detector = FaceDetector(threshold=0.6, min_face_size=30)
    assert detector.threshold == 0.6
    assert detector.min_face_size == 30

def test_iou():
    detector = FaceDetector()
    bbox1 = [10, 10, 20, 20]
    bbox2 = [15, 15, 20, 20]  # overlap
    assert detector.iou(bbox1, bbox2) > 0

    bbox3 = [100, 100, 10, 10]
    assert detector.iou(bbox1, bbox3) == 0.0