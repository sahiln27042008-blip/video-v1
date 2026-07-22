"""Face detection module for Video Brain."""
from .detector import FaceDetector
from .models import FaceDetectionResult, Face, SceneFaces

__all__ = ["FaceDetector", "FaceDetectionResult", "Face", "SceneFaces"]