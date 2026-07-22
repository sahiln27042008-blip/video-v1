"""Scene detection module for Video Brain."""
from .detector import SceneDetector
from .models import Scene, SceneDetectionResult

__all__ = ["SceneDetector", "Scene", "SceneDetectionResult"]