"""Scene detection using PySceneDetect (0.7+ API)."""

import logging
from pathlib import Path

from scenedetect import open_video, SceneManager, ContentDetector

from .models import Scene, SceneDetectionResult, format_timedelta

logger = logging.getLogger(__name__)


class SceneDetector:
    """Detects scenes in a video using content-aware detection."""

    def __init__(
        self,
        threshold: float = 30.0,
        min_scene_len: int = 15,
    ):
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def detect(self, video_path: str) -> SceneDetectionResult:
        """Run scene detection on a video file."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video (no context manager, no explicit close needed)
        video = open_video(str(path))
        try:
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(
                    threshold=self.threshold,
                    min_scene_len=self.min_scene_len,
                )
            )
            scene_manager.detect_scenes(video)
        except Exception as e:
            raise RuntimeError(f"Scene detection failed: {e}") from e

        # Get results
        scene_list = scene_manager.get_scene_list()
        fps = video.frame_rate
        duration = video.duration  # seconds

        scenes = []
        for idx, (start_frame, end_frame) in enumerate(scene_list, start=1):
            start_time = start_frame.get_seconds()
            end_time = end_frame.get_seconds()
            scenes.append(
                Scene(
                    id=idx,
                    start=format_timedelta(start_time),
                    end=format_timedelta(end_time),
                    duration=end_time - start_time,
                )
            )

        return SceneDetectionResult(
            video=video_path,
            duration=duration,
            fps=fps,
            scene_count=len(scenes),
            scenes=scenes,
        )