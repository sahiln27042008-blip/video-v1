"""Face detection using InsightFace."""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from .models import Face, SceneFaces, FaceDetectionResult

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects faces in video scenes using InsightFace."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_face_size: int = 20,
        sample_interval: float = 0.5,
        max_faces_per_scene: int = 50,
    ):
        self.threshold = threshold
        self.min_face_size = min_face_size
        self.sample_interval = sample_interval
        self.max_faces_per_scene = max_faces_per_scene
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as e:
                raise ImportError(
                    "InsightFace not installed. Please run: pip install insightface onnxruntime"
                ) from e
            self._model = FaceAnalysis(name="buffalo_l")
            self._model.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace model initialized.")
        return self._model

    def detect_in_frame(self, frame: np.ndarray) -> List[Tuple[List[float], float]]:
        """Detect faces in a single frame."""
        # Convert BGR to RGB (InsightFace expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.model.get(frame_rgb)
        print(f"  Model returned {len(faces)} faces")
        detected = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
            if w < self.min_face_size or h < self.min_face_size:
                print(f"  Face too small: {w}x{h} < {self.min_face_size}")
                continue
            confidence = face.det_score
            if confidence < self.threshold:
                print(f"  Face confidence {confidence:.3f} < {self.threshold}")
                continue
            detected.append(([x, y, w, h], confidence))
            print(f"  Detected face at ({x},{y}) {w}x{h} conf={confidence:.3f}")
        return detected

    def iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def collect_faces_from_scene(
        self, cap: cv2.VideoCapture, start_sec: float, end_sec: float
    ) -> List[Face]:
        print(f"Collecting faces from {start_sec:.2f}s to {end_sec:.2f}s")
        current_time = start_sec
        collected = []  # list of (bbox, confidence)
        while current_time < end_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = cap.read()
            if not ret:
                print(f"  Failed to read frame at {current_time:.2f}s")
                break
            print(f"  Read frame at {current_time:.2f}s, shape={frame.shape}")
            detected = self.detect_in_frame(frame)
            print(f"  Detected {len(detected)} faces")
            for bbox, conf in detected:
                merged = False
                for i, (existing_bbox, existing_conf) in enumerate(collected):
                    if self.iou(bbox, existing_bbox) > 0.5:
                        if conf > existing_conf:
                            collected[i] = (bbox, conf)
                        merged = True
                        break
                if not merged:
                    collected.append((bbox, conf))
                    print(f"  Added face #{len(collected)} with bbox {bbox} conf {conf:.3f}")
                    if len(collected) >= self.max_faces_per_scene:
                        break
            current_time += self.sample_interval
            if len(collected) >= self.max_faces_per_scene:
                break
        print(f"Total faces collected in scene: {len(collected)}")
        faces = []
        for idx, (bbox, conf) in enumerate(collected, start=1):
            faces.append(Face(face_id=idx, bbox=bbox, confidence=conf))
        return faces

    def detect(
        self, video_path: str, scenes: List[dict]
    ) -> FaceDetectionResult:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Test reading a single frame to ensure video works
        ret, test_frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Could not read any frame from video")
        print(f"Test frame read: shape={test_frame.shape}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset

        scene_faces_list = []
        for scene in scenes:
            scene_id = scene["id"]
            start_str = scene["start"]
            end_str = scene["end"]

            def parse_time(s: str) -> float:
                parts = s.split(":")
                if len(parts) == 3:
                    h, m, sec = parts
                    return float(h) * 3600 + float(m) * 60 + float(sec)
                else:
                    return float(s)

            start_sec = parse_time(start_str)
            end_sec = parse_time(end_str)
            faces = self.collect_faces_from_scene(cap, start_sec, end_sec)
            scene_faces_list.append(
                SceneFaces(scene_id=scene_id, faces=faces)
            )

        cap.release()
        return FaceDetectionResult(
            video=video_path,
            scenes=scene_faces_list,
        )