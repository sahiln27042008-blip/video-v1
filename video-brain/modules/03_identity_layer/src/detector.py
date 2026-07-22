"""Identity tracking using InsightFace embeddings and DBSCAN clustering."""

import logging
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .models import Person, ScenePerson, SceneIdentity, IdentityResult

logger = logging.getLogger(__name__)


class IdentityDetector:
    """Track identities across scenes using face embeddings."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 2,
        metric: str = "cosine",
        min_confidence: float = 0.5,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.min_confidence = min_confidence
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError:
                raise ImportError("InsightFace not installed")
            self._model = FaceAnalysis(name="buffalo_l")
            self._model.prepare(ctx_id=0, det_size=(640, 640))
        return self._model

    def iou(self, bbox1, bbox2):
        """IoU for [x, y, w, h]."""
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

    def load_faces_from_json(self, faces_json_path: str, scenes_json_path: str) -> Tuple[List[dict], List[dict]]:
        with open(faces_json_path, 'r') as f:
            faces_data = json.load(f)
        with open(scenes_json_path, 'r') as f:
            scenes_data = json.load(f)
        if isinstance(scenes_data, dict) and "scenes" in scenes_data:
            scenes_list = scenes_data["scenes"]
        else:
            scenes_list = scenes_data
        return faces_data, scenes_list

    def extract_embeddings(
        self,
        video_path: str,
        faces_data: dict,
        scenes_list: List[dict]
    ) -> Tuple[List[np.ndarray], List[dict]]:
        """
        For each face in faces_data, extract embedding by re-detecting on the full frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        embeddings = []
        metadata = []

        # Build a lookup: scene_id -> list of faces
        scene_faces_map = {}
        for sf in faces_data.get("scenes", []):
            scene_faces_map[sf["scene_id"]] = sf["faces"]

        for scene in scenes_list:
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

            # Get faces for this scene
            faces_for_scene = scene_faces_map.get(scene_id, [])
            if not faces_for_scene:
                continue

            # Sample at mid-point of scene
            mid_sec = (start_sec + end_sec) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000)
            ret, frame = cap.read()
            if not ret:
                # Try from start
                cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue

            # Detect faces in the full frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.model.get(frame_rgb)
            print(f"Scene {scene_id}: found {len(detections)} faces in frame")

            for face_meta in faces_for_scene:
                if face_meta["confidence"] < self.min_confidence:
                    continue
                bbox = face_meta["bbox"]  # [x, y, w, h]
                # Find best matching detection by IoU
                best_match = None
                best_iou = 0.0
                for det in detections:
                    det_bbox = det.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
                    det_bbox_wh = [det_bbox[0], det_bbox[1], det_bbox[2]-det_bbox[0], det_bbox[3]-det_bbox[1]]
                    iou = self.iou(bbox, det_bbox_wh)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = det
                if best_match is None or best_iou < 0.3:
                    print(f"  No match for bbox {bbox}, best IoU {best_iou:.2f}")
                    continue
                # Use embedding from best_match
                emb = best_match.normed_embedding
                if emb is not None:
                    embeddings.append(emb)
                    metadata.append({
                        "scene_id": scene_id,
                        "face_id": face_meta["face_id"],
                        "bbox": bbox,
                        "confidence": face_meta["confidence"],
                        "timestamp": mid_sec,
                        "det_conf": best_match.det_score,
                    })
                    print(f"  Extracted embedding for face {face_meta['face_id']} with IoU {best_iou:.2f}")

        cap.release()
        print(f"Total embeddings extracted: {len(embeddings)}")
        return embeddings, metadata

    def cluster_identities(self, embeddings: List[np.ndarray]) -> np.ndarray:
        if len(embeddings) == 0:
            return np.array([])
        X = np.array(embeddings)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        labels = clustering.fit_predict(X)
        return labels

    def build_identity_result(
        self,
        video_path: str,
        scenes_list: List[dict],
        metadata: List[dict],
        labels: np.ndarray,
        embeddings: List[np.ndarray]
    ) -> IdentityResult:
        # Map labels to person IDs
        unique_labels = set(labels)
        label_to_person = {}
        person_id_counter = 1
        for lab in unique_labels:
            if lab == -1:
                continue
            label_to_person[lab] = f"person_{person_id_counter}"
            person_id_counter += 1

        person_map = {}
        for idx, lab in enumerate(labels):
            if lab == -1:
                pid = f"person_{person_id_counter}"
                person_id_counter += 1
                person_map[idx] = pid
            else:
                person_map[idx] = label_to_person[lab]

        # Build summary
        person_summary = {}
        for idx, pid in person_map.items():
            meta = metadata[idx]
            scene_id = meta["scene_id"]
            timestamp = meta["timestamp"]
            if pid not in person_summary:
                person_summary[pid] = {
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "appearances": set()
                }
            person_summary[pid]["appearances"].add(scene_id)
            if timestamp < person_summary[pid]["first_seen"]:
                person_summary[pid]["first_seen"] = timestamp
            if timestamp > person_summary[pid]["last_seen"]:
                person_summary[pid]["last_seen"] = timestamp

        def fmt_time(sec):
            hours = int(sec // 3600)
            minutes = int((sec % 3600) // 60)
            secs = int(sec % 60)
            millis = int((sec - int(sec)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

        people_list = []
        for pid, info in person_summary.items():
            people_list.append(Person(
                person_id=pid,
                first_seen=fmt_time(info["first_seen"]),
                last_seen=fmt_time(info["last_seen"]),
                appearances=len(info["appearances"])
            ))

        # Build scene identities
        scene_identities = []
        for scene in scenes_list:
            scene_id = scene["id"]
            scene_faces = []
            for idx, meta in enumerate(metadata):
                if meta["scene_id"] == scene_id:
                    pid = person_map[idx]
                    emb = embeddings[idx]
                    lab = labels[idx]
                    if lab != -1 and lab in label_to_person:
                        cluster_indices = [i for i, l in enumerate(labels) if l == lab]
                        cluster_embs = np.array([embeddings[i] for i in cluster_indices])
                        centroid = np.mean(cluster_embs, axis=0)
                        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                        sim = cosine_similarity([emb], [centroid])[0][0]
                    else:
                        sim = 1.0
                    scene_faces.append(ScenePerson(
                        person_id=pid,
                        bbox=meta["bbox"],
                        confidence=meta["confidence"],
                        embedding_similarity=float(sim)
                    ))
            scene_identities.append(SceneIdentity(
                id=scene_id,
                start=scene["start"],
                end=scene["end"],
                people=scene_faces
            ))

        return IdentityResult(
            video=video_path,
            people=people_list,
            scenes=scene_identities
        )

    def process(
        self,
        video_path: str,
        faces_json_path: str,
        scenes_json_path: str
    ) -> IdentityResult:
        faces_data, scenes_list = self.load_faces_from_json(faces_json_path, scenes_json_path)
        embeddings, metadata = self.extract_embeddings(video_path, faces_data, scenes_list)
        if len(embeddings) == 0:
            people_list = []
            scene_identities = []
            for scene in scenes_list:
                scene_identities.append(SceneIdentity(
                    id=scene["id"],
                    start=scene["start"],
                    end=scene["end"],
                    people=[]
                ))
            return IdentityResult(video=video_path, people=[], scenes=scene_identities)
        labels = self.cluster_identities(embeddings)
        return self.build_identity_result(video_path, scenes_list, metadata, labels, embeddings)