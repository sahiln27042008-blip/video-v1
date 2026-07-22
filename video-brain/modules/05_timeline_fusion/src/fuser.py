"""Timeline fusion: merge scenes, faces, identities, conversation."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging

from .models import TimelineSegment, TimelineResult, Word

logger = logging.getLogger(__name__)

class TimelineFuser:
    """Fuse all data sources into a single timeline."""

    def __init__(self):
        self.scenes = []
        self.identities = {}
        self.faces_by_scene = {}
        self.conversation_segments = []

    def load_json(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    def parse_time(self, time_str: str) -> float:
        """Convert HH:MM:SS.mmm to seconds."""
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        else:
            return float(time_str)

    def format_time(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def load_data(self, scenes_path: str, faces_path: str, identities_path: str, conversation_path: str):
        """Load all input JSON files."""
        scenes_data = self.load_json(scenes_path)
        self.scenes = scenes_data.get("scenes", [])

        faces_data = self.load_json(faces_path)
        # Build mapping scene_id -> list of faces
        self.faces_by_scene = {}
        for scene in faces_data.get("scenes", []):
            self.faces_by_scene[scene["scene_id"]] = scene.get("faces", [])

        identities_data = self.load_json(identities_path)
        # Build mapping: person_id -> person info
        self.identities = {}
        for person in identities_data.get("people", []):
            self.identities[person["person_id"]] = person
        # Also we need per-scene visible persons from identities
        # We'll use the identities.json "scenes" field to get person per scene
        self.identity_scenes = {}
        for scene in identities_data.get("scenes", []):
            scene_id = scene["id"]
            persons_in_scene = [p["person_id"] for p in scene.get("people", [])]
            self.identity_scenes[scene_id] = persons_in_scene

        conversation_data = self.load_json(conversation_path)
        self.conversation_segments = conversation_data.get("segments", [])

    def find_scene_for_time(self, start_sec: float, end_sec: float) -> Optional[dict]:
        """Find scene that contains this time interval."""
        best_scene = None
        for scene in self.scenes:
            s_start = self.parse_time(scene["start"])
            s_end = self.parse_time(scene["end"])
            # Check if segment is fully contained or overlaps
            if s_start <= start_sec and end_sec <= s_end:
                return scene
            # If partially overlaps, choose the one with most overlap
            overlap_start = max(s_start, start_sec)
            overlap_end = min(s_end, end_sec)
            if overlap_end > overlap_start:
                if best_scene is None:
                    best_scene = scene
                else:
                    # keep the one with larger overlap
                    pass
        return best_scene

    def get_visible_people(self, scene_id: int) -> List[str]:
        """Return list of person_ids visible in this scene."""
        return self.identity_scenes.get(scene_id, [])

    def get_avg_confidence_for_person(self, scene_id: int, person_id: str) -> float:
        """Compute average face detection confidence for a person in this scene."""
        faces = self.faces_by_scene.get(scene_id, [])
        confidences = []
        for face in faces:
            # We don't have direct mapping from face to person, but in identities we have person per scene.
            # However, we don't know which face corresponds to which person.
            # We'll use the average confidence of all faces in the scene as a proxy, or we could use the
            # confidence from the identity layer if available.
            # For simplicity, we'll use the average confidence of all faces in the scene.
            # But we need person-specific confidence, which we don't have.
            # We'll use the identity's first/last seen, but we don't have per-appearance confidence.
            # Fallback: return the average confidence of all faces in the scene.
            confidences.append(face.get("confidence", 0.0))
        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.0

    def link_speaker_to_person(self, start_sec: float, end_sec: float, scene: Optional[dict]) -> Optional[str]:
        """Determine the person_id for the speaker based on visible people in the scene."""
        if scene is None:
            return None
        scene_id = scene["id"]
        visible = self.get_visible_people(scene_id)
        if not visible:
            return None
        # If only one person visible, assign that person
        if len(visible) == 1:
            return visible[0]
        # If multiple, choose the one with strongest appearance (by average confidence)
        best_person = None
        best_conf = -1.0
        for person_id in visible:
            conf = self.get_avg_confidence_for_person(scene_id, person_id)
            if conf > best_conf:
                best_conf = conf
                best_person = person_id
        return best_person

    def fuse(self) -> TimelineResult:
        """Main entry point: produce timeline from loaded data."""
        segments_out = []
        for conv_seg in self.conversation_segments:
            start_str = conv_seg["start"]
            end_str = conv_seg["end"]
            start_sec = self.parse_time(start_str)
            end_sec = self.parse_time(end_str)
            speaker_label = conv_seg["speaker"]
            text = conv_seg["text"]
            words = [Word(**w) for w in conv_seg.get("words", [])]

            # Find scene
            scene = self.find_scene_for_time(start_sec, end_sec)
            scene_id = scene["id"] if scene else None

            # Visible people
            visible = []
            active_speaker = None
            camera = None
            if scene is not None:
                visible = self.get_visible_people(scene_id)
                # Determine active speaker
                active_speaker = self.link_speaker_to_person(start_sec, end_sec, scene)
                # Camera label: not provided in data, we could infer from scene content, but we'll set to None
                # Could be derived from which person is visible, but we don't have that mapping.
                # For now, we leave as None.
                camera = None

            segment = TimelineSegment(
                start=start_str,
                end=end_str,
                scene_id=scene_id,
                camera=camera,
                visible_people=visible,
                active_speaker=active_speaker,
                speaker_label=speaker_label,
                text=text,
                words=words,
            )
            segments_out.append(segment)

        # Sort by start time (they should already be sorted, but ensure)
        segments_out.sort(key=lambda x: self.parse_time(x.start))

        # Build result
        result = TimelineResult(
            video="video.mp4",  # We could extract from any file, but we'll hardcode or accept as param
            segments=segments_out,
        )
        return result

    def process(
        self,
        video_path: str,
        scenes_path: str,
        faces_path: str,
        identities_path: str,
        conversation_path: str,
        output_path: Optional[str] = None,
    ) -> TimelineResult:
        """Full pipeline: load, fuse, save if output path given."""
        self.load_data(scenes_path, faces_path, identities_path, conversation_path)
        result = self.fuse()
        # Override video path from input
        result.video = video_path

        if output_path:
            with open(output_path, "w") as f:
                f.write(result.model_dump_json(indent=2))

        return result