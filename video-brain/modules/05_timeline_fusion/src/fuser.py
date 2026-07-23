"""Timeline fusion: merge scenes, faces, identities, conversation."""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
from collections import defaultdict, Counter

from .models import TimelineSegment, TimelineResult, Word, PersonInfo, PeopleResult

logger = logging.getLogger(__name__)

class TimelineFuser:
    """Fuse all data sources into a single timeline."""

    def __init__(self):
        self.scenes = []
        self.identities = {}
        self.faces_by_scene = {}
        self.conversation_segments = []
        self.people_mapping = {}  # speaker -> person_id
        self.person_segments = defaultdict(set)  # person_id -> set of segment indices
        self.face_to_person = {}  # face_id -> person_id

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
        self.faces_by_scene = {}
        for scene in faces_data.get("scenes", []):
            self.faces_by_scene[scene["scene_id"]] = scene.get("faces", [])

        identities_data = self.load_json(identities_path)
        self.identities = {}
        for person in identities_data.get("people", []):
            self.identities[person["person_id"]] = person
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
        best_overlap = 0.0
        for scene in self.scenes:
            s_start = self.parse_time(scene["start"])
            s_end = self.parse_time(scene["end"])
            overlap_start = max(s_start, start_sec)
            overlap_end = min(s_end, end_sec)
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_scene = scene
        return best_scene

    def get_visible_people(self, scene_id: int) -> List[str]:
        """Return list of person_ids visible in this scene."""
        return self.identity_scenes.get(scene_id, [])

    def _map_speakers_to_persons(self, timeline_segments: List[dict]) -> Dict[str, Optional[str]]:
        """
        Map each speaker to the most frequent visible person during their speaking segments.
        Returns dict speaker -> person_id or None.
        """
        speaker_person_counts = defaultdict(Counter)
        for seg in timeline_segments:
            speaker = seg.get("speaker_label")
            if not speaker:
                continue
            visible = seg.get("visible_people", [])
            if visible:
                for person in visible:
                    speaker_person_counts[speaker][person] += 1
        mapping = {}
        for speaker, counter in speaker_person_counts.items():
            if counter:
                best_person = counter.most_common(1)[0][0]
                mapping[speaker] = best_person
            else:
                mapping[speaker] = None
        return mapping

    def _generate_people_json(self, timeline_segments: List[dict], speaker_mapping: Dict[str, Optional[str]]) -> List[PersonInfo]:
        """Build people.json from timeline segments and speaker mapping."""
        person_segments = defaultdict(set)
        for idx, seg in enumerate(timeline_segments):
            person = seg.get("person")
            if person:
                person_segments[person].add(idx + 1)
            else:
                visible = seg.get("visible_people", [])
                if len(visible) == 1:
                    person_segments[visible[0]].add(idx + 1)

        face_to_person = {}
        for scene_id, persons in self.identity_scenes.items():
            faces = self.faces_by_scene.get(scene_id, [])
            if len(faces) == len(persons) and len(faces) > 0:
                for face, person in zip(faces, persons):
                    face_id = face.get("face_id")
                    if face_id is not None:
                        face_to_person[face_id] = person

        people_list = []
        for person_id, seg_set in person_segments.items():
            speaker = None
            for sp, p in speaker_mapping.items():
                if p == person_id:
                    speaker = sp
                    break
            face_ids = [fid for fid, pid in face_to_person.items() if pid == person_id]
            people_list.append(PersonInfo(
                person_id=person_id,
                speaker=speaker,
                face_ids=face_ids,
                segments=sorted(seg_set)
            ))
        return people_list

    def fuse(self) -> Tuple[TimelineResult, List[PersonInfo]]:
        """Main entry point: produce timeline and people list."""
        segments_out = []
        for conv_seg in self.conversation_segments:
            start_str = conv_seg["start"]
            end_str = conv_seg["end"]
            start_sec = self.parse_time(start_str)
            end_sec = self.parse_time(end_str)
            speaker_label = conv_seg["speaker"]
            text = conv_seg["text"]
            words = [Word(**w) for w in conv_seg.get("words", [])]

            scene = self.find_scene_for_time(start_sec, end_sec)
            scene_id = scene["id"] if scene else None
            visible = []
            active_speaker = None
            if scene is not None:
                visible = self.get_visible_people(scene_id)
                if len(visible) == 1:
                    active_speaker = visible[0]

            segment = {
                "start": start_str,
                "end": end_str,
                "scene_id": scene_id,
                "camera": None,
                "visible_people": visible,
                "active_speaker": active_speaker,
                "speaker_label": speaker_label,
                "text": text,
                "words": words,
            }
            segments_out.append(segment)

        speaker_mapping = self._map_speakers_to_persons(segments_out)

        for seg in segments_out:
            speaker = seg["speaker_label"]
            seg["person"] = speaker_mapping.get(speaker)

        people_list = self._generate_people_json(segments_out, speaker_mapping)

        timeline_segments = []
        for seg in segments_out:
            timeline_segments.append(TimelineSegment(
                start=seg["start"],
                end=seg["end"],
                scene_id=seg["scene_id"],
                camera=seg["camera"],
                visible_people=seg["visible_people"],
                active_speaker=seg["active_speaker"],
                person=seg["person"],
                speaker_label=seg["speaker_label"],
                text=seg["text"],
                words=seg["words"],
            ))

        timeline_segments.sort(key=lambda x: self.parse_time(x.start))

        result = TimelineResult(
            video="",  # will be set later
            segments=timeline_segments,
        )
        return result, people_list

    def process(
        self,
        video_path: str,
        scenes_path: str,
        faces_path: str,
        identities_path: str,
        conversation_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[TimelineResult, List[PersonInfo]]:
        """Full pipeline: load, fuse, save if output path given."""
        self.load_data(scenes_path, faces_path, identities_path, conversation_path)
        result, people_list = self.fuse()
        result.video = video_path

        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            # Write timeline.json
            with open(output_path, "w") as f:
                f.write(result.model_dump_json(indent=2))
            logger.info(f"Timeline written to {output_path}")

            # Also write timeline_with_words.json (same content)
            timeline_words_path = output_dir / "timeline_with_words.json"
            shutil.copy(output_path, timeline_words_path)
            logger.info(f"Timeline with words written to {timeline_words_path}")

            # Write people.json
            people_path = output_dir / "people.json"
            people_result = PeopleResult(people=people_list)
            with open(people_path, "w") as f:
                f.write(people_result.model_dump_json(indent=2))
            logger.info(f"people.json written to {people_path}")

        return result, people_list