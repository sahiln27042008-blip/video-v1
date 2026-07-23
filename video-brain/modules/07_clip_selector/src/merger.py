"""Merging of adjacent segments into candidate clips."""

from typing import List, Dict, Any, Optional
from .config import ClipSelectorConfig

def merge_segments(
    segments: List[Dict[str, Any]],
    config: ClipSelectorConfig
) -> List[Dict[str, Any]]:
    """
    Merge adjacent segments that belong to the same person/speaker and are within gap.

    Args:
        segments: List of segment dicts (already scored and sorted by time).
        config: Configuration.

    Returns:
        List of merged clip dicts.
    """
    if not segments:
        return []

    clips = []
    current = None
    prev_end = None

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_speaker = seg.get("speaker", "")
        seg_person = seg.get("person")

        if current is None:
            # Start first clip
            current = {
                "start": seg_start,
                "end": seg_end,
                "speaker": seg_speaker,
                "person": seg_person,
                "segments": [seg["segment_id"]],
                "transcript": seg.get("text", ""),
                "keywords": seg.get("keywords", []),
                "duration": seg_end - seg_start,
                "engagement_score": seg.get("engagement_score", 0.0),
                "speech_density": seg.get("speech_density", 0.0),
                "vocabulary_richness": seg.get("vocabulary_richness", 0.0),
                "readability": seg.get("reading_ease", 0.0),
                "word_count": seg.get("word_count", 0),
                "filler_count": seg.get("filler_count", 0),
                "reading_grade": seg.get("reading_grade", 0.0),
                "sub_scores": seg.get("sub_scores", {}),
                "score": seg.get("score", 0.0),
            }
            prev_end = seg_end
            continue

        # Check if merge conditions are met
        gap = seg_start - prev_end
        if gap <= config.merge_gap_seconds:
            same_person = (seg_person == current["person"]) if config.same_person_required else True
            same_speaker = (seg_speaker == current["speaker"]) if config.same_speaker_required else True
            if same_person and same_speaker:
                # Merge into current
                current["end"] = seg_end
                current["segments"].append(seg["segment_id"])
                current["transcript"] += " " + seg.get("text", "")
                # Combine keywords (deduplicate)
                current["keywords"] = list(set(current["keywords"] + seg.get("keywords", [])))
                current["duration"] = seg_end - current["start"]
                # Update duration-related scores
                current["word_count"] += seg.get("word_count", 0)
                current["filler_count"] += seg.get("filler_count", 0)
                # We'll recompute scores later, so just store raw
                current["engagement_score"] = (current["engagement_score"] + seg.get("engagement_score", 0.0)) / 2  # simple avg
                current["speech_density"] = (current["speech_density"] + seg.get("speech_density", 0.0)) / 2
                current["vocabulary_richness"] = max(current["vocabulary_richness"], seg.get("vocabulary_richness", 0.0))
                current["readability"] = max(current["readability"], seg.get("reading_ease", 0.0))
                current["sub_scores"] = {}  # will be recomputed
                current["score"] = 0.0  # will be recomputed
                prev_end = seg_end
                continue

        # Cannot merge, finalise current and start new
        clips.append(current)
        # Start new
        current = {
            "start": seg_start,
            "end": seg_end,
            "speaker": seg_speaker,
            "person": seg_person,
            "segments": [seg["segment_id"]],
            "transcript": seg.get("text", ""),
            "keywords": seg.get("keywords", []),
            "duration": seg_end - seg_start,
            "engagement_score": seg.get("engagement_score", 0.0),
            "speech_density": seg.get("speech_density", 0.0),
            "vocabulary_richness": seg.get("vocabulary_richness", 0.0),
            "readability": seg.get("reading_ease", 0.0),
            "word_count": seg.get("word_count", 0),
            "filler_count": seg.get("filler_count", 0),
            "reading_grade": seg.get("reading_grade", 0.0),
            "sub_scores": seg.get("sub_scores", {}),
            "score": seg.get("score", 0.0),
        }
        prev_end = seg_end

    if current:
        clips.append(current)

    return clips