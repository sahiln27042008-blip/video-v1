"""Main orchestrator for candidate clip generation."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import timedelta

from .models import CandidateClip, CandidateClipReason, CandidateClipsResult
from .config import ClipSelectorConfig
from .scorer import score_segment, compute_individual_scores, compute_final_score
from .merger import merge_segments

logger = logging.getLogger(__name__)

def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

class ClipSelector:
    """Generate candidate clips from timeline and metrics."""

    def __init__(self, config: Optional[ClipSelectorConfig] = None):
        self.config = config or ClipSelectorConfig()
        self.config.validate()

    def load_json(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_inputs(
        self,
        timeline_path: str,
        segment_metrics_path: str,
        people_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load all required inputs and return a unified structure."""
        timeline_data = self.load_json(timeline_path)
        segment_metrics_data = self.load_json(segment_metrics_path)

        # Build a map from segment ID to metrics
        metrics_map = {m["segment_id"]: m for m in segment_metrics_data}

        # Combine timeline segments with metrics
        segments = []
        for seg in timeline_data.get("segments", []):
            seg_id = len(segments) + 1  # assign ID if not present
            if "segment_id" not in seg:
                seg["segment_id"] = seg_id
            # Merge metrics
            metrics = metrics_map.get(seg_id, {})
            # We'll use the timeline segment as base, but override with metrics where present
            merged = {**seg, **metrics}
            # Ensure start/end are floats
            if isinstance(merged.get("start"), str):
                merged["start"] = self._parse_time(merged["start"])
            if isinstance(merged.get("end"), str):
                merged["end"] = self._parse_time(merged["end"])
            merged["duration"] = merged["end"] - merged["start"]
            # Add person if not already
            if "person" not in merged:
                merged["person"] = seg.get("person")
            # Add speaker
            if "speaker" not in merged:
                merged["speaker"] = seg.get("speaker_label") or seg.get("speaker", "UNKNOWN")
            # Add text
            if "text" not in merged:
                merged["text"] = seg.get("text", "")
            segments.append(merged)

        return {
            "segments": segments,
            "video": timeline_data.get("video", ""),
        }

    def _parse_time(self, time_str: str) -> float:
        """Convert HH:MM:SS.mmm to seconds."""
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        else:
            return float(time_str)

    def score_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score each segment individually."""
        scored = []
        for seg in segments:
            result = score_segment(seg, self.config)
            seg["score"] = result["score"]
            seg["sub_scores"] = result["sub_scores"]
            scored.append(seg)
        return scored

    def filter_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove segments that fail duration or content thresholds."""
        filtered = []
        for seg in segments:
            dur = seg.get("duration", 0.0)
            if dur < self.config.min_duration_seconds:
                continue
            if dur > self.config.max_duration_seconds:
                continue
            # Almost empty transcript
            text = seg.get("text", "")
            if len(text.strip()) < 5:
                continue
            # Could add more filters
            filtered.append(seg)
        return filtered

    def merge_and_score_clips(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge adjacent segments and compute final score for each clip."""
        # Sort segments by start time
        segments_sorted = sorted(segments, key=lambda x: x.get("start", 0.0))
        # Merge
        merged = merge_segments(segments_sorted, self.config)
        # Score each merged clip
        scored_clips = []
        for clip in merged:
            # Compute individual scores for the merged clip using its aggregated metrics
            # We'll use the same scoring functions but with aggregated metrics
            # Recompute sub_scores
            sub_scores = compute_individual_scores(clip, self.config)
            # Compute final score
            final_score = compute_final_score(sub_scores, self.config)
            clip["score"] = final_score
            clip["sub_scores"] = sub_scores
            scored_clips.append(clip)
        return scored_clips

    def produce_candidates(
        self,
        timeline_path: str,
        segment_metrics_path: str,
        people_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> CandidateClipsResult:
        """Full pipeline: load, score, merge, filter, sort, output."""
        # Load data
        data = self.load_inputs(timeline_path, segment_metrics_path, people_path, metrics_path)
        segments = data["segments"]
        video = data["video"] or video_path or ""

        # Score segments
        scored_segments = self.score_segments(segments)
        # Filter
        filtered_segments = self.filter_segments(scored_segments)
        # Merge and score clips
        merged_clips = self.merge_and_score_clips(filtered_segments)
        # Sort descending by score
        merged_clips.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        # Take top K
        top_clips = merged_clips[:self.config.top_k]

        # Build CandidateClip objects
        candidates = []
        for idx, clip in enumerate(top_clips, start=1):
            # Extract keywords (top max_keywords_per_clip)
            keywords = clip.get("keywords", [])
            if isinstance(keywords, list):
                if keywords and isinstance(keywords[0], dict):
                    keywords = [k.get("keyword", "") for k in keywords[:self.config.max_keywords_per_clip]]
                else:
                    keywords = keywords[:self.config.max_keywords_per_clip]
            else:
                keywords = []

            reason = CandidateClipReason(
                engagement=clip["sub_scores"].get("engagement", 0.0),
                speech_density=clip["sub_scores"].get("speech_density", 0.0),
                keywords=clip["sub_scores"].get("keyword", 0.0),
                duration=clip["sub_scores"].get("duration", 0.0),
                readability=clip["sub_scores"].get("readability", 0.0),
                vocabulary_richness=clip["sub_scores"].get("vocab_richness", 0.0),
                filler_penalty=clip["sub_scores"].get("filler_penalty", 0.0),
            )

            candidates.append(CandidateClip(
                candidate_id=idx,
                start=format_time(clip["start"]),
                end=format_time(clip["end"]),
                duration=clip["duration"],
                person=clip.get("person"),
                speaker=clip["speaker"],
                segments=clip["segments"],
                score=clip["score"],
                keywords=keywords,
                transcript=clip["transcript"],
                reason=reason,
            ))

        result = CandidateClipsResult(
            module="clip_selector",
            version="0.1.0",
            video=video,
            candidates=candidates,
        )
        return result