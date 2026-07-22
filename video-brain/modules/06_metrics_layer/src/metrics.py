"""Metrics computation from timeline_with_words.json (global and per-segment)."""

import json
import re
import math
import statistics
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import textstat
import yake
from nltk.tokenize import sent_tokenize
import nltk

from .models import (
    MetricsResult,
    SummaryMetrics,
    SpeechMetrics,
    ReadabilityMetrics,
    KeywordMetrics,
    EngagementMetrics,
)

# Ensure NLTK tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def parse_time(time_str: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    else:
        return float(time_str)


class MetricsCalculator:
    """Compute all metrics from timeline_with_words.json."""

    FILLER_WORDS = {"uh", "um", "like", "you know", "actually", "basically", "literally", "so"}

    def __init__(self, timeline_path: Path):
        self.timeline_path = timeline_path
        with open(timeline_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.segments = self.data.get("segments", [])
        if not self.segments:
            raise ValueError("No segments found in timeline file")

        # Convert string timestamps to float and ensure words exist
        for seg in self.segments:
            if "start" in seg and isinstance(seg["start"], str):
                seg["start"] = parse_time(seg["start"])
            if "end" in seg and isinstance(seg["end"], str):
                seg["end"] = parse_time(seg["end"])
            for w in seg.get("words", []):
                if "start" in w and isinstance(w["start"], str):
                    w["start"] = parse_time(w["start"])
                if "end" in w and isinstance(w["end"], str):
                    w["end"] = parse_time(w["end"])
                if "confidence" not in w:
                    w["confidence"] = 1.0

        # Extract words for global metrics
        self.words = self._extract_words()
        self.full_text = " ".join([seg.get("text", "") for seg in self.segments])
        self.duration = self._compute_duration()
        self.scene_ids = self._get_scene_ids()

        # Precompute segment text and word lists for per-segment metrics
        self.segment_texts = [seg.get("text", "") for seg in self.segments]
        self.segment_words = [self._extract_words_from_segment(seg) for seg in self.segments]

    def _extract_words(self) -> List[Dict[str, Any]]:
        words = []
        for seg in self.segments:
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                    "confidence": w.get("confidence", 1.0),
                    "speaker": seg.get("speaker_label") or seg.get("speaker", "UNKNOWN"),
                    "segment_start": seg.get("start", 0.0),
                    "segment_end": seg.get("end", 0.0),
                })
        return words

    def _extract_words_from_segment(self, seg: dict) -> List[Dict[str, Any]]:
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": w.get("start", 0.0),
                "end": w.get("end", 0.0),
                "confidence": w.get("confidence", 1.0),
            })
        return words

    def _compute_duration(self) -> float:
        if self.words:
            return self.words[-1]["end"] - self.words[0]["start"]
        elif self.segments:
            starts = [seg.get("start", 0) for seg in self.segments if "start" in seg]
            ends = [seg.get("end", 0) for seg in self.segments if "end" in seg]
            if starts and ends:
                return max(ends) - min(starts)
        return 0.0

    def _get_scene_ids(self) -> List[int]:
        scenes = set()
        for seg in self.segments:
            if "scene_id" in seg and seg["scene_id"] is not None:
                scenes.add(seg["scene_id"])
        return sorted(scenes) if scenes else [0]

    # ---------- Global metrics (existing) ----------
    def compute_summary(self) -> SummaryMetrics:
        num_scenes = len(self.scene_ids)
        avg_scene_duration = self.duration / num_scenes if num_scenes > 0 else 0.0
        unique_words = set(w["word"].lower() for w in self.words if w["word"])
        total_words = len(self.words)
        vocab_richness = len(unique_words) / total_words if total_words > 0 else 0.0
        confidences = [w.get("confidence", 1.0) for w in self.words if "confidence" in w]
        avg_conf = statistics.mean(confidences) if confidences else 0.0
        return SummaryMetrics(
            video_duration=self.duration,
            num_scenes=num_scenes,
            average_scene_duration=avg_scene_duration,
            vocabulary_richness=vocab_richness,
            average_confidence=avg_conf,
        )

    def compute_speech(self) -> SpeechMetrics:
        if not self.words and not self.segments:
            return SpeechMetrics(
                total_words=0, words_per_minute=0.0, speaking_time=0.0, silence_time=0.0,
                speech_ratio=0.0, average_sentence_length=0.0, filler_word_count=0,
                longest_monologue=0.0, speakers_count=0, speaker_speaking_time={},
                average_pause_length=0.0, longest_pause=0.0,
            )
        total_words = len(self.words)
        duration_minutes = self.duration / 60.0 if self.duration > 0 else 1.0
        wpm = total_words / duration_minutes if duration_minutes > 0 else 0.0
        speaking_time = 0.0
        for seg in self.segments:
            start = seg.get("start", 0.0); end = seg.get("end", 0.0)
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                speaking_time += end - start
        silence_time = self.duration - speaking_time if self.duration > 0 else 0.0
        speech_ratio = speaking_time / self.duration if self.duration > 0 else 0.0
        sentences = sent_tokenize(self.full_text) if self.full_text else []
        avg_sentence_len = total_words / len(sentences) if sentences else 0.0
        filler_count = sum(1 for w in self.words if w["word"].lower() in self.FILLER_WORDS)
        # longest monologue
        longest_monologue = 0.0
        current_speaker = None; current_duration = 0.0
        for seg in self.segments:
            speaker = seg.get("speaker_label") or seg.get("speaker", "UNKNOWN")
            seg_dur = seg.get("end", 0.0) - seg.get("start", 0.0)
            if isinstance(seg_dur, (int, float)):
                if speaker == current_speaker:
                    current_duration += seg_dur
                else:
                    if current_duration > longest_monologue:
                        longest_monologue = current_duration
                    current_speaker = speaker
                    current_duration = seg_dur
        if current_duration > longest_monologue:
            longest_monologue = current_duration
        # speakers
        speaker_time = defaultdict(float)
        speakers = set()
        for seg in self.segments:
            sp = seg.get("speaker_label") or seg.get("speaker", "UNKNOWN")
            speakers.add(sp)
            seg_dur = seg.get("end", 0.0) - seg.get("start", 0.0)
            if isinstance(seg_dur, (int, float)):
                speaker_time[sp] += seg_dur
        speakers_count = len(speakers)
        pauses = []
        if len(self.words) > 1:
            for i in range(len(self.words) - 1):
                gap = self.words[i+1]["start"] - self.words[i]["end"]
                if gap > 0.01:
                    pauses.append(gap)
        avg_pause = statistics.mean(pauses) if pauses else 0.0
        longest_pause = max(pauses) if pauses else 0.0
        return SpeechMetrics(
            total_words=total_words, words_per_minute=wpm,
            speaking_time=speaking_time, silence_time=silence_time,
            speech_ratio=speech_ratio, average_sentence_length=avg_sentence_len,
            filler_word_count=filler_count, longest_monologue=longest_monologue,
            speakers_count=speakers_count, speaker_speaking_time=dict(speaker_time),
            average_pause_length=avg_pause, longest_pause=longest_pause,
        )

    def compute_readability(self, text: str = None) -> ReadabilityMetrics:
        if text is None:
            text = self.full_text if self.full_text else " "
        flesch = textstat.flesch_reading_ease(text)
        grade = textstat.flesch_kincaid_grade(text)
        return ReadabilityMetrics(flesch_reading_ease=flesch, flesch_kincaid_grade=grade)

    def compute_keywords(self, text: str = None, top_n: int = 20) -> KeywordMetrics:
        if text is None:
            text = self.full_text if self.full_text else " "
        extractor = yake.KeywordExtractor(lan="en", top=top_n)
        keywords_raw = extractor.extract_keywords(text)
        keywords = [{"keyword": kw, "score": score} for kw, score in keywords_raw]
        return KeywordMetrics(keywords=keywords)

    def compute_engagement(self, speech: SpeechMetrics, summary: SummaryMetrics,
                           avg_pause: float = None) -> EngagementMetrics:
        wpm = speech.words_per_minute
        rate_score = min(1.0, wpm / 200.0) if wpm < 200 else max(0.0, 1.0 - (wpm - 200) / 200.0)
        rate_score = max(0.0, min(1.0, rate_score))
        vocab = summary.vocabulary_richness
        vocab_score = min(1.0, vocab / 0.6)
        flesch = self.compute_readability().flesch_reading_ease
        readability_score = max(0.0, min(1.0, flesch / 100.0))
        filler_ratio = speech.filler_word_count / speech.total_words if speech.total_words > 0 else 0.0
        filler_score = max(0.0, 1.0 - (filler_ratio / 0.05))
        pause = avg_pause if avg_pause is not None else speech.average_pause_length
        pause_score = max(0.0, 1.0 - (pause / 2.0))
        weights = {"rate": 0.25, "vocab": 0.20, "readability": 0.20, "filler": 0.20, "pause": 0.15}
        total = (weights["rate"] * rate_score + weights["vocab"] * vocab_score +
                 weights["readability"] * readability_score + weights["filler"] * filler_score +
                 weights["pause"] * pause_score)
        return EngagementMetrics(score=total * 100)

    def compute_all(self) -> MetricsResult:
        summary = self.compute_summary()
        speech = self.compute_speech()
        readability = self.compute_readability()
        keywords = self.compute_keywords()
        engagement = self.compute_engagement(speech, summary)
        return MetricsResult(
            module="metrics_layer", version="0.1.0",
            summary=summary, speech=speech, readability=readability,
            keywords=keywords, engagement=engagement,
        )

    # ---------- Per‑segment metrics ----------
    def compute_segment_metrics(self) -> List[Dict[str, Any]]:
        """Compute metrics for each segment and return list of dicts."""
        segment_metrics = []
        n = len(self.segments)
        for idx, seg in enumerate(self.segments):
            seg_words = self.segment_words[idx]
            text = self.segment_texts[idx]

            # Basic
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            seg_duration = seg_end - seg_start if seg_end > seg_start else 0.0
            word_count = len(seg_words)
            duration_min = seg_duration / 60.0 if seg_duration > 0 else 1.0
            wpm = word_count / duration_min if duration_min > 0 else 0.0

            # Filler count
            filler_count = sum(1 for w in seg_words if w["word"].lower() in self.FILLER_WORDS)

            # Vocabulary richness
            unique_words = set(w["word"].lower() for w in seg_words)
            vocab_richness = len(unique_words) / word_count if word_count > 0 else 0.0

            # Readability
            if text.strip():
                readability = self.compute_readability(text)
                reading_ease = readability.flesch_reading_ease
                reading_grade = readability.flesch_kincaid_grade
            else:
                reading_ease = 0.0
                reading_grade = 0.0

            # Keywords (top 5)
            if text.strip():
                kw = self.compute_keywords(text, top_n=5)
                keywords = [k["keyword"] for k in kw.keywords]
            else:
                keywords = []

            # Pauses before/after
            if idx == 0:
                pause_before = 0.0
            else:
                prev_end = self.segments[idx-1].get("end", 0.0)
                pause_before = seg_start - prev_end if seg_start > prev_end else 0.0
            if idx == n - 1:
                pause_after = 0.0
            else:
                next_start = self.segments[idx+1].get("start", 0.0)
                pause_after = next_start - seg_end if next_start > seg_end else 0.0

            # Speech density
            if seg_words:
                total_word_duration = sum(w["end"] - w["start"] for w in seg_words)
                speech_density = total_word_duration / seg_duration if seg_duration > 0 else 0.0
            else:
                speech_density = 0.0

            # Average internal pause
            if len(seg_words) > 1:
                internal_pauses = [seg_words[i+1]["start"] - seg_words[i]["end"] for i in range(len(seg_words)-1) if seg_words[i+1]["start"] > seg_words[i]["end"]]
                avg_internal_pause = statistics.mean(internal_pauses) if internal_pauses else 0.0
            else:
                avg_internal_pause = 0.0

            # Engagement for this segment
            seg_summary = SummaryMetrics(
                video_duration=seg_duration,
                num_scenes=1,
                average_scene_duration=seg_duration,
                vocabulary_richness=vocab_richness,
                average_confidence=1.0
            )
            seg_speech = SpeechMetrics(
                total_words=word_count,
                words_per_minute=wpm,
                speaking_time=seg_duration,
                silence_time=0.0,
                speech_ratio=1.0,
                average_sentence_length=word_count / max(1, len(sent_tokenize(text))),
                filler_word_count=filler_count,
                longest_monologue=seg_duration,
                speakers_count=1,
                speaker_speaking_time={"": seg_duration},
                average_pause_length=avg_internal_pause,
                longest_pause=avg_internal_pause,
            )
            engagement = self.compute_engagement(seg_speech, seg_summary, avg_pause=avg_internal_pause)

            # Extract speaker, person, and scene
            speaker = seg.get("speaker_label") or seg.get("speaker", "UNKNOWN")
            person = seg.get("person")  # canonical person ID from timeline fusion
            scene = seg.get("scene_id")

            segment_metrics.append({
                "segment_id": idx + 1,
                "start": seg_start,
                "end": seg_end,
                "duration": seg_duration,
                "speaker": speaker,
                "person": person,
                "scene": scene,
                "word_count": word_count,
                "words_per_minute": wpm,
                "filler_count": filler_count,
                "vocabulary_richness": vocab_richness,
                "reading_grade": reading_grade,
                "reading_ease": reading_ease,
                "keywords": keywords,
                "pause_before": pause_before,
                "pause_after": pause_after,
                "speech_density": speech_density,
                "engagement_score": engagement.score,
            })
        return segment_metrics