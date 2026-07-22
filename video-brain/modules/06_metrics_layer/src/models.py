"""Pydantic models for metrics output."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

class SpeechMetrics(BaseModel):
    """Metrics related to speech and silence."""

    total_words: int = Field(..., description="Total number of words")
    words_per_minute: float = Field(..., description="Words per minute")
    speaking_time: float = Field(..., description="Total speaking time in seconds")
    silence_time: float = Field(..., description="Total silence time in seconds")
    speech_ratio: float = Field(..., description="Speaking time / total duration")
    average_sentence_length: float = Field(..., description="Average words per sentence")
    filler_word_count: int = Field(..., description="Count of filler words (uh, um, like, etc.)")
    longest_monologue: float = Field(..., description="Longest continuous speaking segment in seconds")
    speakers_count: int = Field(..., description="Number of distinct speakers")
    speaker_speaking_time: Dict[str, float] = Field(..., description="Speaking time per speaker in seconds")
    average_pause_length: float = Field(..., description="Average pause between words in seconds")
    longest_pause: float = Field(..., description="Longest pause in seconds")

    model_config = ConfigDict(frozen=True)

class ReadabilityMetrics(BaseModel):
    """Readability scores."""

    flesch_reading_ease: float = Field(..., description="Flesch Reading Ease score")
    flesch_kincaid_grade: float = Field(..., description="Flesch-Kincaid Grade Level")

    model_config = ConfigDict(frozen=True)

class KeywordMetrics(BaseModel):
    """Keyword extraction results."""

    keywords: List[Dict[str, Any]] = Field(..., description="Top keywords with scores")

    model_config = ConfigDict(frozen=True)

class EngagementMetrics(BaseModel):
    """Estimated engagement score (0-100)."""

    score: float = Field(..., description="Engagement score (0-100)")

    model_config = ConfigDict(frozen=True)

class SummaryMetrics(BaseModel):
    """High-level summary metrics."""

    video_duration: float = Field(..., description="Total video duration in seconds")
    num_scenes: int = Field(..., description="Number of scenes")
    average_scene_duration: float = Field(..., description="Average scene duration in seconds")
    vocabulary_richness: float = Field(..., description="Unique words / total words")
    average_confidence: float = Field(..., description="Average word confidence")

    model_config = ConfigDict(frozen=True)

class MetricsResult(BaseModel):
    """Complete metrics result."""

    module: str = Field("metrics_layer", description="Module name")
    version: str = Field("0.1.0", description="Module version")
    summary: SummaryMetrics
    speech: SpeechMetrics
    readability: ReadabilityMetrics
    keywords: KeywordMetrics
    engagement: EngagementMetrics

    model_config = ConfigDict(frozen=True)