"""Pydantic models for candidate clip output."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class CandidateClipReason(BaseModel):
    """Breakdown of sub‑scores for a candidate clip."""

    engagement: float = Field(..., description="Normalized engagement score (0-100)")
    speech_density: float = Field(..., description="Normalized speech density score (0-100)")
    keywords: float = Field(..., description="Normalized keyword score (0-100)")
    duration: float = Field(..., description="Normalized duration score (0-100)")
    readability: float = Field(..., description="Normalized readability score (0-100)")
    vocabulary_richness: Optional[float] = Field(None, description="Vocabulary richness score (0-100)")
    filler_penalty: Optional[float] = Field(None, description="Penalty applied for fillers (0-100)")

    model_config = ConfigDict(frozen=True)

class CandidateClip(BaseModel):
    """A single candidate clip generated from merged segments."""

    candidate_id: int = Field(..., description="Unique ID (1-based)")
    start: str = Field(..., description="Start timestamp (HH:MM:SS.mmm)")
    end: str = Field(..., description="End timestamp (HH:MM:SS.mmm)")
    duration: float = Field(..., description="Duration in seconds")
    person: Optional[str] = Field(None, description="Canonical person ID if assigned")
    speaker: str = Field(..., description="Speaker label (SPEAKER_00)")
    segments: List[int] = Field(..., description="List of original segment IDs merged into this clip")
    score: float = Field(..., description="Overall score (0-100)")
    keywords: List[str] = Field(default_factory=list, description="Top keywords for this clip")
    transcript: str = Field(..., description="Merged transcript text")
    reason: CandidateClipReason = Field(..., description="Breakdown of sub‑scores")

    model_config = ConfigDict(frozen=True)

class CandidateClipsResult(BaseModel):
    """Complete result of candidate clip generation."""

    module: str = Field("clip_selector", description="Module name")
    version: str = Field("0.1.0", description="Module version")
    video: str = Field(..., description="Path to input video")
    candidates: List[CandidateClip] = Field(..., description="Sorted candidate clips")

    model_config = ConfigDict(frozen=True)