"""Pydantic models for AI edit planner output."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class ZoomEvent(BaseModel):
    """A zoom event within a clip."""

    time: str = Field(..., description="Timestamp (HH:MM:SS.mmm) or seconds")
    type: str = Field(..., description="Zoom type: Punch Zoom, Slow Zoom, etc.")
    strength: Optional[str] = Field(None, description="Strength: Low, Medium, High")

    model_config = ConfigDict(frozen=True)

class SoundEffect(BaseModel):
    """A sound effect to place at a specific time."""

    time: str = Field(..., description="Timestamp")
    effect: str = Field(..., description="Effect name: Whoosh, Impact, etc.")

    model_config = ConfigDict(frozen=True)

class BrollSuggestion(BaseModel):
    """A B-roll suggestion with timestamp and search query."""

    time: str = Field(..., description="Timestamp")
    query: str = Field(..., description="Search query for B-roll footage")

    model_config = ConfigDict(frozen=True)

class CaptionStyle(BaseModel):
    """Caption styling options."""

    font: str = Field(..., description="Font style: Bold, Sans, etc.")
    animation: str = Field(..., description="Animation: Pop, Fade, Typewriter, etc.")

    model_config = ConfigDict(frozen=True)

class ClipEditInstructions(BaseModel):
    """Editing instructions for a single clip."""

    clip_id: int = Field(..., description="ID from candidate_clips.json")
    editing_style: str = Field(..., description="Overall style: Fast Motivational, Storytelling, etc.")
    subtitle_style: str = Field(..., description="Subtitle approach: Word By Word, Per Sentence, etc.")
    captions: CaptionStyle = Field(..., description="Caption styling")
    zoom_events: List[ZoomEvent] = Field(default_factory=list, description="Zoom events")
    highlight_words: List[str] = Field(default_factory=list, description="Words to highlight")
    sound_effects: List[SoundEffect] = Field(default_factory=list, description="Sound effects")
    broll: List[BrollSuggestion] = Field(default_factory=list, description="B-roll suggestions")
    music: str = Field(..., description="Music style: Motivational, Cinematic, etc.")
    transition: str = Field(..., description="Transition: Hard Cut, Crossfade, etc.")
    ending: str = Field(..., description="Ending style: Fade, CTA Overlay, etc.")
    cta: str = Field(..., description="Call to action text")
    confidence: float = Field(..., description="Confidence in these instructions (0-1)")

    model_config = ConfigDict(frozen=True)

class EditPlanResult(BaseModel):
    """Complete edit plan result."""

    module: str = Field("edit_planner", description="Module name")
    version: str = Field("0.1.0", description="Module version")
    video: str = Field(..., description="Path to input video")
    clips: List[ClipEditInstructions] = Field(..., description="Editing instructions per clip")

    model_config = ConfigDict(frozen=True)