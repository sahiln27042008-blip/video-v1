"""Pydantic models for renderer configuration and results."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, Discriminator, Tag

# ---------- Operation models (for technical_edit_plan.json) ----------
class TrimOp(BaseModel):
    type: str = "trim"
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")

    model_config = ConfigDict(frozen=True)

class SubtitleOp(BaseModel):
    type: str = "subtitle"
    style: str = Field(..., description="Subtitle style")
    font: str = Field(..., description="Font name")
    size: int = Field(..., description="Font size")
    weight: int = Field(..., description="Font weight (100-900)")
    stroke: float = Field(..., description="Stroke width")
    shadow: int = Field(..., description="Shadow size")
    alignment: str = Field(..., description="Alignment: bottom_center, center, etc.")
    animation_type: str = Field(..., description="Animation: word_by_word, pop, fade, etc.")
    highlight_color: str = Field(..., description="Color for highlighted words")
    normal_color: str = Field(..., description="Color for normal text")
    safe_margin: float = Field(..., description="Margin as fraction of screen")
    word_timing: List[Dict[str, Any]] = Field(..., description="List of {word, start, end}")
    highlight_words: Optional[List[Dict[str, str]]] = Field(None, description="List of {word, color}")

    model_config = ConfigDict(frozen=True)

class ZoomOp(BaseModel):
    type: str = "zoom"
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    parameters: Dict[str, Any] = Field(..., description="scale, anchor, easing")

    model_config = ConfigDict(frozen=True)

class MusicOp(BaseModel):
    type: str = "music"
    style: str = Field(..., description="Music style")
    volume: float = Field(..., description="Volume in dB (negative)")
    ducking: bool = Field(..., description="Whether to duck under speech")
    fade_in: float = Field(..., description="Fade in duration in seconds")
    fade_out: float = Field(..., description="Fade out duration in seconds")

    model_config = ConfigDict(frozen=True)

class SoundEffectOp(BaseModel):
    type: str = "sound_effect"
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    parameters: Dict[str, Any] = Field(..., description="type, intensity, etc.")

    model_config = ConfigDict(frozen=True)

class TransitionOp(BaseModel):
    type: str = "transition"
    style: str = Field(..., description="hard_cut, crossfade, etc.")
    duration: Optional[float] = Field(None, description="Duration of transition in seconds")

    model_config = ConfigDict(frozen=True)

# Discriminated union for operations
Operation = Union[
    TrimOp,
    SubtitleOp,
    ZoomOp,
    MusicOp,
    SoundEffectOp,
    TransitionOp,
]

class Clip(BaseModel):
    clip_id: int = Field(..., description="Clip ID")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")
    operations: List[Operation] = Field(..., description="List of operations")

    model_config = ConfigDict(frozen=True)

class TechnicalEditPlan(BaseModel):
    module: str = Field(..., description="Module name")
    version: str = Field(..., description="Version")
    clips: List[Clip] = Field(..., description="List of clips with operations")

    model_config = ConfigDict(frozen=True)

# ---------- Old models (backward compatibility) ----------
class ClipEdit(BaseModel):
    clip_id: int
    editing_style: str
    subtitle_style: str
    captions: Dict[str, Any]
    zoom_events: List[Dict[str, Any]] = Field(default_factory=list)
    highlight_words: List[str] = Field(default_factory=list)
    sound_effects: List[Dict[str, Any]] = Field(default_factory=list)
    broll: List[Dict[str, Any]] = Field(default_factory=list)
    music: str
    transition: str
    ending: str
    cta: str
    confidence: float

    model_config = ConfigDict(frozen=True)

class CandidateClip(BaseModel):
    candidate_id: int
    start: str
    end: str
    duration: float
    person: Optional[str] = None
    speaker: str
    segments: List[int]
    score: float
    keywords: List[str] = Field(default_factory=list)
    transcript: str
    reason: Dict[str, Any]

    model_config = ConfigDict(frozen=True)

class RenderConfig(BaseModel):
    video_path: str
    editor_plan_path: str
    candidate_clips_path: Optional[str] = None
    timeline_words_path: Optional[str] = None
    output_dir: str
    background_music_path: Optional[str] = None
    temp_dir: str = "./temp_render"

    model_config = ConfigDict(frozen=True)

class RenderResult(BaseModel):
    success: bool
    output_video: Optional[str] = None
    output_subtitles: Optional[str] = None
    log: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    model_config = ConfigDict(frozen=True)