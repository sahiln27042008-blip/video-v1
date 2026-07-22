"""Pydantic models for timeline output."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class Word(BaseModel):
    """A single word with timestamp."""
    word: str
    start: float
    end: float

    model_config = ConfigDict(frozen=True)

class TimelineSegment(BaseModel):
    """A single segment in the fused timeline."""

    start: str = Field(..., description="Segment start timestamp (HH:MM:SS.mmm)")
    end: str = Field(..., description="Segment end timestamp")
    scene_id: Optional[int] = Field(None, description="Scene ID (1-based) if found")
    camera: Optional[str] = Field(None, description="Camera label (e.g., 'guest', 'host', null if unknown)")
    visible_people: List[str] = Field(default_factory=list, description="List of person_ids visible during this segment")
    active_speaker: Optional[str] = Field(None, description="person_id of active speaker, if linked")
    person: Optional[str] = Field(None, description="Canonical person ID assigned after speaker-person mapping")
    speaker_label: str = Field(..., description="Original speaker label from diarization (SPEAKER_00)")
    text: str = Field(..., description="Transcribed text")
    words: List[Word] = Field(default_factory=list, description="Word-level timestamps")

    model_config = ConfigDict(frozen=True)

class TimelineResult(BaseModel):
    """Complete fused timeline."""

    module: str = Field("timeline_fusion", description="Module name")
    version: str = Field("0.1.0", description="Module version")
    video: str = Field(..., description="Path to input video")
    segments: List[TimelineSegment] = Field(..., description="Fused timeline segments")

    model_config = ConfigDict(frozen=True)

class PersonInfo(BaseModel):
    """Canonical person with associated face IDs, speaker, and segments."""

    person_id: str = Field(..., description="Canonical person ID (person_1, person_2, ...)")
    speaker: Optional[str] = Field(None, description="Assigned speaker label if mapped")
    face_ids: List[int] = Field(..., description="List of face IDs from face detection belonging to this person")
    segments: List[int] = Field(..., description="List of timeline segment IDs where this person appears")

    model_config = ConfigDict(frozen=True)

class PeopleResult(BaseModel):
    """List of all canonical persons in the video."""

    people: List[PersonInfo] = Field(..., description="All persons")

    model_config = ConfigDict(frozen=True)