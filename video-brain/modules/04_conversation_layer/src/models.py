"""Pydantic models for conversation output."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class Word(BaseModel):
    """A single word with timestamp."""
    word: str = Field(..., description="The word text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")

    model_config = ConfigDict(frozen=True)

class ConversationSegment(BaseModel):
    """A single segment of conversation with speaker and text."""

    start: str = Field(..., description="Segment start timestamp (HH:MM:SS.mmm)")
    end: str = Field(..., description="Segment end timestamp")
    speaker: str = Field(..., description="Speaker label (SPEAKER_00, SPEAKER_01, ...)")
    text: str = Field(..., description="Transcribed text")
    words: List[Word] = Field(default_factory=list, description="Word-level timestamps")

    model_config = ConfigDict(frozen=True)

class ConversationResult(BaseModel):
    """Complete conversation result."""

    module: str = Field("conversation_layer", description="Module name")
    version: str = Field("0.2.0", description="Module version (WhisperX)")
    video: str = Field(..., description="Path to input video")
    segments: List[ConversationSegment] = Field(..., description="All conversation segments")

    model_config = ConfigDict(frozen=True)