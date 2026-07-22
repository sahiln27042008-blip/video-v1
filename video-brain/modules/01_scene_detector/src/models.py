"""Pydantic models for scene detection output."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import timedelta

class Scene(BaseModel):
    """Represents a single detected scene."""

    id: int = Field(..., description="Scene index (1-based)")
    start: str = Field(..., description="Start timestamp in HH:MM:SS.mmm format")
    end: str = Field(..., description="End timestamp in HH:MM:SS.mmm format")
    duration: float = Field(..., description="Duration in seconds")

    model_config = ConfigDict(frozen=True)

class SceneDetectionResult(BaseModel):
    """Complete result of scene detection on a video."""

    video: str = Field(..., description="Path to the input video file")
    duration: float = Field(..., description="Total duration in seconds")
    fps: float = Field(..., description="Frames per second of the video")
    scene_count: int = Field(..., description="Number of detected scenes")
    scenes: List[Scene] = Field(..., description="List of detected scenes")

    model_config = ConfigDict(frozen=True)

def format_timedelta(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"