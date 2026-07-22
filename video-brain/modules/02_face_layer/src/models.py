"""Pydantic models for face detection output."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class Face(BaseModel):
    """A single detected face."""

    face_id: int = Field(..., description="Face index within the scene (1-based)")
    bbox: List[float] = Field(..., description="Bounding box [x, y, width, height]")
    confidence: float = Field(..., description="Detection confidence score")

    model_config = ConfigDict(frozen=True)

class SceneFaces(BaseModel):
    """Faces detected within a single scene."""

    scene_id: int = Field(..., description="Scene ID (1-based)")
    faces: List[Face] = Field(..., description="List of detected faces")

    model_config = ConfigDict(frozen=True)

class FaceDetectionResult(BaseModel):
    """Complete result of face detection on a video."""

    module: str = Field("face_layer", description="Module name")
    version: str = Field("0.1.0", description="Module version")
    video: str = Field(..., description="Path to the input video file")
    scenes: List[SceneFaces] = Field(..., description="Faces per scene")

    model_config = ConfigDict(frozen=True)