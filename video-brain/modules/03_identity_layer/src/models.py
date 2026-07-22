"""Pydantic models for identity tracking output."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class Person(BaseModel):
    """A persistent person identity."""

    person_id: str = Field(..., description="Unique identifier (person_1, person_2, ...)")
    first_seen: str = Field(..., description="Timestamp of first appearance")
    last_seen: str = Field(..., description="Timestamp of last appearance")
    appearances: int = Field(..., description="Number of scenes where person appears")

    model_config = ConfigDict(frozen=True)

class ScenePerson(BaseModel):
    """A person appearance within a specific scene."""

    person_id: str = Field(..., description="Person identifier")
    bbox: List[float] = Field(..., description="Bounding box [x, y, width, height]")
    confidence: float = Field(..., description="Detection confidence")
    embedding_similarity: float = Field(..., description="Cosine similarity to cluster centroid")

    model_config = ConfigDict(frozen=True)

class SceneIdentity(BaseModel):
    """Identity info for a single scene."""

    id: int = Field(..., description="Scene ID (1-based)")
    start: str = Field(..., description="Start timestamp in HH:MM:SS.mmm format")
    end: str = Field(..., description="End timestamp in HH:MM:SS.mmm format")
    people: List[ScenePerson] = Field(..., description="People present in this scene")

    model_config = ConfigDict(frozen=True)

class IdentityResult(BaseModel):
    """Complete result of identity tracking."""

    video: str = Field(..., description="Path to the input video file")
    people: List[Person] = Field(..., description="All distinct people in the video")
    scenes: List[SceneIdentity] = Field(..., description="Per-scene identity info")

    model_config = ConfigDict(frozen=True)