"""Pydantic models for AI planner output."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class SelectedClip(BaseModel):
    """A clip selected by the LLM for the final edit."""

    clip_id: int = Field(..., description="ID from candidate_clips.json")
    reason: str = Field(..., description="Why this clip was selected")
    score: float = Field(..., description="LLM-assigned score (0-100)")

    model_config = ConfigDict(frozen=True)

class FinalClipPlan(BaseModel):
    """Complete plan produced by the AI planner."""

    module: str = Field("ai_planner", description="Module name")
    version: str = Field("0.1.0", description="Module version")
    video: str = Field(..., description="Path to input video")
    selected_clips: List[SelectedClip] = Field(..., description="Clips chosen for the final edit")
    reasoning: str = Field(..., description="Overall reasoning from the LLM about the selection")

    model_config = ConfigDict(frozen=True)