"""Candidate clip generator module for Video Brain."""
from .selector import ClipSelector
from .models import CandidateClip, CandidateClipsResult

__all__ = ["ClipSelector", "CandidateClip", "CandidateClipsResult"]