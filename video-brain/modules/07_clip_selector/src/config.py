"""Configuration for clip selector."""

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ClipSelectorConfig:
    """All configurable parameters for the clip selector."""

    # Scoring weights (must sum to 1.0)
    weight_engagement: float = 0.35
    weight_keyword: float = 0.20
    weight_speech_density: float = 0.15
    weight_vocab_richness: float = 0.10
    weight_duration: float = 0.10
    weight_readability: float = 0.10

    # Penalties
    filler_penalty_factor: float = 0.02  # penalty per filler word (percentage of total words)
    short_clip_penalty: float = 0.1      # penalty if below min duration

    # Duration limits
    min_duration_seconds: float = 2.0
    max_duration_seconds: float = 60.0

    # Merging
    merge_gap_seconds: float = 1.0  # max gap between segments to merge
    same_person_required: bool = True
    same_speaker_required: bool = True

    # Output
    top_k: int = 20  # max number of candidates to return

    # Keyword extraction
    max_keywords_per_clip: int = 5

    # Engagement scaling (used to normalise segment engagement to 0-1)
    engagement_lower_bound: float = 0.0
    engagement_upper_bound: float = 100.0

    @property
    def weights_dict(self) -> Dict[str, float]:
        """Return weights as a dict for validation."""
        return {
            "engagement": self.weight_engagement,
            "keyword": self.weight_keyword,
            "speech_density": self.weight_speech_density,
            "vocab_richness": self.weight_vocab_richness,
            "duration": self.weight_duration,
            "readability": self.weight_readability,
        }

    def validate(self) -> None:
        """Ensure weights sum to 1.0."""
        total = sum(self.weights_dict.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights sum to {total}, expected 1.0")