"""Scoring functions for candidate clips."""

import math
from typing import List, Dict, Any, Optional
from .config import ClipSelectorConfig

def compute_keyword_score(segment: Dict[str, Any], config: ClipSelectorConfig) -> float:
    """Compute keyword score from segment metrics."""
    # segment_metrics contains a list of keywords with scores
    # We'll compute a score based on the presence of keywords and their scores
    # Simple version: use the average score of top keywords
    keywords = segment.get("keywords", [])
    if not keywords:
        return 0.0
    # If keywords are strings (just the keyword text), we can't compute a numeric score
    # If they are dicts with "score", use that
    if isinstance(keywords[0], dict):
        scores = [k.get("score", 0.0) for k in keywords[:config.max_keywords_per_clip]]
        return min(100.0, sum(scores) / len(scores) * 100.0 if scores else 0.0)
    else:
        # Fallback: just count keywords as signal
        return min(100.0, len(keywords) / config.max_keywords_per_clip * 100.0)

def compute_duration_score(duration: float, config: ClipSelectorConfig) -> float:
    """Compute duration score based on ideal clip length."""
    # Ideal duration range: 5-15 seconds
    ideal_min = 5.0
    ideal_max = 15.0
    if duration < config.min_duration_seconds:
        return 0.0
    if duration > config.max_duration_seconds:
        return 0.0
    if duration < ideal_min:
        # Short clip: score from 0 to 100 based on distance from ideal_min
        return (duration - config.min_duration_seconds) / (ideal_min - config.min_duration_seconds) * 100.0
    if duration <= ideal_max:
        return 100.0
    # Longer than ideal: score drops gradually
    return max(0.0, 100.0 * (1 - (duration - ideal_max) / (config.max_duration_seconds - ideal_max)))

def compute_readability_score(segment: Dict[str, Any], config: ClipSelectorConfig) -> float:
    """Compute readability score from segment metrics."""
    reading_ease = segment.get("reading_ease", 0.0)
    # Flesch reading ease: 0-100, higher is better
    return max(0.0, min(100.0, reading_ease))

def compute_vocab_richness_score(segment: Dict[str, Any], config: ClipSelectorConfig) -> float:
    """Compute vocabulary richness score."""
    richness = segment.get("vocabulary_richness", 0.0)
    # 0-1 scale, multiply by 100
    return min(100.0, richness * 100.0)

def compute_filler_penalty(segment: Dict[str, Any], config: ClipSelectorConfig) -> float:
    """Compute penalty based on filler word ratio."""
    word_count = segment.get("word_count", 1)
    filler_count = segment.get("filler_count", 0)
    if word_count == 0:
        return 0.0
    filler_ratio = filler_count / word_count
    # Penalty scales with filler ratio, up to a maximum penalty (e.g., 20 points)
    penalty = min(20.0, filler_ratio * 100.0 * config.filler_penalty_factor)
    return penalty

def compute_speech_density_score(segment: Dict[str, Any], config: ClipSelectorConfig) -> float:
    """Compute speech density score."""
    density = segment.get("speech_density", 0.0)
    # density is between 0 and 1; multiply by 100
    return min(100.0, density * 100.0)

def compute_engagement_score(segment: Dict[str, Any], config: ClipSelectorConfig) -> float:
    """Compute engagement score from segment metrics."""
    engagement = segment.get("engagement_score", 0.0)
    # Already 0-100
    return max(0.0, min(100.0, engagement))

def compute_individual_scores(segment: Dict[str, Any], config: ClipSelectorConfig) -> Dict[str, float]:
    """Compute all sub‑scores for a segment."""
    engagement = compute_engagement_score(segment, config)
    keyword = compute_keyword_score(segment, config)
    speech_density = compute_speech_density_score(segment, config)
    vocab_richness = compute_vocab_richness_score(segment, config)
    duration = compute_duration_score(segment.get("duration", 0.0), config)
    readability = compute_readability_score(segment, config)
    filler_penalty = compute_filler_penalty(segment, config)

    return {
        "engagement": engagement,
        "keyword": keyword,
        "speech_density": speech_density,
        "vocab_richness": vocab_richness,
        "duration": duration,
        "readability": readability,
        "filler_penalty": filler_penalty,
    }

def compute_final_score(scores: Dict[str, float], config: ClipSelectorConfig) -> float:
    """Combine individual scores with weights and penalties."""
    weights = config.weights_dict
    raw_score = (
        weights["engagement"] * scores["engagement"] +
        weights["keyword"] * scores["keyword"] +
        weights["speech_density"] * scores["speech_density"] +
        weights["vocab_richness"] * scores["vocab_richness"] +
        weights["duration"] * scores["duration"] +
        weights["readability"] * scores["readability"]
    )
    # Apply filler penalty
    penalty = scores.get("filler_penalty", 0.0)
    final_score = max(0.0, raw_score - penalty)
    return final_score

def score_segment(segment: Dict[str, Any], config: ClipSelectorConfig) -> Dict[str, Any]:
    """Compute full scoring for a single segment."""
    sub_scores = compute_individual_scores(segment, config)
    final = compute_final_score(sub_scores, config)
    return {
        "score": final,
        "sub_scores": sub_scores,
    }