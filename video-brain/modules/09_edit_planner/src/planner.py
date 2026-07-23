"""AI Edit Planner: uses an LLM to generate editing instructions."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .models import EditPlanResult, ClipEditInstructions, CaptionStyle, ZoomEvent, SoundEffect, BrollSuggestion
from .prompt import load_system_prompt, get_user_prompt
from .providers import BaseLLMProvider, DeepSeekProvider

logger = logging.getLogger(__name__)


class EditPlanner:
    """Orchestrates the AI edit planning process."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or DeepSeekProvider()

    def load_json(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def plan(
        self,
        final_clip_plan_path: str,
        timeline_path: str,
        people_path: str,
        metrics_path: str,
        segment_metrics_path: str,
        candidate_clips_path: str,
        video_path: Optional[str] = None,
    ) -> EditPlanResult:
        """Run the edit planner and return the edit plan."""
        # Load all inputs
        final_clip_plan = self.load_json(final_clip_plan_path)
        timeline = self.load_json(timeline_path)
        people = self.load_json(people_path) if Path(people_path).exists() else {}
        metrics = self.load_json(metrics_path) if Path(metrics_path).exists() else {}
        segment_metrics = self.load_json(segment_metrics_path) if Path(segment_metrics_path).exists() else {}
        candidate_clips = self.load_json(candidate_clips_path) if Path(candidate_clips_path).exists() else {}

        # Build user prompt
        user_prompt = get_user_prompt(
            final_clip_plan,
            timeline,
            people,
            metrics,
            segment_metrics,
            candidate_clips,
        )

        # Load system prompt
        system_prompt = load_system_prompt()

        logger.info("Sending prompt to LLM...")
        response = self.provider.generate(user_prompt, system_prompt=system_prompt, temperature=0.7, max_tokens=3000)
        logger.info("Received response from LLM")

        # Parse JSON
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")

        # Validate structure
        clips_data = data.get("clips", [])
        clip_instructions = []
        for item in clips_data:
            # Construct nested objects
            captions = CaptionStyle(
                font=item["captions"].get("font", "Bold"),
                animation=item["captions"].get("animation", "Pop"),
            )
            zoom_events = [ZoomEvent(**z) for z in item.get("zoom_events", [])]
            sound_effects = [SoundEffect(**s) for s in item.get("sound_effects", [])]
            broll = [BrollSuggestion(**b) for b in item.get("broll", [])]

            clip = ClipEditInstructions(
                clip_id=item["clip_id"],
                editing_style=item["editing_style"],
                subtitle_style=item["subtitle_style"],
                captions=captions,
                zoom_events=zoom_events,
                highlight_words=item.get("highlight_words", []),
                sound_effects=sound_effects,
                broll=broll,
                music=item["music"],
                transition=item["transition"],
                ending=item["ending"],
                cta=item["cta"],
                confidence=item["confidence"],
            )
            clip_instructions.append(clip)

        result = EditPlanResult(
            module="edit_planner",
            version="0.1.0",
            video=video_path or "",
            clips=clip_instructions,
        )
        return result