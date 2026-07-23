"""AI Planner: uses an LLM to select the best clips."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .models import FinalClipPlan, SelectedClip
from .prompt import SYSTEM_PROMPT_TEMPLATE
from .providers import BaseLLMProvider, DeepSeekProvider

logger = logging.getLogger(__name__)


class Planner:
    """Orchestrates the AI planning process."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or DeepSeekProvider()

    def load_json(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def prepare_context(
        self,
        candidate_clips_path: str,
        timeline_path: str,
        people_path: str,
        metrics_path: str,
        segment_metrics_path: str,
    ) -> Dict[str, Any]:
        """Load all inputs and prepare a summary for the prompt."""
        clips = self.load_json(candidate_clips_path)
        timeline = self.load_json(timeline_path)
        people = self.load_json(people_path) if Path(people_path).exists() else {}
        metrics = self.load_json(metrics_path) if Path(metrics_path).exists() else {}
        segment_metrics = self.load_json(segment_metrics_path) if Path(segment_metrics_path).exists() else {}

        # Truncate large lists for context window
        clips_summary = clips.get("candidates", [])[:20]
        timeline_summary = timeline.get("segments", [])[:10]
        seg_summary = segment_metrics[:10] if isinstance(segment_metrics, list) else []

        context = {
            "clips": json.dumps(clips_summary, indent=2),
            "timeline": json.dumps(timeline_summary, indent=2),
            "people": json.dumps(people.get("people", []), indent=2),
            "metrics": json.dumps(metrics, indent=2),
            "segment_metrics": json.dumps(seg_summary, indent=2),
        }
        return context

    def plan(
        self,
        candidate_clips_path: str,
        timeline_path: str,
        people_path: str,
        metrics_path: str,
        segment_metrics_path: str,
        video_path: Optional[str] = None,
    ) -> FinalClipPlan:
        """Run the planner and return the final plan."""
        context = self.prepare_context(
            candidate_clips_path,
            timeline_path,
            people_path,
            metrics_path,
            segment_metrics_path,
        )

        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            clips=context["clips"],
            timeline=context["timeline"],
            people=context["people"],
            metrics=context["metrics"],
            segment_metrics=context["segment_metrics"],
        )

        logger.info("Sending prompt to LLM...")
        response = self.provider.generate(prompt, system_prompt=None)
        logger.info("Received response from LLM")

        # Parse the JSON response
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
        selected_clips = []
        for item in data.get("selected_clips", []):
            selected_clips.append(
                SelectedClip(
                    clip_id=item["clip_id"],
                    reason=item.get("reason", ""),
                    score=item.get("score", 0.0),
                )
            )

        result = FinalClipPlan(
            module="ai_planner",
            version="0.1.0",
            video=video_path or "",
            selected_clips=selected_clips,
            reasoning=data.get("reasoning", ""),
        )
        return result