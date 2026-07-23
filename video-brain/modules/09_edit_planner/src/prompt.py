"""Load system prompt from editor.md."""

import os
from pathlib import Path

PROMPT_FILE = Path(__file__).parent.parent / "editor.md"

def load_system_prompt() -> str:
    """Load the system prompt from editor.md."""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt file not found: {PROMPT_FILE}")
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def get_user_prompt(
    final_clip_plan: dict,
    timeline: dict,
    people: dict,
    metrics: dict,
    segment_metrics: dict,
    candidate_clips: dict,
) -> str:
    """Build the user prompt from loaded data."""
    # Truncate large lists
    clips_summary = final_clip_plan.get("selected_clips", [])
    timeline_summary = timeline.get("segments", [])[:20]
    people_summary = people.get("people", [])
    metrics_summary = metrics
    seg_metrics_summary = segment_metrics[:20] if isinstance(segment_metrics, list) else []

    import json
    context = {
        "final_clip_plan": json.dumps(clips_summary, indent=2),
        "timeline": json.dumps(timeline_summary, indent=2),
        "people": json.dumps(people_summary, indent=2),
        "metrics": json.dumps(metrics_summary, indent=2),
        "segment_metrics": json.dumps(seg_metrics_summary, indent=2),
        "candidate_clips": json.dumps(candidate_clips.get("candidates", [])[:20], indent=2),
    }

    prompt = f"""You are the Video Brain Editor. Using the following data, generate detailed editing instructions for each selected clip.

Here is the data:

---
Final Clip Plan (from Creative Director):
{context['final_clip_plan']}

---
Full Timeline (first 20 segments):
{context['timeline']}

---
People:
{context['people']}

---
Global Metrics:
{context['metrics']}

---
Segment Metrics (first 20):
{context['segment_metrics']}

---
Candidate Clips (first 20):
{context['candidate_clips']}

---

Now produce the edit plan. Output only valid JSON that matches the expected schema. Do not include any extra text."""
    return prompt