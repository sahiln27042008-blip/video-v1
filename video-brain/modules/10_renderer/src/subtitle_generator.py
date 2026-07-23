"""Subtitle generation for renderer (supports both timeline-based and word_timing from plan)."""

import os
import json
from typing import List, Dict, Any, Optional

def parse_time(time_str: str) -> float:
    if isinstance(time_str, (int, float)):
        return float(time_str)
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    else:
        return float(time_str)

def format_time_ass(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def generate_ass_from_word_timing(
    word_timing: List[Dict[str, Any]],
    font: str,
    size: int,
    weight: int,
    stroke: float,
    shadow: int,
    alignment: str,
    highlight_color: str,
    normal_color: str,
    output_path: str,
) -> None:
    """Generate ASS subtitle from word_timing list."""
    if not word_timing:
        return

    # Map alignment string to ASS alignment number
    align_map = {
        "bottom_center": 2,
        "center": 5,
        "top_center": 8,
    }
    ass_alignment = align_map.get(alignment, 2)

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00{normal_color.lstrip('#')},&H000000FF,&H00000000,&H00000000,{1 if weight >= 600 else 0},0,0,0,100,100,0,0,1,{int(stroke)},{shadow},{ass_alignment},10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for w in word_timing:
        start = w.get("start", 0.0)
        end = w.get("end", 0.0)
        word = w.get("word", "")
        # Check if this word is highlighted
        color = highlight_color if w.get("highlight", False) else normal_color
        # For ASS, we can use a different style or just use color
        # We'll keep it simple: use default style with color tag
        text = f"{{\\c&H{color.lstrip('#')}&}}{word}"
        start_ass = format_time_ass(start)
        end_ass = format_time_ass(end)
        events.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")

    ass_content = ass_header + "\n".join(events)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)

def generate_ass(
    timeline_words_path: str,
    clip_segments: List[int],
    clip_start: float,
    clip_end: float,
    caption_style: Dict[str, Any],
    output_path: str,
) -> None:
    """Legacy function: generate ASS from timeline_with_words.json."""
    with open(timeline_words_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get("segments", [])
    segment_map = {seg.get("segment_id", idx+1): seg for idx, seg in enumerate(segments)}

    all_words = []
    for seg_id in clip_segments:
        seg = segment_map.get(seg_id)
        if seg:
            words = seg.get("words", [])
            for w in words:
                start_abs = w.get("start", 0.0)
                end_abs = w.get("end", 0.0)
                if start_abs >= clip_start and end_abs <= clip_end:
                    all_words.append({
                        "word": w.get("word", ""),
                        "start": start_abs - clip_start,
                        "end": end_abs - clip_start,
                    })

    if not all_words:
        all_words = [{"word": "No word timings available", "start": 0.0, "end": clip_end - clip_start}]

    font = caption_style.get("font", "Arial")
    size = caption_style.get("size", 24)
    color = caption_style.get("color", "FFFFFF")
    bold = caption_style.get("bold", False)
    alignment = caption_style.get("alignment", 2)

    # Use the new function with a basic style
    # We'll convert old style to parameters for the new function
    # For simplicity, we'll just call the new function with a basic config
    # but we need to handle highlight words separately.
    # We'll just produce standard subtitles without highlights.
    word_timing = [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in all_words]
    generate_ass_from_word_timing(
        word_timing,
        font,
        size,
        700 if bold else 400,
        2.0,
        0,
        "bottom_center" if alignment == 2 else "center",
        "#FFFFFF",  # not used for fallback
        color,
        output_path
    )