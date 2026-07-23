"""Build FFmpeg commands for rendering clips."""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def build_trim_command(input_path: str, start: float, end: float, output_path: str) -> List[str]:
    """Build command to trim a video segment."""
    return [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", input_path,
        "-c", "copy",
        output_path,
    ]

def build_filter_command(
    input_path: str,
    output_path: str,
    subtitles_path: Optional[str] = None,
    zoom_events: List[Dict[str, Any]] = None,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> List[str]:
    """
    Build FFmpeg command with filters: subtitles, zoom, fade.
    Ensures dimensions are even for codec compatibility.
    """
    filters = []

    if subtitles_path and os.path.exists(subtitles_path):
        sub_path = subtitles_path.replace("\\", "/")
        filters.append(f"subtitles='{sub_path}'")

    if zoom_events and len(zoom_events) > 0:
        scale_factor = 1.5
        # Use floor/ceil to ensure even dimensions
        # crop=iw/1.5:ih/1.5, but we need to make sure width and height are even
        # We'll use the expression with bitwise AND to round down to even
        filters.append(
            f"crop=floor(iw/{scale_factor}/2)*2:floor(ih/{scale_factor}/2)*2:"
            f"(iw-floor(iw/{scale_factor}/2)*2)/2:(ih-floor(ih/{scale_factor}/2)*2)/2,"
            f"scale=floor(iw*{scale_factor}/2)*2:floor(ih*{scale_factor}/2)*2"
        )

    if fade_in > 0:
        filters.append(f"fade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        filters.append(f"fade=t=out:st={fade_out}:d={fade_out}")

    if not filters:
        return ["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path]

    filter_str = ",".join(filters)
    return ["ffmpeg", "-y", "-i", input_path, "-vf", filter_str, "-c:a", "copy", output_path]

def concat_clips(clip_paths: List[str], output_path: str) -> List[str]:
    concat_file = "concat_list.txt"
    with open(concat_file, 'w') as f:
        for path in clip_paths:
            f.write(f"file '{path.replace('\\', '/')}'\n")
    return [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_path,
    ]

def add_background_music(video_path: str, music_path: str, output_path: str, volume: float = 0.3) -> List[str]:
    return [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", music_path,
        "-filter_complex",
        f"[0:a]volume=1[a0];[1:a]volume={volume}[a1];[a0][a1]amix=inputs=2:duration=first",
        "-c:v", "copy",
        output_path,
    ]

def normalize_audio(input_path: str, output_path: str) -> List[str]:
    return [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-c:v", "copy",
        output_path,
    ]