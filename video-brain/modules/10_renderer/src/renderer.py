"""Main renderer orchestrator (supports both old and new edit plans)."""

import os
import json
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import (
    RenderConfig, RenderResult,
    ClipEdit, CandidateClip,
    TechnicalEditPlan, Clip, Operation,
    TrimOp, SubtitleOp, ZoomOp, MusicOp, SoundEffectOp, TransitionOp,
)
from .subtitle_generator import generate_ass, generate_ass_from_word_timing, parse_time, format_time_ass
from .ffmpeg_builder import (
    build_trim_command,
    build_filter_command,
    concat_clips,
    add_background_music,
    normalize_audio,
)

logger = logging.getLogger(__name__)


class Renderer:
    """Orchestrates the rendering process."""

    def __init__(self, config: RenderConfig):
        self.config = config
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []

    def log(self, msg: str):
        logger.info(msg)
        self.logs.append(msg)

    def load_json(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def parse_timestamp(self, ts: str) -> float:
        return parse_time(ts)

    def render(self) -> RenderResult:
        """Execute the rendering pipeline."""
        try:
            self.log("Loading editor plan...")
            plan_data = self.load_json(self.config.editor_plan_path)

            # Detect plan type
            if "clips" in plan_data and isinstance(plan_data["clips"], list):
                # New technical plan
                if plan_data["clips"] and "operations" in plan_data["clips"][0]:
                    self.log("Detected new technical_edit_plan format.")
                    return self._render_technical(plan_data)
                else:
                    # Old plan with clip_edits
                    self.log("Detected old edit_plan format.")
                    return self._render_legacy(plan_data)
            elif "clip_edits" in plan_data:
                self.log("Detected old edit_plan format.")
                return self._render_legacy(plan_data)
            else:
                raise ValueError("Unknown editor plan format")

        except Exception as e:
            logger.exception("Render failed")
            return RenderResult(success=False, error=str(e), log=self.logs)
        finally:
            # Clean up temp dir if needed
            pass

    def _render_technical(self, plan_data: dict) -> RenderResult:
        """Render using the new technical_edit_plan.json format."""
        plan = TechnicalEditPlan(**plan_data)
        self.log(f"Loaded technical plan with {len(plan.clips)} clips")

        processed_clips = []
        for clip in plan.clips:
            self.log(f"Processing clip {clip.clip_id}...")
            clip_path = self._process_technical_clip(clip)
            if clip_path:
                processed_clips.append(clip_path)

        if not processed_clips:
            return RenderResult(success=False, error="No clips processed", log=self.logs)

        return self._finalize(processed_clips)

    def _process_technical_clip(self, clip: Clip) -> Optional[str]:
        """Process a single clip from technical plan."""
        clip_id = clip.clip_id
        start = clip.start
        end = clip.end
        duration = end - start

        trimmed_path = self.temp_dir / f"clip_{clip_id}_trimmed.mp4"
        filtered_path = self.temp_dir / f"clip_{clip_id}_filtered.mp4"

        # 1. Trim
        self.log(f"  Trimming from {start} to {end}")
        trim_cmd = build_trim_command(self.config.video_path, start, end, str(trimmed_path))
        self._run(trim_cmd)

        # 2. Apply operations: subtitle, zoom, music, sound_effect, transition
        # For MVP, we handle zoom and subtitle.
        subtitle_path = None
        zoom_ops = [op for op in clip.operations if isinstance(op, ZoomOp)]
        subtitle_ops = [op for op in clip.operations if isinstance(op, SubtitleOp)]

        # Generate subtitles if present
        if subtitle_ops:
            sub_op = subtitle_ops[0]  # use first subtitle op
            self.log(f"  Generating subtitles from word_timing")
            subtitle_file = self.temp_dir / f"clip_{clip_id}_subs.ass"
            generate_ass_from_word_timing(
                sub_op.word_timing,
                sub_op.font,
                sub_op.size,
                sub_op.weight,
                sub_op.stroke,
                sub_op.shadow,
                sub_op.alignment,
                sub_op.highlight_color,
                sub_op.normal_color,
                str(subtitle_file)
            )
            subtitle_path = str(subtitle_file)

        # Apply zoom: use first zoom op if any
        zoom_event = None
        if zoom_ops:
            z = zoom_ops[0]
            zoom_event = {
                "start": z.start,
                "end": z.end,
                "scale": z.parameters.get("scale", 1.0),
                "anchor": z.parameters.get("anchor", "center"),
                "easing": z.parameters.get("easing", "linear"),
            }

        # Apply filters (zoom, subtitles)
        # For technical plan, we ignore fade_in/out from old style
        # We'll use the zoom if scale != 1.0
        self.log(f"  Applying filters (zoom: {zoom_event is not None}, subtitle: {subtitle_path is not None})")
        filter_cmd = build_filter_command(
            str(trimmed_path),
            str(filtered_path),
            subtitle_path,
            [zoom_event] if zoom_event else [],
            fade_in=0.0,
            fade_out=0.0
        )
        self._run(filter_cmd)

        self.log(f"  Clip {clip_id} processed -> {filtered_path}")
        return str(filtered_path)

    def _render_legacy(self, plan_data: dict) -> RenderResult:
        """Render using the old edit_plan format."""
        # Load required files
        if not self.config.candidate_clips_path or not self.config.timeline_words_path:
            return RenderResult(success=False, error="Missing candidate_clips or timeline_words for legacy plan", log=self.logs)

        # Load candidate clips and timeline
        candidate_data = self.load_json(self.config.candidate_clips_path)
        candidates = [CandidateClip(**c) for c in candidate_data.get("candidates", [])]
        candidate_map = {c.candidate_id: c for c in candidates}

        clip_edits = [ClipEdit(**item) for item in plan_data.get("clip_edits", [])]

        processed_clips = []
        for edit in clip_edits:
            candidate = candidate_map.get(edit.clip_id)
            if not candidate:
                self.log(f"Warning: Clip {edit.clip_id} not found in candidates, skipping.")
                continue
            clip_path = self._process_legacy_clip(edit, candidate)
            if clip_path:
                processed_clips.append(clip_path)

        if not processed_clips:
            return RenderResult(success=False, error="No clips processed", log=self.logs)

        return self._finalize(processed_clips)

    def _process_legacy_clip(self, edit: ClipEdit, candidate: CandidateClip) -> Optional[str]:
        # Old processing logic (kept for backward compatibility)
        clip_start = self.parse_timestamp(candidate.start)
        clip_end = self.parse_timestamp(candidate.end)
        clip_id = edit.clip_id
        trimmed_path = self.temp_dir / f"clip_{clip_id}_trimmed.mp4"
        filtered_path = self.temp_dir / f"clip_{clip_id}_filtered.mp4"

        trim_cmd = build_trim_command(self.config.video_path, clip_start, clip_end, str(trimmed_path))
        self._run(trim_cmd)

        subtitle_path = None
        if candidate.segments:
            subtitle_file = self.temp_dir / f"clip_{clip_id}_subs.ass"
            generate_ass(
                self.config.timeline_words_path,
                candidate.segments,
                clip_start,
                clip_end,
                edit.captions,
                str(subtitle_file)
            )
            subtitle_path = str(subtitle_file)

        zoom_events = edit.zoom_events
        fade_in = 0.5 if "Fade In" in edit.editing_style else 0.0
        fade_out = 0.5 if "Fade Out" in edit.editing_style else 0.0

        filter_cmd = build_filter_command(
            str(trimmed_path),
            str(filtered_path),
            subtitle_path,
            zoom_events,
            fade_in,
            fade_out
        )
        self._run(filter_cmd)

        return str(filtered_path)

    def _finalize(self, processed_clips: List[str]) -> RenderResult:
        """Concatenate clips, normalize audio, add music, and output final video."""
        self.log("Concatenating clips...")
        concat_output = self.temp_dir / "concatenated.mp4"
        concat_cmd = concat_clips(processed_clips, str(concat_output))
        self._run(concat_cmd)
        self.log(f"Concatenated to {concat_output}")

        self.log("Normalizing audio...")
        normalized_output = self.temp_dir / "normalized.mp4"
        norm_cmd = normalize_audio(str(concat_output), str(normalized_output))
        self._run(norm_cmd)

        final_video = normalized_output
        if self.config.background_music_path and os.path.exists(self.config.background_music_path):
            self.log("Adding background music...")
            music_output = self.temp_dir / "with_music.mp4"
            music_cmd = add_background_music(
                str(normalized_output),
                self.config.background_music_path,
                str(music_output),
                volume=0.3
            )
            self._run(music_cmd)
            final_video = music_output

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = output_dir / "final_video.mp4"
        shutil.copy(str(final_video), str(final_output_path))
        self.log(f"Final video written to {final_output_path}")

        self.log("Rendering complete.")
        return RenderResult(
            success=True,
            output_video=str(final_output_path),
            log=self.logs,
        )

    def _run(self, cmd: List[str]):
        self.log(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                self.log(result.stdout)
            if result.stderr:
                self.log(result.stderr)
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed with code {e.returncode}")
            self.log(f"stderr: {e.stderr}")
            self.log(f"stdout: {e.stdout}")
            raise