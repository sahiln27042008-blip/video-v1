"""Conversation processor using WhisperX 3.8.6 pipeline."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

import whisperx
import torch

from .models import ConversationSegment, ConversationResult, Word

logger = logging.getLogger(__name__)


class ConversationProcessor:
    """Process video to extract conversation with speaker attribution using WhisperX."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN must be set for diarization")
        self._model = None
        self._diarize_model = None

    @property
    def model(self):
        if self._model is None:
            self._model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                language=self.language,
            )
        return self._model

    @property
    def diarize_model(self):
        if self._diarize_model is None:
            # WhisperX 3.8.6 uses whisperx.diarize.DiarizationPipeline
            from whisperx.diarize import DiarizationPipeline
            self._diarize_model = DiarizationPipeline(
                token=self.hf_token,
                device=self.device,
            )
        return self._diarize_model

    def extract_audio(self, video_path: str, output_wav: str) -> None:
        """Extract audio from video using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_wav,
            "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def format_time(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def process(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> ConversationResult:
        """Full pipeline: transcribe, align, diarize, assign speakers."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        try:
            self.extract_audio(video_path, audio_path)
            logger.info("Audio extracted to %s", audio_path)

            # Load audio once
            audio = whisperx.load_audio(audio_path)

            # 1. Transcribe
            logger.info("Transcribing...")
            result = self.model.transcribe(audio)
            logger.info("Detected language: %s", result["language"])

            # 2. Align
            logger.info("Loading alignment model...")
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.device,
            )
            logger.info("Running alignment...")
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
            )

            # 3. Diarize
            logger.info("Running diarization...")
            diarize_segments = self.diarize_model(audio_path)
            # Do NOT convert – the diarize_model returns a format assign_word_speakers expects.

            # 4. Assign speakers to transcription segments
            logger.info("Assigning speakers...")
            result = whisperx.assign_word_speakers(
                diarize_segments,
                result,
            )

            segments_out = []
            for seg in result["segments"]:
                speaker = seg.get("speaker", "UNKNOWN")
                words = []
                for w in seg.get("words", []):
                    if "start" not in w or "end" not in w:
                        continue
                    words.append(
                        Word(
                            word=w["word"],
                            start=w["start"],
                            end=w["end"],
                        )
                    )
                segments_out.append(
                    ConversationSegment(
                        start=self.format_time(seg["start"]),
                        end=self.format_time(seg["end"]),
                        speaker=speaker,
                        text=seg["text"].strip(),
                        words=words,
                    )
                )

            result_obj = ConversationResult(
                video=video_path,
                segments=segments_out,
            )

            if output_path:
                Path(output_path).write_text(
                    result_obj.model_dump_json(indent=2),
                    encoding="utf8",
                )

            return result_obj

        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)