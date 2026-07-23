#!/usr/bin/env python3
"""Master orchestrator for Video Brain pipeline (Modules 01–07)."""

import sys
import os
import json
import subprocess
import time
import datetime
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field
import logging

# Ensure project root is in path for importing models
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Rich for progress and logging
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not installed. Install with: pip install rich")

# Pydantic models from each module
try:
    from video_brain.modules._01_scene_detector.src.models import SceneDetectionResult
    from video_brain.modules._02_face_layer.src.models import FaceDetectionResult
    from video_brain.modules._03_identity_layer.src.models import IdentityResult
    from video_brain.modules._04_conversation_layer.src.models import ConversationResult
    from video_brain.modules._05_timeline_fusion.src.models import TimelineResult
    from video_brain.modules._06_metrics_layer.src.models import MetricsResult
    from video_brain.modules._07_clip_selector.src.models import CandidateClipsResult
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Warning: Could not import Pydantic models. Validation will be skipped.")
    # Define dummy validators
    def dummy_validate(path): return True
    SceneDetectionResult = None
    FaceDetectionResult = None
    IdentityResult = None
    ConversationResult = None
    TimelineResult = None
    MetricsResult = None
    CandidateClipsResult = None

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass
class PipelineConfig:
    input_video: Path
    output_dir: Path
    whisper_model: str = "tiny"
    device: str = "cpu"
    hf_token: Optional[str] = None
    checkpoint_file: Optional[Path] = None
    retry_count: int = 2
    # Derived
    def __post_init__(self):
        self.output_dir = Path(self.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_file is None:
            self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.hf_token = self.hf_token or os.environ.get("HF_TOKEN")

# ------------------------------------------------------------------
# Pipeline Step
# ------------------------------------------------------------------
class PipelineStep:
    def __init__(
        self,
        name: str,
        module_path: str,
        args: List[str],
        output_file: str,
        validation_model: Optional[Type] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.module_path = module_path
        self.args = args
        self.output_file = output_file
        self.validation_model = validation_model
        self.env_vars = env_vars or {}
        self.start_time = None
        self.end_time = None

    def get_cmd(self, config: PipelineConfig) -> List[str]:
        """Build the command list without subcommands."""
        cmd = [sys.executable, str(PROJECT_ROOT / self.module_path)]
        # Replace placeholders in args
        resolved = []
        for arg in self.args:
            if arg == "{video}":
                resolved.append(str(config.input_video))
            elif arg == "{output_dir}":
                resolved.append(str(config.output_dir))
            elif arg.startswith("{output_dir}/"):
                rel = arg.replace("{output_dir}/", "")
                resolved.append(str(config.output_dir / rel))
            else:
                resolved.append(arg)
        cmd.extend(resolved)
        return cmd

    def get_output_path(self, config: PipelineConfig) -> Path:
        """Get the absolute output file path."""
        return config.output_dir / self.output_file

    def is_completed(self, config: PipelineConfig) -> bool:
        """Check if this step is already completed via checkpoint."""
        if config.checkpoint_file and config.checkpoint_file.exists():
            with open(config.checkpoint_file, 'r') as f:
                data = json.load(f)
            return data.get("steps", {}).get(self.name, False)
        return False

    def mark_completed(self, config: PipelineConfig):
        """Mark this step as completed in checkpoint."""
        if config.checkpoint_file:
            data = {}
            if config.checkpoint_file.exists():
                with open(config.checkpoint_file, 'r') as f:
                    data = json.load(f)
            if "steps" not in data:
                data["steps"] = {}
            data["steps"][self.name] = True
            data["last_updated"] = datetime.datetime.now().isoformat()
            with open(config.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)

    def validate_output(self, config: PipelineConfig) -> bool:
        """Validate the output file using Pydantic model."""
        if self.validation_model is None:
            return True
        path = self.get_output_path(config)
        if not path.exists():
            return False
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            # Attempt to parse
            self.validation_model.model_validate(data)
            return True
        except Exception as e:
            logger.error(f"Validation failed for {self.name}: {e}")
            return False

    def run(self, config: PipelineConfig, console) -> bool:
        """Execute this pipeline step."""
        if self.is_completed(config):
            console.print(f"[dim]⏭️  Skipping {self.name} (already completed)[/dim]")
            return True

        output_path = self.get_output_path(config)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self.get_cmd(config)
        env = os.environ.copy()
        env.update(self.env_vars)
        # If conversation step, ensure HF_TOKEN is set
        if "conversation" in self.name.lower() and config.hf_token:
            env["HF_TOKEN"] = config.hf_token

        console.print(f"[bold cyan]▶ Running {self.name}[/bold cyan]")
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

        self.start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
                encoding="utf-8",
                errors="replace"
            )
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time

            if result.returncode != 0:
                console.print(f"[red]❌ {self.name} failed (exit code {result.returncode})[/red]")
                console.print(f"[red]stdout:[/red]\n{result.stdout}")
                console.print(f"[red]stderr:[/red]\n{result.stderr}")
                return False

            console.print(f"[green]✅ {self.name} completed in {elapsed:.2f}s[/green]")

            # Validate output
            if not self.validate_output(config):
                console.print(f"[red]❌ Output validation failed for {self.name}[/red]")
                return False

            self.mark_completed(config)
            return True

        except Exception as e:
            console.print(f"[red]❌ {self.name} raised exception: {e}[/red]")
            return False

# ------------------------------------------------------------------
# Pipeline Definition (no subcommands)
# ------------------------------------------------------------------
def get_pipeline_steps() -> List[PipelineStep]:
    """Define all pipeline steps without subcommands."""
    steps = []

    # Stage 1: Scene Detection
    steps.append(PipelineStep(
        name="Scene Detection",
        module_path="video-brain/modules/01_scene_detector/main.py",
        args=["{video}", "--output", "{output_dir}/scenes.json"],
        output_file="scenes.json",
        validation_model=SceneDetectionResult,
    ))

    # Stage 2: Face Detection
    steps.append(PipelineStep(
        name="Face Detection",
        module_path="video-brain/modules/02_face_layer/main.py",
        args=["{video}", "{output_dir}/scenes.json", "--output", "{output_dir}/faces.json"],
        output_file="faces.json",
        validation_model=FaceDetectionResult,
    ))

    # Stage 3: Identity Tracking
    steps.append(PipelineStep(
        name="Identity Tracking",
        module_path="video-brain/modules/03_identity_layer/main.py",
        args=["{video}", "{output_dir}/faces.json", "{output_dir}/scenes.json", "--output", "{output_dir}/identities.json"],
        output_file="identities.json",
        validation_model=IdentityResult,
    ))

    # Stage 4: Conversation (WhisperX)
    steps.append(PipelineStep(
        name="Conversation (WhisperX)",
        module_path="video-brain/modules/04_conversation_layer/main.py",
        args=["{video}", "--output", "{output_dir}/conversation.json", "--model", "tiny", "--device", "cpu"],
        output_file="conversation.json",
        validation_model=ConversationResult,
        env_vars={"PYTHONUTF8": "1"},
    ))

    # Stage 5: Timeline Fusion
    steps.append(PipelineStep(
        name="Timeline Fusion",
        module_path="video-brain/modules/05_timeline_fusion/main.py",
        args=[
            "{video}",
            "{output_dir}/scenes.json",
            "{output_dir}/faces.json",
            "{output_dir}/identities.json",
            "{output_dir}/conversation.json",
            "--output", "{output_dir}/timeline.json"
        ],
        output_file="timeline.json",
        validation_model=TimelineResult,
    ))

    # Stage 6: Metrics (produces both metrics.json and segment_metrics.json)
    steps.append(PipelineStep(
        name="Metrics",
        module_path="video-brain/modules/06_metrics_layer/main.py",
        args=[
            "{output_dir}/timeline_with_words.json",
            "--output", "{output_dir}/metrics.json",
            "--segments-output", "{output_dir}/segment_metrics.json"
        ],
        output_file="metrics.json",
        validation_model=MetricsResult,
    ))

    # Stage 7: Clip Selector (generates candidate_clips.json)
    steps.append(PipelineStep(
        name="Clip Selector",
        module_path="video-brain/modules/07_clip_selector/main.py",
        args=[
            "{output_dir}/timeline_with_words.json",
            "{output_dir}/segment_metrics.json",
            "--output", "{output_dir}/candidate_clips.json",
            "--top-k", "20"
        ],
        output_file="candidate_clips.json",
        validation_model=CandidateClipsResult,
    ))

    return steps

# ------------------------------------------------------------------
# Main Pipeline Runner
# ------------------------------------------------------------------
def run_pipeline(config: PipelineConfig):
    """Run the full pipeline."""
    console = Console() if RICH_AVAILABLE else None
    if console is None:
        # Fallback to print
        logging.basicConfig(level=logging.INFO)
        console = None

    steps = get_pipeline_steps()

    total_start = time.time()
    if console:
        console.print(Panel.fit("[bold green]Video Brain Pipeline[/bold green]", border_style="green"))

    # Show config
    if console:
        console.print(f"[cyan]Video:[/cyan] {config.input_video}")
        console.print(f"[cyan]Output:[/cyan] {config.output_dir}")
        console.print(f"[cyan]Whisper model:[/cyan] {config.whisper_model}")
        console.print(f"[cyan]Device:[/cyan] {config.device}")
        console.print(f"[cyan]HF Token:[/cyan] {'Set' if config.hf_token else 'Not set'}")
        console.print("")

    success = True
    for step in steps:
        if not step.run(config, console if console else None):
            success = False
            break

    total_elapsed = time.time() - total_start

    if success:
        # Final summary
        if console:
            console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]")
            console.print(f"[bold]Total time:[/bold] {total_elapsed:.2f}s")
            console.print("\n[bold]Artifacts:[/bold]")
            for step in steps:
                out = step.get_output_path(config)
                if out.exists():
                    size = out.stat().st_size
                    console.print(f"  [green]✔[/green] {out.name} ({size} bytes)")
        else:
            print("Pipeline completed successfully.")
            print(f"Total time: {total_elapsed:.2f}s")
            print("Artifacts:")
            for step in steps:
                out = step.get_output_path(config)
                if out.exists():
                    print(f"  {out.name}")
    else:
        if console:
            console.print("[bold red]❌ Pipeline failed.[/bold red]")
        else:
            print("Pipeline failed.")
        sys.exit(1)

# ------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Brain Pipeline (Modules 01-07)")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--whisper-model", default="tiny", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu, cuda)")
    parser.add_argument("--hf-token", help="Hugging Face token for diarization")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input video not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output).resolve()
    if args.no_checkpoint:
        checkpoint_file = None
    else:
        checkpoint_file = output_dir / "checkpoint.json"

    config = PipelineConfig(
        input_video=input_path,
        output_dir=output_dir,
        whisper_model=args.whisper_model,
        device=args.device,
        hf_token=args.hf_token,
        checkpoint_file=checkpoint_file,
    )

    run_pipeline(config)

if __name__ == "__main__":
    main()