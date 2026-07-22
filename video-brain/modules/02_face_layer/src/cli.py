"""Command-line interface for face detection."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .detector import FaceDetector
from .models import FaceDetectionResult

app = typer.Typer(
    name="face-layer",
    help="Detect faces in video scenes and output JSON.",
    add_completion=False,
)
console = Console()

@app.command()
def detect(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to input video"),
    scenes: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to scenes.json"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file path."),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Confidence threshold."),
    min_face_size: int = typer.Option(20, "--min-face-size", help="Minimum face size in pixels."),
    sample_interval: float = typer.Option(0.5, "--sample-interval", help="Frame sample interval (seconds)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
) -> None:
    """Detect faces in each scene and save as JSON."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    # Load scenes
    with open(scenes, "r") as f:
        scenes_data = json.load(f)
    # The scenes data is a dict with "scenes" list
    if isinstance(scenes_data, dict) and "scenes" in scenes_data:
        scenes_list = scenes_data["scenes"]
    else:
        scenes_list = scenes_data  # assume list

    detector = FaceDetector(
        threshold=threshold,
        min_face_size=min_face_size,
        sample_interval=sample_interval,
    )

    try:
        console.print(f"[bold]Detecting faces in:[/] {video}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Processing video...", total=len(scenes_list))
            result = detector.detect(str(video), scenes_list)
            progress.update(task, completed=len(scenes_list))

        # Output result
        result_json = result.model_dump_json(indent=2)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                f.write(result_json)
            console.print(f"[green]✓[/] Output written to: [bold]{output}[/]")
        else:
            console.print(result_json)

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        raise typer.Exit(code=1)

def main():
    app()

if __name__ == "__main__":
    main()