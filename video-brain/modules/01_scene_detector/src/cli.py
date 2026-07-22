"""Command-line interface for scene detection."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .detector import SceneDetector
from .models import SceneDetectionResult

app = typer.Typer(
    name="scene-detector",
    help="Detect scenes in a video and output JSON.",
    add_completion=False,
)
console = Console()


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@app.command()
def detect(
    input: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to input video"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file path. If not provided, prints to console."),
    threshold: float = typer.Option(30.0, "--threshold", "-t", help="Content detection threshold (higher = fewer cuts)."),
    min_scene_len: int = typer.Option(15, "--min-scene-len", "-m", help="Minimum scene length in frames."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
    json_only: bool = typer.Option(False, "--json-only", "-j", help="Output only JSON (no progress or table)."),
) -> None:
    """Detect scenes in a video and save as JSON."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    detector = SceneDetector(
        threshold=threshold,
        min_scene_len=min_scene_len,
    )

    try:
        if not json_only:
            console.print(f"[bold]Detecting scenes in:[/] {input}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console if not json_only else None,
            transient=not json_only,
        ) as progress:
            task = progress.add_task("Processing video...", total=None)
            result = detector.detect(str(input))
            progress.update(task, completed=True)

        # Output result
        result_json = result.model_dump_json(indent=2)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                f.write(result_json)
            if not json_only:
                console.print(f"[OK] Output written to: [bold]{output}[/]")
        else:
            if not json_only:
                console.print("\n[bold]Scene Detection Results[/]\n")
                table = Table(title="Scenes", show_header=True, header_style="bold cyan")
                table.add_column("ID", style="dim", width=6)
                table.add_column("Start", width=14)
                table.add_column("End", width=14)
                table.add_column("Duration", width=12)
                for scene in result.scenes:
                    table.add_row(
                        str(scene.id),
                        scene.start,
                        scene.end,
                        _format_duration(scene.duration),
                    )
                console.print(table)
                console.print(f"\n[bold]Total scenes:[/] {result.scene_count}")
                console.print(f"[bold]Duration:[/] {_format_duration(result.duration)}")
                console.print(f"[bold]FPS:[/] {result.fps:.2f}")
            else:
                # Print raw JSON
                console.print(result_json)

    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()