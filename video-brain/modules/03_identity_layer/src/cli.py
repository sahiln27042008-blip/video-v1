"""Command-line interface for identity tracking."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .detector import IdentityDetector

app = typer.Typer(name="identity-layer", help="Track identities across scenes.", add_completion=False)
console = Console()


@app.command()
def track(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to input video"),
    faces_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to faces.json"),
    scenes_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to scenes.json"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file path."),
    eps: float = typer.Option(0.5, "--eps", help="DBSCAN epsilon (cosine distance)."),
    min_samples: int = typer.Option(2, "--min-samples", help="DBSCAN min samples."),
    min_confidence: float = typer.Option(0.5, "--min-confidence", help="Min face confidence."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
) -> None:
    """Run identity tracking and save JSON."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    detector = IdentityDetector(
        eps=eps,
        min_samples=min_samples,
        min_confidence=min_confidence
    )

    try:
        console.print(f"[bold]Tracking identities in:[/] {video}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Processing identities...", total=None)
            result = detector.process(str(video), str(faces_json), str(scenes_json))
            progress.update(task, completed=True)

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