"""Command-line interface for metrics layer."""

import sys
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .metrics import MetricsCalculator
from .models import MetricsResult

app = typer.Typer(name="metrics-layer", help="Compute metrics from timeline.", add_completion=False)
console = Console()

@app.command()
def compute(
    timeline: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to timeline_with_words.json"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file path for global metrics."),
    segments_output: Optional[Path] = typer.Option(None, "--segments-output", help="Output JSON file path for per-segment metrics."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
) -> None:
    """Compute global and per-segment metrics from timeline and save JSON."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        console.print(f"[bold]Computing metrics from:[/] {timeline}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Calculating metrics...", total=None)
            calculator = MetricsCalculator(timeline)
            global_result = calculator.compute_all()
            segment_result = calculator.compute_segment_metrics()
            progress.update(task, completed=True)

        # Determine output paths
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                f.write(global_result.model_dump_json(indent=2))
            console.print(f"[OK] Global metrics written to: [bold]{output}[/]")
            if segments_output is None:
                segments_output = output.parent / "segment_metrics.json"
        else:
            console.print(global_result.model_dump_json(indent=2))
            if segments_output is None:
                console.print("[dim]No output path provided; not writing segment metrics.[/dim]")
                return

        if segments_output:
            segments_output.parent.mkdir(parents=True, exist_ok=True)
            with open(segments_output, "w", encoding="utf-8") as f:
                json.dump(segment_result, f, indent=2)
            console.print(f"[OK] Segment metrics written to: [bold]{segments_output}[/]")

    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        raise typer.Exit(code=1)

def main():
    app()

if __name__ == "__main__":
    main()