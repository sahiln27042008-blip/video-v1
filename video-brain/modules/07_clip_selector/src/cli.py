"""Command-line interface for candidate clip generation."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .selector import ClipSelector
from .config import ClipSelectorConfig

app = typer.Typer(name="clip-selector", help="Generate candidate clips from timeline and metrics.", add_completion=False)
console = Console()

@app.command()
def generate(
    timeline: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to timeline_with_words.json"),
    segment_metrics: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to segment_metrics.json"),
    people: Optional[Path] = typer.Option(None, "--people", help="Path to people.json (optional)"),
    metrics: Optional[Path] = typer.Option(None, "--metrics", help="Path to metrics.json (optional)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file path."),
    top_k: int = typer.Option(20, "--top-k", help="Maximum number of candidates to return."),
    min_duration: float = typer.Option(2.0, "--min-duration", help="Minimum clip duration in seconds."),
    max_duration: float = typer.Option(60.0, "--max-duration", help="Maximum clip duration in seconds."),
    merge_gap: float = typer.Option(1.0, "--merge-gap", help="Max gap between segments to merge (seconds)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
) -> None:
    """Generate candidate clips and save JSON."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        config = ClipSelectorConfig(
            top_k=top_k,
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            merge_gap_seconds=merge_gap,
        )
        selector = ClipSelector(config=config)

        console.print(f"[bold]Generating candidate clips from:[/] {timeline}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Selecting clips...", total=None)
            result = selector.produce_candidates(
                str(timeline),
                str(segment_metrics),
                str(people) if people else None,
                str(metrics) if metrics else None,
            )
            progress.update(task, completed=True)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            console.print(f"[OK] Output written to: [bold]{output}[/]")
        else:
            console.print(result.model_dump_json(indent=2))

        # Print summary table
        table = Table(title=f"Top {len(result.candidates)} Candidates")
        table.add_column("ID", style="dim")
        table.add_column("Start", width=14)
        table.add_column("End", width=14)
        table.add_column("Duration", width=10)
        table.add_column("Score", width=8)
        table.add_column("Person")
        table.add_column("Speaker")
        for c in result.candidates:
            table.add_row(
                str(c.candidate_id),
                c.start,
                c.end,
                f"{c.duration:.1f}s",
                f"{c.score:.1f}",
                c.person or "-",
                c.speaker,
            )
        console.print(table)

    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        raise typer.Exit(code=1)

def main():
    app()

if __name__ == "__main__":
    main()